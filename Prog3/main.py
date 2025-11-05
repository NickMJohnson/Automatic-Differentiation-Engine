import torch
import time

torch.set_grad_enabled(False)

def load_olmo2(device='cpu'):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Olmo2Config
    from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
    repo_id = "allenai/OLMo-2-0425-1B-Instruct"
    state_dict_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    state_dict = load_file(state_dict_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    config = Olmo2Config.from_pretrained(repo_id)
    rot_emb = Olmo2RotaryEmbedding(config)
    (rope_cos_precomputed, rope_sin_precomputed) = rot_emb(torch.zeros(0,dtype=config.torch_dtype,device=device),torch.arange(config.max_position_embeddings,device=device)[None,:])
    rope_cos_precomputed = rope_cos_precomputed.squeeze(0).to(config.torch_dtype)
    rope_sin_precomputed = rope_sin_precomputed.squeeze(0).to(config.torch_dtype)
    head_size = config.hidden_size // config.num_attention_heads
    rope_precomputed = torch.zeros(config.max_position_embeddings, head_size//2, 2, 2, dtype=config.torch_dtype, device=device)
    rope_precomputed[:,:,0,0] = rope_cos_precomputed.view(config.max_position_embeddings, 2, head_size//2)[:,0,:]
    rope_precomputed[:,:,1,0] = -rope_sin_precomputed.view(config.max_position_embeddings, 2, head_size//2)[:,0,:]
    rope_precomputed[:,:,1,1] = rope_cos_precomputed.view(config.max_position_embeddings, 2, head_size//2)[:,1,:]
    rope_precomputed[:,:,0,1] = rope_sin_precomputed.view(config.max_position_embeddings, 2, head_size//2)[:,1,:]
    return (state_dict, tokenizer, config, rope_precomputed)

class BumpAllocatorScope:
    def __init__(self, alloc):
        self.alloc = alloc

    def __enter__(self):
        self.bytes_allocated = self.alloc.bytes_allocated

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.alloc.bytes_allocated = self.bytes_allocated

class BumpAllocator:
    def __init__(self, total_bytes):
        # round up to a multiple of 256 bytes, just to make sure no alignment weirdness
        total_bytes = (((total_bytes - 1) // 256) + 1) * 256
        self.data = torch.empty(total_bytes // 8, dtype=torch.int64).view(torch.int8)
        self.bytes_allocated = 0

    # allocates a new empty tensor from this bump allocator
    def alloc(self, shape, dtype):
        # get the number of bytes of the requested allocation
        num_bytes = torch.empty(shape,device='meta',dtype=dtype).view(-1).view(torch.int8).shape[0]
        # round up to multiple of 256
        rounded_bytes = (((num_bytes - 1) // 256) + 1) * 256
        assert(self.bytes_allocated + rounded_bytes <= self.data.shape[0])
        allocated = self.data[self.bytes_allocated:(self.bytes_allocated+num_bytes)].view(dtype).view(shape)
        self.bytes_allocated += rounded_bytes
        return allocated

    # builds a context manager used to scope bump allocations
    def scope(self):
        return BumpAllocatorScope(self)

class Olmo2Model(torch.nn.Module):
    def __init__(self,
                 config,                 # Olmo2Config describing configuration for this model
                 state_dict,             # dict with weights of Olmo2 model
                 rope_precomputed,       # precomputed coefficients for RoPE
                 cache_seqlen=None,      # the length used for memory preallocated for the KV cache
                 max_inputlen=1          # the maximum number of tokens passed forward through the model at once
                ):
        super().__init__()
        if cache_seqlen is None:
            cache_seqlen = config.max_position_embeddings
        self.cache_seqlen = cache_seqlen
        self.max_inputlen = max_inputlen
        self.config = config
        self.state_dict = state_dict
        head_size = config.hidden_size // config.num_attention_heads
        self.k_cache = torch.zeros(config.num_hidden_layers, config.num_attention_heads, cache_seqlen, head_size, dtype=config.torch_dtype)
        self.v_cache = torch.zeros(config.num_hidden_layers, config.num_attention_heads, cache_seqlen, head_size, dtype=config.torch_dtype)
        self.embeddings = torch.zeros(max_inputlen, config.hidden_size, dtype=config.torch_dtype)
        self.positions = torch.zeros(max_inputlen, dtype=torch.int64)
        self.rope_precomputed = rope_precomputed
        self.rope_matrix = torch.zeros(max_inputlen, head_size//2, 2, 2, dtype=config.torch_dtype)
        self.attn_mask = torch.zeros(max_inputlen, cache_seqlen)
        self.cache_positions = torch.arange(cache_seqlen)
        self.attn_scaling = head_size**-0.5 # be sure to remember to use this when you implement attention!
        self.logits = torch.zeros(config.vocab_size, dtype=config.torch_dtype) # last token logits
        # set bump allocator memory

        bump_allocator_memory = 2 * (config.hidden_size * max_inputlen + 4 * config.hidden_size * config.intermediate_size + config.num_attention_heads * (cache_seqlen * (config.hidden_size // config.num_attention_heads)) * 2
        self.alloc = BumpAllocator(int(bump_allocator_memory))

    # compute the forward pass of the model
    # result has the output embeddings in the self.embeddings field
    #     and the logits of the LAST token in the input in the self.logits field 
    def forward(self, token_ids, positions):
        assert(len(token_ids.shape) == 1) # must be a vector
        assert(len(positions.shape) == 1) # must be a vector
        n_tokens = token_ids.shape[0]
        assert(n_tokens == positions.shape[0]) # number of position IDs must match number of tokens
        assert(n_tokens <= self.max_inputlen)
        # copy input positions into preallocated positions array
        self.positions[:n_tokens].copy_(positions)
        torch.index_select(self.state_dict['model.embed_tokens.weight'], 0, token_ids, out=self.embeddings[:n_tokens,:])
        # set up RoPE matrix
        torch.index_select(self.rope_precomputed, 0, positions, out=self.rope_matrix[:n_tokens])
        # set up attention mask
        torch.le(self.cache_positions[None,:], positions[:,None], out=self.attn_mask[:n_tokens,:]) # 1 if shouldn't be masked, 0 otherwise
        self.attn_mask.neg_()
        self.attn_mask.reciprocal_()
        self.attn_mask.add_(1.0)
        # run each layer of the OLMO model
        for ilayer in range(self.config.num_hidden_layers):
            self.self_attn_(ilayer, n_tokens)
            self.mlp_(ilayer, n_tokens)
        self.layer_norm_(self.embeddings[:n_tokens], self.state_dict['model.norm.weight'])
        torch.mv(state_dict['lm_head.weight'], self.embeddings[n_tokens-1], out=self.logits)

    def layer_norm_(self, x, weights):
        # compute layer norm with weights {weights}
        # this function mutates the tensor x
        assert(len(x.shape) == 2)
        assert(len(weights.shape) == 1)
        assert(x.shape[-1] == weights.shape[-1])
        # mutates x
        with self.alloc.scope():
            x_fp32 = self.alloc.alloc(x.shape, dtype=torch.float32)
            x_square_means = self.alloc.alloc((x.shape[0]), dtype=torch.float32)
            x_fp32.copy_(x)
            x_fp32.square_()
            torch.mean(x_fp32, dim=1, out=x_square_means)
            x_square_means.add_(config.rms_norm_eps)
            x_square_means.rsqrt_()
            x.mul_(x_square_means[:,None])
            x.mul_(weights[None,:])
            # both x_fp32 and x_square_means are freed at the end of this scope block
        return x

    def apply_rope_(self, q_or_k):
        # mutates q_or_k
        # q_or_k must be of shape (num_attention_heads, n_tokens, head_size)
        assert(len(q_or_k.shape) == 3)
        assert(q_or_k.shape[0] == self.config.num_attention_heads)
        n_tokens = q_or_k.shape[1]
        head_size = self.config.hidden_size // self.config.num_attention_heads
        assert(q_or_k.shape[2] == head_size)
        with self.alloc.scope():
            # reshape the embedding dim to apply RoPE
            q_or_k = q_or_k.view(self.config.num_attention_heads, n_tokens, 2, head_size//2, 1).permute(0,1,3,4,2)
            q_or_k_postrope = self.alloc.alloc((self.config.num_attention_heads, n_tokens, 2, head_size//2, 1), dtype=self.config.torch_dtype)
            q_or_k_postrope = q_or_k_postrope.permute(0,1,3,4,2)
            torch.matmul(q_or_k, self.rope_matrix[:n_tokens], out=q_or_k_postrope) # batched matmul
            q_or_k.copy_(q_or_k_postrope)

    def self_attn_(self, ilayer, n_tokens):
        # compute self attention block (+post attention layernorm & residual) for layer {ilayer}
        # this function mutates the embeddings in self.embeddings
        # this function uses memory from the bump allocator self.alloc, and must reset it before returning
        #     (this should happen automatically due to the scope context manager!)
        assert(self.alloc.bytes_allocated == 0) # check allocator is in expected state
        cfg = self.config
        head_size = cfg.hidden_size // cfg.num_attention_heads
        # make some convenient references to the embeddings, positions, & weights from the state dict
        q_norm_weight = self.state_dict[f'model.layers.{ilayer}.self_attn.q_norm.weight']
        q_proj_weight = self.state_dict[f'model.layers.{ilayer}.self_attn.q_proj.weight']
        k_norm_weight = self.state_dict[f'model.layers.{ilayer}.self_attn.k_norm.weight']
        k_proj_weight = self.state_dict[f'model.layers.{ilayer}.self_attn.k_proj.weight']
        v_proj_weight = self.state_dict[f'model.layers.{ilayer}.self_attn.v_proj.weight']
        o_proj_weight = self.state_dict[f'model.layers.{ilayer}.self_attn.o_proj.weight']
        layernorm_weight = self.state_dict[f'model.layers.{ilayer}.post_attention_layernorm.weight']
        embeddings = self.embeddings[:n_tokens,:]
        positions = self.positions[:n_tokens]
        head_size = self.config.hidden_size // self.config.num_attention_heads
        # allocate memory
        with self.alloc.scope():
            q = self.alloc.alloc((cfg.num_attention_heads, n_tokens, head_size), dtype=cfg.torch_dtype)
            k = self.alloc.alloc((cfg.num_attention_heads, n_tokens, head_size), dtype=cfg.torch_dtype)
            v = self.alloc.alloc((cfg.num_attention_heads, n_tokens, head_size), dtype=cfg.torch_dtype)

            q_tmp = self.alloc.alloc((n_tokens, cfg.hidden_size), dtype=cfg.torch_dtype)
            k_tmp = self.alloc.alloc((n_tokens, cfg.hidden_size), dtype=cfg.torch_dtype)
            v_tmp = self.alloc.alloc((n_tokens, cfg.hidden_size), dtype=cfg.torch_dtype)

            # projections
            torch.mm(embeddings, q_proj_weight.t(), out=q_tmp)
            torch.mm(embeddings, k_proj_weight.t(), out=k_tmp)
            torch.mm(embeddings, v_proj_weight.t(), out=v_tmp)

            # reshape
            q.copy_(q_tmp.view(n_tokens, cfg.num_attention_heads, head_size).permute(1,0,2))
            k.copy_(k_tmp.view(n_tokens, cfg.num_attention_heads, head_size).permute(1,0,2))
            v.copy_(v_tmp.view(n_tokens, cfg.num_attention_heads, head_size).permute(1,0,2))

            # norms + RoPE
            self.layer_norm_(q_tmp, q_norm_weight)
            self.layer_norm_(k_tmp, k_norm_weight)
            self.apply_rope_(q)
            self.apply_rope_(k)

            # attention scores
            attn = self.alloc.alloc((cfg.num_attention_heads, n_tokens, n_tokens), dtype=torch.float32)
            torch.bmm(q, k.transpose(1,2), out=attn)
            attn.mul_(self.attn_scaling)

            # safe softmax
            attn_max = attn.amax(dim=-1, keepdim=True)
            attn.sub_(attn_max)
            attn.exp_()
            attn_sum = attn.sum(dim=-1, keepdim=True)
            attn.div_(attn_sum)

            # output context
            ctx = self.alloc.alloc((cfg.num_attention_heads, n_tokens, head_size), dtype=cfg.torch_dtype)
            torch.bmm(attn, v, out=ctx)

            # merge heads
            ctx_merged = self.alloc.alloc((n_tokens, cfg.hidden_size), dtype=cfg.torch_dtype)
            ctx.permute(1,0,2).contiguous().view_as(ctx_merged).copy_(ctx.permute(1,0,2).reshape(n_tokens, cfg.hidden_size))
            torch.mm(ctx_merged, o_proj_weight.t(), out=ctx_merged)

            # residual + layernorm
            embeddings.add_(ctx_merged)
            self.layer_norm_(embeddings, layernorm_weight)

    def mlp_(self, ilayer, n_tokens):
        # compute mlp block (+post mlp layernorm & residual) for layer {ilayer}
        # this function mutates the embeddings in self.embeddings
        # this function uses memory from the bump allocator self.alloc, and must reset it before returning
        #     (this should happen automatically due to the scope context manager!)
        assert(self.alloc.bytes_allocated == 0) # check allocator is in expected state
        assert(self.config.hidden_act == 'silu') # you can assume silu nonlinearity (this checks that's right)
        # make some convenient references to the embeddings and the weights from the state dict
        up_proj_weight = self.state_dict[f'model.layers.{ilayer}.mlp.up_proj.weight']
        gate_proj_weight = self.state_dict[f'model.layers.{ilayer}.mlp.gate_proj.weight']
        down_proj_weight = self.state_dict[f'model.layers.{ilayer}.mlp.down_proj.weight']
        layernorm_weight = self.state_dict[f'model.layers.{ilayer}.post_feedforward_layernorm.weight']
        embeddings = self.embeddings[:n_tokens,:]
        # allocate memory
        with self.alloc.scope():
            # 1) allocate activations
            up_out   = self.alloc.alloc((n_tokens, up_proj_weight.shape[0]), dtype=self.config.torch_dtype)
            gate_out = self.alloc.alloc((n_tokens, gate_proj_weight.shape[0]), dtype=self.config.torch_dtype)

            # 2) linear layers
            torch.mm(embeddings, up_proj_weight.t(),   out=up_out)
            torch.mm(embeddings, gate_proj_weight.t(), out=gate_out)

            # 3) silu + elementwise product
            torch.nn.functional.silu(gate_out, inplace=True)
            up_out.mul_(gate_out)

            # gate_out freed at scope exit
            with self.alloc.scope():
                down_out = self.alloc.alloc((n_tokens, down_proj_weight.shape[0]), dtype=self.config.torch_dtype)
                torch.mm(up_out, down_proj_weight.t(), out=down_out)

            # 4) add & norm
            embeddings.add_(down_out)
            self.layer_norm_(embeddings, layernorm_weight)

# simple greedy decoding; prints result as it's being generated
def generate(prompt, olmo_model, tokenizer, max_tokens_to_generate=128):
    enc = tokenizer.encode(prompt)
    ii = 0
    while ii < len(enc):
        enc_ii = enc[ii:(ii+olmo_model.max_inputlen)]
        input_tokens = torch.tensor(enc_ii)
        input_positions = torch.arange(len(enc_ii)) + ii
        olmo_model(input_tokens, input_positions)
        ii += olmo_model.max_inputlen
    a = olmo_model.logits.argmax().item() # argmax is token ID of most likely continuation
    for ii in range(max_tokens_to_generate):
        print(f'{tokenizer.decode(a)}', end='')
        input_tokens = torch.tensor([a])
        input_positions = torch.tensor([ii + len(enc)])
        olmo_model(input_tokens, input_positions)
        a = olmo_model.logits.argmax().item()
        if a == enc[0]: # enc[0] must be the <|endoftext|> token for this to make sense
            break

if __name__ == "__main__":
    device = 'cpu'
    (state_dict, tokenizer, config, rope_precomputed) = load_olmo2(device=device)
    prompt = '<|endoftext|><|user|>\nWhat is 13+5?\n<|assistant|>\n'
    max_inputlen = 8
    with torch.device(device):
        olmo_model = Olmo2Model(config, state_dict, rope_precomputed, max_inputlen = max_inputlen)
        generate(prompt, olmo_model, tokenizer)
        # we can also time the forward pass
        # be sure that when you modify this to use cuda, we call torch.cuda.synchronize
        input_tokens = torch.zeros(max_inputlen, dtype=torch.int64) # dummy data
        input_positions = torch.arange(max_inputlen)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        olmo_model(input_tokens, input_positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        stop = time.time()
        print(f'\nTime for a single forward with {max_inputlen} tokens: {stop - start} seconds')
        input_tokens = torch.zeros(1, dtype=torch.int64) # dummy data
        input_positions = torch.arange(1)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        olmo_model(input_tokens, input_positions)
        if device == 'cuda':
            torch.cuda.synchronize()
        stop = time.time()
        print(f'Time for a single forward with 1 token: {stop - start} seconds')
