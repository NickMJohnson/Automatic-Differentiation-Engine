
import numpy as np

### a function to create a unique increasing ID
### note that this is just a quick-and-easy way to create a global order
### it's not the only way to do it
global_order_counter = 0
def get_next_order():
    global global_order_counter
    rv = global_order_counter
    global_order_counter = global_order_counter + 1
    return rv

### a helper function to convert constants into BackproppableArray objects
def to_ba(x):
    if isinstance(x, BackproppableArray):
        return x
    elif isinstance(x, np.ndarray):
        return BackproppableArray(x)
    elif isinstance(x, float):
        return BackproppableArray(np.array(x))
    elif isinstance(x, int):
        return BackproppableArray(np.array(float(x)))
    else:
        raise Exception("could not convert {} to BackproppableArray".format(x))

### a class for an array that can be "packpropped-through"
class BackproppableArray(object):
    # np_array     numpy array that stores the data for this object
    def __init__(self, np_array, dependencies=[]):
        super().__init__()
        self.data = np_array

        # grad holds the gradient, an array of the same shape as data
        # before backprop, grad is None
        # during backprop before grad_fn is called, grad holds the partially accumulated gradient
        # after backprop, grad holds the gradient of the loss (the thing we call backward on)
        #     with respect to this array
        # if you want to use the same array object to call backward twice, you need to re-initialize
        #     grad to zero first
        self.grad = None

        # an counter that increments monotonically over the course of the application
        # we know that arrays with higher order must depend only on arrays with lower order
        # we can use this to order the arrays for backpropagation
        self.order = get_next_order()

        # a list of other BackproppableArray objects on which this array directly depends
        # we'll use this later to decide which BackproppableArray objects need to participate in the backward pass
        self.dependencies = dependencies

    # represents me as a string
    def __repr__(self):
        return "({}, type={})".format(self.data, type(self).__name__)

    # returns a list containing this array and ALL the dependencies of this array, not just
    #    the direct dependencies listed in self.dependencies
    # that is, this list should include this array, the arrays in self.dependencies,
    #     plus all the arrays those arrays depend on, plus all the arrays THOSE arrays depend on, et cetera
    # the returned list must only include each dependency ONCE
    def all_dependencies(self):
        # Boiler plate BFS for finding all dependencies, use of id for hashability/object comparison 
        visited = set()
        queue = [self]
        all_deps = []
        while queue:
            current = queue.pop(0)
            if id(current) in visited:
                continue
            visited.add(id(current))
            all_deps.append(current)
            for dep in current.dependencies:
                if id(dep) not in visited:
                    queue.append(dep)
        return all_deps

    # compute gradients of this array with respect to everything it depends on
    def backward(self):
        # can only take the gradient of a scalar
        assert(self.data.size == 1)

        # depth-first search to find all dependencies of this array
        all_my_dependencies = self.all_dependencies()

        #   (1) sort the found dependencies so that the ones computed last go FIRST
        sorted_dependencies = sorted(all_my_dependencies, key=lambda x: x.order, reverse=True)
        #   (2) initialize and zero out all the gradient accumulators (.grad) for all the dependencies
        for dep in sorted_dependencies:
            dep.grad = np.zeros_like(dep.data)
        #   (3) set the gradient accumulator of this array to 1, as an initial condition
        #           since the gradient of a number with respect to itself is 1'
        self.grad = np.ones_like(self.data)
        #   (4) call the grad_fn function for all the dependencies in the sorted reverse order
        for dep in sorted_dependencies:
            dep.grad_fn()

    # function that is called to process a single step of backprop for this array
    # when called, it must be the case that self.grad contains the gradient of the loss (the
    #     thing we are differentating) with respect to this array
    # this function should update the .grad field of its dependencies
    #
    # this should just say "pass" for the parent class
    #
    # child classes override this
    def grad_fn(self):
        pass

    # operator overloading
    def __add__(self, other):
        return BA_Add(self, to_ba(other))
    def __sub__(self, other):
        return BA_Sub(self, to_ba(other))
    def __mul__(self, other):
        return BA_Mul(self, to_ba(other))
    def __truediv__(self, other):
        return BA_Div(self, to_ba(other))

    def __radd__(self, other):
        return BA_Add(to_ba(other), self)
    def __rsub__(self, other):
        return BA_Sub(to_ba(other), self)
    def __rmul__(self, other):
        return BA_Mul(to_ba(other), self)
    def __rtruediv__(self, other):
        return BA_Div(to_ba(other), self)

# (2.2) Adding operator overloading for matrix multiplication

    def __matmul__(self, other):
        return BA_MatMul(self, to_ba(other))

    def __rmatmul__(self, other):
        return BA_MatMul(to_ba(other), self)

    def sum(self, axis=None, keepdims=True):
        return BA_Sum(self, axis)

    def reshape(self, shape):
        return BA_Reshape(self, shape)

    def transpose(self, axes = None):
        if axes is None:
            axes = range(self.data.ndim)[::-1]
        return BA_Transpose(self, axes)

# Implementing any helper functions you'll need to backprop through vectors
def _unbroadcast_to(grad, shape):
    g = grad
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, (gdim, sdim) in enumerate(zip(g.shape, shape)):
        if sdim == 1 and gdim != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)

# a class for an array that's the result of an addition operation
class BA_Add(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data + y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        
        # (1.3) improve grad fn for Add
        # self.x.grad += self.grad
        # self.y.grad += self.grad

        # (2.3) improve grad fn for Add: broadcasting-aware accumulation
        self.x.grad += _unbroadcast_to(self.grad, self.x.data.shape)
        self.y.grad += _unbroadcast_to(self.grad, self.y.data.shape)

# a class for an array that's the result of a subtraction operation
class BA_Sub(BackproppableArray):
    # x - y
    def __init__(self, x, y):
        super().__init__(x.data - y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):

        # (1.3) implement grad fn for Sub
        # d(x-y)/dx = 1
        # self.x.grad += self.grad
        # d(x-y)/dy = -1
        # self.y.grad -= self.grad

        # (2.3) improve grad fn for Sub: broadcasting-aware accumulation
        self.x.grad += _unbroadcast_to(self.grad, self.x.data.shape)
        self.y.grad += _unbroadcast_to(-self.grad, self.y.data.shape)

# a class for an array that's the result of a multiplication operation
class BA_Mul(BackproppableArray):
    # x * y
    def __init__(self, x, y):
        super().__init__(x.data * y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
    
        # (1.3) implement grad fn for Mul
        # d(x*y)/dx = y
        # self.x.grad += self.y.data * self.grad
        # d(x*y)/dy = x
        # self.y.grad += self.x.data * self.grad

        # (2.3) improve grad fn for Multiplication: broadcasting-aware accumulation
        self.x.grad += _unbroadcast_to(self.grad * self.y.data, self.x.data.shape)
        self.y.grad += _unbroadcast_to(self.grad * self.x.data, self.y.data.shape)

# a class for an array that's the result of a division operation
class BA_Div(BackproppableArray):
    # x / y
    def __init__(self, x, y):
        super().__init__(x.data / y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # (1.3) implement grad fn for Div
        # d(x/y)/dx = (1/y)
        # self.x.grad += self.grad / self.y.data
        # d(x/y)/dy = (-x/y^2)
        # self.y.grad += self.grad * (-self.x.data / (self.y.data ** 2))

        # (2.3) improve grad fn for Division: broadcasting-aware accumulation
        self.x.grad += _unbroadcast_to(self.grad / self.y.data, self.x.data.shape)
        self.y.grad += _unbroadcast_to(-self.grad * (self.x.data / (self.y.data ** 2)), self.y.data.shape)


# a class for an array that's the result of a matrix multiplication operation
class BA_MatMul(BackproppableArray):
    # x @ y
    def __init__(self, x, y):
        # we only support multiplication of matrices, i.e. arrays with shape of length 2
        assert(len(x.data.shape) == 2)
        assert(len(y.data.shape) == 2)
        super().__init__(x.data @ y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # (2.1) implement grad fn for MatMul
        self.x.grad += self.grad @ self.y.data.T
        self.y.grad += self.x.data.T @ self.grad
    
pass


# a class for an array that's the result of an exponential operation
class BA_Exp(BackproppableArray):
    # exp(x)
    def __init__(self, x):
        super().__init__(np.exp(x.data), [x])
        self.x = x

    def grad_fn(self):
        # (1.3) implement grad fn for Exp
        # d(exp(x))/dx = exp(x)
        self.x.grad += np.exp(self.x.data)* self.grad

def exp(x):
    if isinstance(x, BackproppableArray):
        return BA_Exp(x)
    else:
        return np.exp(x)

# a class for an array that's the result of an logarithm operation
class BA_Log(BackproppableArray):
    # log(x)
    def __init__(self, x):
        super().__init__(np.log(x.data), [x])
        self.x = x

    def grad_fn(self):
        # (1.3) implement grad fn for Log
        # d(log(x))/dx = 1/x
        self.x.grad += (1.0 / self.x.data) * self.grad

def log(x):
    if isinstance(x, BackproppableArray):
        return BA_Log(x)
    else:
        return np.log(x)

# TODO: Add your own function
# END TODO

# a class for an array that's the result of a sum operation
class BA_Sum(BackproppableArray):
    # x.sum(axis, keepdims=True)
    def __init__(self, x, axis):
        super().__init__(x.data.sum(axis, keepdims=True), [x])
        self.x = x
        self.axis = axis

    def grad_fn(self):
        # (2.1) implement grad fn for Sum
        self.x.grad += np.broadcast_to(self.grad, self.x.data.shape)

# a class for an array that's the result of a reshape operation
class BA_Reshape(BackproppableArray):
    # x.reshape(shape)
    def __init__(self, x, shape):
        super().__init__(x.data.reshape(shape), [x])
        self.x = x
        self.shape = shape

    def grad_fn(self):
        # (2.1) implement grad fn for Reshape
        self.x.grad += self.grad.reshape(self.x.data.shape)

# a class for an array that's the result of a transpose operation
class BA_Transpose(BackproppableArray):
    # x.transpose(axes)
    def __init__(self, x, axes):
        super().__init__(x.data.transpose(axes), [x])
        self.x = x
        self.axes = axes

    def grad_fn(self):
        # (2.1) implement grad fn for Transpose
        if self.axes is None:
            self.x.grad += self.grad.transpose()
        else:
            inv = np.argsort(self.axes)
            self.x.grad += self.grad.transpose(tuple(inv))


# numerical derivative of scalar function f at x, using tolerance eps
def numerical_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps))/(2*eps)

def numerical_grad(f, x, eps=1e-5):
    # TODO: (2.5) implement numerical gradient function
    #       this should compute the gradient by applying something like
    #       numerical_diff independently for each entry of the input x
    pass

# automatic derivative of scalar function f at x, using backprop
def backprop_diff(f, x):
    ba_x = to_ba(x)
    fx = f(ba_x)
    fx.backward()
    return ba_x.grad



# class to store test functions

class TestFxs(object):
    # scalar-to-scalar tests
    @staticmethod
    def f1(x):
        return x * 2 + 3

    @staticmethod
    def df1dx(x):
        return 2.0

    @staticmethod
    def f2(x):
        return x * x

    @staticmethod
    def df2dx(x):
        return 2.0 * x

    @staticmethod
    def f3(x):
        u = (x - 2.0)
        return u / (u*u + 1.0)

    @staticmethod
    def df3dx(x):
        u = x - 2.0
        return (1.0 - u*u) / ((u*u + 1.0)**2)

    @staticmethod
    def f4(x):
        return log(exp(x*x / 8 - 3*x + 5) + x)

    # scalar-to-scalar tests that use vectors in the middle
    @staticmethod
    def g1(x):
        a = np.ones(3,dtype="float64")
        ax = x + a
        return (ax*ax).sum().reshape(())

    @staticmethod
    def g2(x):
        a = np.ones((4,5),dtype="float64")
        b = np.arange(20,dtype="float64")
        ax = x - a
        bx = log((x + b)*(x + b)).reshape((4,5)).transpose()
        y = bx @ ax
        return y.sum().reshape(())

    # vector-to-scalar tests
    @staticmethod
    def h1(x):  # takes an input of shape (5,)
        b = np.arange(5,dtype="float64")
        xb = x * b - 4
        return (xb * xb).sum().reshape(())
    

if __name__ == "__main__":
    
    """
    Runs the provided Part 1 checks (f1..f4) & the Part 2 checks (g1, g2).
    - Part 1 compares: symbolic vs backprop vs numerical_diff (for f1..f3) and
      numerical_diff vs backprop (for f4).
    - Part 2 (sub-part 2.4) compares: numerical_diff vs backprop on g1 and g2.

    Notes:
    * g1 and g2 are scalar to scalar functions that use vectors/matrices internally
      (exercising broadcasting and @, transpose, reshape).
    * For g2, x values where (x + b) == 0 for some integer b, are avoided because log(0) is undefined. 
    """
    
    print("Testing backprop implementation.\n")

    # ---------- Part 1: the original scalar tests ----------
    
    test_points_part1 = [0.0, 1.0, 2.0, -1.0, 3.5]
    test_functions_part1 = [
        ("f1", TestFxs.f1, TestFxs.df1dx),
        ("f2", TestFxs.f2, TestFxs.df2dx),
        ("f3", TestFxs.f3, TestFxs.df3dx),
        ("f4", TestFxs.f4, None),  # numerical vs backprop only
    ]

    for func_name, func, symbolic_deriv in test_functions_part1:
        for x in test_points_part1:
            num_deriv  = numerical_diff(func, x)      # the finite-diff (scalar x)
            auto_deriv = backprop_diff(func, x)       # the autodiff (scalar x)

            # numerical ~ backprop
            assert np.allclose(num_deriv, auto_deriv, atol=1e-5), \
                f"{func_name}({x}): numerical={num_deriv}, auto={auto_deriv}"

            # (f1...f3) also check symbolic ~ backprop
            if symbolic_deriv is not None:
                sym_deriv = symbolic_deriv(x)
                assert np.allclose(sym_deriv, auto_deriv, atol=1e-10), \
                    f"{func_name}({x}): symbolic={sym_deriv}, auto={auto_deriv}"

    print("All Part 1 tests passed \n")

    # ---------- Part 2: (especially 2.4) numerical vs backprop on g1, g2 ----------
    
    # g1/g2 are scalar to scalar but include vectors/matrices internally:
    # g1: tests elementwise ops + broadcasting through a sum
    # g2: tests broadcasting + reshape + transpose + @ + sum
    #
    # Choosing 'safe' test points for g2 to avoid log(0) at (x + b) == 0:
    
    test_points_part2 = [0.3, 1.1, 2.7, -0.8]  

    test_functions_part2 = [
        ("g1", TestFxs.g1),  # numerical vs backprop
        ("g2", TestFxs.g2),  # numerical vs backprop
    ]

    for func_name, func in test_functions_part2:
        for x in test_points_part2:
            num_deriv  = numerical_diff(func, x)     # central finite difference at x
            auto_deriv = backprop_diff(func, x)      # backprop at x
            assert np.allclose(num_deriv, auto_deriv, atol=1e-5), \
                f"{func_name}({x}): numerical={num_deriv}, auto={auto_deriv}"

    print("Part 2 (i.e., 2.4) tests (g1, g2) passed")

    print("\n All tests are passed")
