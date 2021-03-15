class Number:
    def __init__(self, num, requires_grad=False):
        self.num = num
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = Number(0, requires_grad=False)
        else:
            self.grad = None

        self.leaves = None
        self.op = None

    def zero_(self):
        self.num = 0
        return self

    def zero_grad(self):
        if self.requires_grad:
            self.grad = Number(0)
        if self.leaves is not None:
            for leaf in self.leaves:
                leaf.zero_grad()

    def requires_grad_(self, requires_grad=True):
        if self.requires_grad is requires_grad:
            return self

        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = Number(0, requires_grad=False)
        else:
            self.grad = None

        return self

    def __str__(self):
        return f'Number[{self.num}, {self.grad}]'

    def __eq__(self, that):
        if isinstance(that, Number):
            return self.num == that.num
        elif isinstance(that, (int, float)):
            return self.num == that

    def __add__(self, that):
        assert isinstance(that, Number)

        res = Number(self.num + that.num, requires_grad=True)
        res.leaves = [self, that]
        res.op = 'add'
        return res

    def __sub__(self, that):
        assert isinstance(that, Number)

        res = Number(self.num - that.num, requires_grad=True)
        res.leaves = [self, that]
        res.op = 'sub'
        return res

    def __mul__(self, that):
        assert isinstance(that, Number)

        res = Number(self.num * that.num, requires_grad=True)
        res.leaves = [self, that]
        res.op = 'mul'
        return res

    def __truediv__(self, that):
        assert isinstance(that, Number)
        if that.num == 0:
            raise ZeroDivisionError('that.num is zero')

        res = Number(self.num / that.num, requires_grad=True)
        res.leaves = [self, that]
        res.op = 'truediv'
        return res

def autograd(outp, inp, grad_output=1):
    inp.requires_grad_()
    return _autograd(outp, inp, grad_output)

def _autograd(outp, inp, grad_output):
    assert outp.requires_grad
    assert isinstance(outp, Number)
    assert isinstance(inp, Number)
    outp: Number
    inp: Number
    outp.grad.num = grad_output
    outp.grad.requires_grad_()

    if outp is inp:
        return Number(grad_output)
    if outp.op is None:
        return

    elif outp.op == 'add':
        if outp.leaves[0].requires_grad:
            outp.leaves[0].grad += outp.grad
        if outp.leaves[1].requires_grad:
            outp.leaves[1].grad += outp.grad
        
    elif outp.op == 'sub':
        if outp.leaves[0].requires_grad:
            outp.leaves[0].grad += outp.grad
        if outp.leaves[1].requires_grad:
            outp.leaves[1].grad -= outp.grad

    elif outp.op == 'mul':
        if outp.leaves[0].requires_grad:
            outp.leaves[0].grad += outp.grad * outp.leaves[1]
        if outp.leaves[1].requires_grad:
            outp.leaves[1].grad += outp.grad * outp.leaves[0]
    
    elif outp.op == 'truediv':
        if outp.leaves[0].requires_grad:
            outp.leaves[0].grad += outp.grad * Number(1.0) / outp.leaves[1]
        if outp.leaves[1].requires_grad:
            outp.leaves[1].grad += \
                outp.grad * Number(-1.0) * outp.leaves[0] / outp.leaves[1] / outp.leaves[1]
    
    else:
        raise RuntimeError('unknown operator')
    
    has_output = False
    for leaf in outp.leaves:
        if leaf.requires_grad:
            if _autograd(leaf, inp, leaf.grad.num) is not None:
                has_output = True
    
    return inp.grad if has_output else None


def test1():
    a = Number(3)
    b = Number(4)

    c = a + b
    assert c == 7, c
    c.zero_grad()
    ag = autograd(c, a); assert ag == 1, ag
    c.zero_grad()
    bg = autograd(c, b); assert bg == 1, bg

    d = a / b
    assert d == 3 / 4, d
    d.zero_grad()
    ag = autograd(d, a); assert ag == 1 / 4, ag
    d.zero_grad()
    bg = autograd(d, b); assert bg == - 3 / 4 / 4, bg

    # print()
test1()

def test2():
    a = Number(3)
    b = Number(4)
    c = Number(5)
    d = Number(6)
    e = Number(7)
    f = Number(2)
    # nums = Numbers(a, b, c, d, e, f)

    s = (a + b * c - d / e) * f
    assert s == 44.285714285714285, s
    s.zero_grad()
    ag = autograd(s, a); assert ag == 2, ag
    s.zero_grad()
    bg = autograd(s, b); assert bg == 10, bg
    s.zero_grad()
    cg = autograd(s, c); assert cg == 8, cg
    s.zero_grad()
    dg = autograd(s, d); assert dg == - 2 / 7, dg
    s.zero_grad()
    eg = autograd(s, e); assert eg == 6 * 2 / 7 / 7, eg
    s.zero_grad()
    fg = autograd(s, f); assert fg == 3 + 4 * 5 - 6 / 7, fg
test2()

def test3():
    a = Number(3)

    s1 = a + a
    assert s1 == 6
    s1.zero_grad()
    ag = autograd(s1, a); assert ag == 2, ag

    s2 = a * a
    assert s2 == 9
    s2.zero_grad()
    ag2 = autograd(s2, a); assert ag2 == 6, ag2

    b = Number(2)

    s3 = (a + b) * a
    assert s3 == 15, s3
    s3.zero_grad()
    ag3 = autograd(s3, a); assert ag3 == 8, ag3
    s3.zero_grad()
    bg = autograd(s3, b); assert bg == 3, bg
test3()

def test4():
    def f(a, b):
        return a * a * b + b * b

    a = Number(4)
    b = Number(2)

    s = f(a, b)
    assert s == f(a.num, b.num), s
    s.zero_grad()
    ag = autograd(s, a); assert ag == 2 * a.num * b.num, ag
    s.zero_grad()
    bg = autograd(s, b); assert bg == a.num ** 2 + 2 * b.num, bg

    print(s, a, sep='\n')
    s.zero_grad()
    agg = autograd(ag, a)
    print(s, a, agg, sep='\n')

test4()