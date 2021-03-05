from collections import defaultdict, deque
import inspect
import ast
import textwrap
import numbers


# single op, SSA, maybe dead code
# variable names do not start with 'd', or 'grad'
def f(x, y, z):
    t = x - y
    u = t * z
    p = u / y
    r = p + 1
    q = r + y
    m = p + q
    return p, q


def parse(func):
    result = {}
    source = inspect.getsource(func)
    module = ast.parse(source)
    function_def = module.body[0]

    result['name'] = function_def.name

    args = [x.arg for x in function_def.args.args]
    result['args'] = args

    body = []
    for stmt in function_def.body[:-1]:
        target = stmt.targets[0].id
        value = stmt.value

        def unpack(x):
            if isinstance(x, ast.Name):
                return x.id
            assert isinstance(x, ast.Constant)
            return x.value

        left = unpack(value.left)
        right = unpack(value.right)
        ast_to_str = {
            ast.Add: '+',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Sub: '-',
        }
        op = ast_to_str[type(value.op)]
        body.append((target, '=', left, op, right))
    result['body'] = body

    return_ = function_def.body[-1].value
    if isinstance(return_, ast.Name):
        return_ = (return_,)
    else:
        assert isinstance(return_, ast.Tuple)
        return_ = tuple(x.id for x in return_.elts)
    result['return'] = return_
    return result


parsed = parse(f)
print(parsed)


def autograd(func):
    args = tuple(f'grad_{x}' for x in func['return']) + tuple(
        x for x in func['args']) + tuple(x for x in func['return'])
    def_line = f"def grad_{func['name']}({', '.join(args)}):"

    # get forward and backward graph
    variables = {}
    grads = defaultdict(lambda: [], {x: [f'grad_{x}'] for x in func['return']})
    for out, eq, left, op, right in reversed(func['body']):
        assert eq == "="
        variables[out] = (left, op, right)
        if out not in grads:
            print('Dead code detected:', f'{out} = {left} {op} {right}')
            continue
        if op == '+':
            for x in (left, right):
                if isinstance(x, str):
                    grads[x].append(f'd{out}')
        elif op == '-':
            if isinstance(left, str):
                grads[left].append(f'd{out}')
            if isinstance(right, str):
                grads[right].append(('-', f'd{out}'))
        elif op == '*':
            for x, y in ((left, right), (right, left)):
                if isinstance(x, str):
                    grads[x].append((f'd{out}', '*', y))
        else:
            assert op == '/'
            if isinstance(left, str):
                grads[left].append((f'd{out}', '/', right))
            if isinstance(right, str):
                grads[right].append(
                    ('-', ((f'd{out}', '*', left), '/', (right, '*', right))))

    # emit backward code
    pending = deque([f"d{x}" for x in func["args"]])
    done = {f'grad_{x}' for x in func['return']} | {
        x for x in func["args"]} | {x for x in func['return']}
    rev_code_lines = [f'return {", ".join(pending)}']

    def emit_expr(expr):
        if isinstance(expr, numbers.Number):
            return str(expr)
        if isinstance(expr, str):
            if expr not in done:
                pending.append(expr)
            return expr
        assert isinstance(expr, tuple)
        if len(expr) == 2:
            assert expr[0] == '-'
            return f'(- {emit_expr(expr[1])})'
        assert len(expr) == 3
        left, op, right = expr
        return f'({emit_expr(left)} {op} {emit_expr(right)})'

    def emit_exprs(exprs):
        exprs = [emit_expr(e) for e in exprs]
        return ' + '.join(exprs)

    while len(pending) > 0:
        next = pending.popleft()
        if next in done:
            continue
        if next.startswith('d'):
            exprs = grads[next[1:]]
        else:
            exprs = [variables[next]]
        code = f'{next} = {emit_exprs(exprs)}'
        rev_code_lines.append(code)

    # deduplicate and revert
    code_lines = []
    known = set()
    for line in reversed(rev_code_lines):
        var = line.split('=')[0].strip()
        if var in known:
            continue
        known.add(var)
        code_lines.append(line)
    code = "\n".join(code_lines)
    return f'{def_line}\n{textwrap.indent(code, "    ")}'


backward = autograd(parsed)
print(backward)
exec(backward)


x = 1
y = 2
z = 3
delta = 0.001
p, q = f(x, y, z)
p1, q1 = f(x + delta, y, z)
p2, q2 = f(x - delta, y, z)
dpdx = (p1 - p2) / (2 * delta)
dqdx = (q1 - q2) / (2 * delta)
p1, q1 = f(x, y + delta, z)
p2, q2 = f(x, y - delta, z)
dpdy = (p1 - p2) / (2 * delta)
dqdy = (q1 - q2) / (2 * delta)
p1, q1 = f(x, y, z + delta)
p2, q2 = f(x, y, z - delta)
dpdz = (p1 - p2) / (2 * delta)
dqdz = (q1 - q2) / (2 * delta)
numerical_jacobian = [
    [dpdx, dpdy, dpdz],
    [dqdx, dqdy, dqdz]
]
dpdx, dpdy, dpdz = grad_f(1, 0, x, y, z, p, q)
dqdx, dqdy, dqdz = grad_f(0, 1, x, y, z, p, q)
autograd_jacobian = [
    [dpdx, dpdy, dpdz],
    [dqdx, dqdy, dqdz]
]

print(numerical_jacobian)
print(autograd_jacobian)
