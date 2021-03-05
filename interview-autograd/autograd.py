import inspect
import ast


def f(x, y, z):
    x = x + 1
    t = x - y
    u = t * z
    p = u / y
    q = u * y
    return p, q


def parse(func):
    result = {}
    source = inspect.getsource(f)
    module = ast.parse(source)
    function_def = module.body[0]

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


print(parse(f))
