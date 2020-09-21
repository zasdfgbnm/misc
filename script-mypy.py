import collections
import functools

ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('original_method'))


class ScriptMeta(type):
    def __init__(cls, name, bases, attrs):  # noqa: B902
        # Aggregate all the ScriptMethods and constants from superclasses
        cls._methods = {}
        cls._constants_set = set(getattr(cls, "__constants__", ()))
        for base in reversed(bases):
            for k, v in getattr(base, "_methods", {}).items():
                cls._methods[k] = v
            base_constants = getattr(base, "_constants_set", set())
            cls._constants_set = cls._constants_set.union(base_constants)

        # find all the script methods of the current class
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                cls._methods[v.original_method.__name__] = v

        if getattr(cls, "_disable_script_meta", False):
            # We leave built-in ScriptModule types alone, since this metaclass
            # is only for compiling user classes that inherit from
            # ScriptModule.
            return super(ScriptMeta, cls).__init__(name, bases, attrs)

        original_init = getattr(cls, "__init__", lambda self: None)

        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

        cls.__init__ = init_then_script
        return super(ScriptMeta, cls).__init__(name, bases, attrs)


class _CachedForward(object):
    def __get__(self, obj, cls):
        return self.__getattr__("forward")


class ScriptModule(metaclass=ScriptMeta):
    """
    ``ScriptModule``s wrap a C++ ``torch::jit::Module``. ``ScriptModule``s
    contain methods, attributes, parameters, and
    constants. These can be accessed the same as on a normal ``nn.Module``.
    """

    def __init__(self):
        super(ScriptModule, self).__init__()

    forward = _CachedForward()


def script(f):
    return ScriptMethodStub(original_method=f)


class M(ScriptModule):
    @script
    def forward(self, x):
        return x