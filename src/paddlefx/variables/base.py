from __future__ import annotations

import inspect
import itertools
import operator
import types

from typing import TYPE_CHECKING, Any, Callable

from ..proxy import Proxy
from ..source import LocalSource, Source

_sym_var_id_counter = itertools.count()

if TYPE_CHECKING:
    from ..pyeval import PyEvalBase


class VariableBase:
    def __init__(
        self,
        *,
        var: Any = None,
        vtype: Any = None,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ) -> None:
        self.var = var
        self.vtype = vtype if var is None else type(var)
        self.tx = tx
        self.source = source
        self.node = node

        self.id = f"id_{next(_sym_var_id_counter)}"

    def __str__(self) -> str:
        # TODO: just workaround, rm it later
        if self.source is not None and isinstance(self.source, LocalSource):
            return self.source.local_name
        elif self.node is not None:
            return self.node.name

        return f"VariableBase({self.vtype}, {self.id})"

    def __repr__(self) -> str:
        return self.__str__()


class ConstantVariable(VariableBase):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        return str(self.value)


class CallableVariable(VariableBase):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __str__(self):
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__
        return f"{self.__class__.__name__}({name})"

    def __call__(self, tx: PyEvalBase, *args, **kwargs) -> Any:
        # TODO: better org
        assert isinstance(self.var, Callable)
        var = self.var
        graph = tx.output.graph

        if var.__module__.startswith("paddle"):
            # TODO: support multiple ouputs and containers
            ot = args[0].vtype
            output = graph.call_function(var, args, kwargs, ot)
            return VariableBase(vtype=ot, node=output)
        elif inspect.isbuiltin(var):
            if var is print:
                raise NotImplementedError("print() is not supported")
            elif var is getattr:
                object, name = args
                attr = getattr(object.var, name.var)
                return VariableBase(var=attr)
            elif var in [operator.add, operator.sub]:
                ot = args[0].vtype
                output = graph.call_function(var, args, kwargs, ot)
                return VariableBase(vtype=ot, node=output)
            elif var in [operator.gt]:
                ot = args[0].vtype
                output = graph.call_function(var, args, kwargs, ot)
                return VariableBase(vtype=ot, node=output)
            else:
                raise NotImplementedError(f"builtin {var} is not supported")

        return tx.inline_call_function(self, args, kwargs)

    def call_function(
        self,
        translator,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> VariableBase:
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        fn_name = self.fn.__name__
        handler = getattr(self, f"call_{fn_name}", None)
        if handler:
            return handler(translator, *args, **kwargs)
        return ObjectVariable(
            translator.output.create_node('call_function', self.fn, args, kwargs)
        )
        # raise NotImplementedError(f"{fn_name} is not implemented now")

    def call_add(
        self,
        translator,
        a: ObjectVariable,
        b: ObjectVariable,
    ) -> ObjectVariable:
        return a.call_method(translator, "__add__", [a, b], {})

    def call_iadd(
        self,
        translator,
        a: ObjectVariable,
        b: ObjectVariable,
    ) -> ObjectVariable:
        return a.call_method(translator, "__iadd__", [a, b], {})


class ObjectVariable(VariableBase):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def __str__(self):
        return str(self.obj)

    def call_function(
        self,
        translator,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> VariableBase:
        return translator.output.create_node(
            "call_method", "__call__", [self] + args, kwargs
        )

    def call_method(
        self,
        translator,
        name: str,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> ObjectVariable:
        # proxy_args, proxy_kwargs = proxy_args_kwargs([self] + args, kwargs)
        # return ObjectVariable(
        #     translator.output.create_proxy(
        #         "call_method", name, proxy_args, proxy_kwargs
        #     )
        # )
        return ObjectVariable(
            translator.output.create_node("call_method", name, args, kwargs)
        )


class BuiltinVariable(CallableVariable):
    def call_print(
        self,
        translator,
        *args: tuple[VariableBase],
        **kwargs: dict[str, VariableBase],
    ) -> ConstantVariable:
        return ConstantVariable(None)

    def call_getattr(
        self, translator, obj: ObjectVariable, name: str
    ) -> ObjectVariable:
        return ObjectVariable(
            translator.output.create_node("call_method", "__getattr__", [obj, name])
        )


class LayerVariable(ObjectVariable):
    def __init__(self, target: str):
        super().__init__(target)
        # TODO: those are used to generate code
        self.args = []
        self.kwargs = {}

    def __str__(self):
        args = ", ".join(self.args)
        kwargs = ", ".join(self.kwargs)
        return f"{self.obj}({args}, {kwargs})"

    def call_function(
        self,
        translator,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> VariableBase:
        self.args = [str(a) for a in args]
        self.kwargs = [f"k{k}={v}" for k, v in kwargs.items()]
        return translator.output.create_node("call_module", self.obj, args, kwargs)


class TensorVariable(ObjectVariable):
    def __init__(self, proxy: Proxy):
        super().__init__(proxy)

    def as_proxy(self) -> Proxy:
        return self.obj


class PaddleVariable(CallableVariable):
    pass


# note: python module
class ModuleVariable(ObjectVariable):
    def __init__(self, module: types.ModuleType):
        super().__init__(module)

    def __str__(self):
        return self.obj.__name__

    def __getattr__(self, attr: str):
        out_obj = getattr(self.obj, attr)
        if isinstance(out_obj, types.ModuleType):
            return ModuleVariable(out_obj)
        elif isinstance(out_obj, types.FunctionType):
            return CallableVariable(out_obj)
        else:
            return ObjectVariable(out_obj)
