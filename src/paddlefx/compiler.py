from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import paddle

import paddlefx

if TYPE_CHECKING:
    import tvm.relay


def paddle_dtype_to_str(dtype: paddle.dtype) -> str:
    if dtype == paddle.float32:
        return "float32"
    elif dtype == paddle.float64:
        return "float64"
    elif dtype == paddle.float16:
        return "float16"
    elif dtype == paddle.int32:
        return "int32"
    elif dtype == paddle.int64:
        return "int64"
    elif dtype == paddle.bool:
        return "bool"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class CompilerBase:
    def __init__(self, *, full_graph=False, print_tabular: bool = False):
        self.full_graph = full_graph  # TODO: support full_graph
        self.print_tabular = print_tabular
        self.input_index = 0

    def __call__(self, gl: paddlefx.GraphLayer, dummy_inputs: list):
        if self.print_tabular:
            gl.graph.print_tabular()
        return self.compile(gl, dummy_inputs)

    def compile(self, gl: paddlefx.GraphLayer, dummy_inputs: list) -> Callable:
        dummy_outputs = gl.forward(*dummy_inputs)
        symbol_table: dict[str, Any] = {}
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node, symbol_table, dummy_inputs)
            self.input_index = 0
            return self.gen_compiled_func(symbol_table, dummy_outputs)
        except (AttributeError, NotImplementedError) as e:
            print(f"AttributeError when compiling graph: {e}")
            self.input_index = 0
            return gl.forward

    def gen_compiled_func(self, symbol_table: dict[str, Any], dummy_outputs: Any):
        raise NotImplementedError("CompilerBase is a abstract class")


class TVMCompiler(CompilerBase):
    def gen_compiled_func(
        self, symbol_table: dict[str, tvm.relay.Var], dummy_outputs: Any
    ):
        import tvm

        from tvm import relay
        from tvm.contrib import graph_executor

        output = symbol_table["output"]
        func = relay.Function(relay.analysis.free_vars(output), output)
        ir_mod = tvm.IRModule()
        ir_mod["main"] = func

        target = "llvm"
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(ir_mod, target, params={})
        dev = tvm.device(target, 0)
        graph_mod = graph_executor.GraphModule(lib["default"](dev))

        def compiled_func(*args):
            inputs = {
                name: tvm.nd.array(arg.numpy(), dev)
                for (name, var), arg in zip(symbol_table.items(), args)
                if var.name_hint.startswith("input")
            }
            dummy_output = dummy_outputs[0]
            output = tvm.nd.empty(
                dummy_output.shape, paddle_dtype_to_str(dummy_output.dtype)
            )
            graph_mod.run(**inputs)
            output = paddle.to_tensor(graph_mod.get_output(0).numpy())
            return (output,)

        return compiled_func

    def compile_placeholder(
        self, node: paddlefx.Node, symbol_table: dict[str, tvm.relay.Var], inputs: list
    ):
        import tvm.relay

        symbol_table[node.name] = tvm.relay.var(
            f"input_{node.name}",
            shape=inputs[self.input_index].shape,
            dtype=paddle_dtype_to_str(inputs[self.input_index].dtype),
        )
        self.input_index += 1

    def compile_call_function(
        self, node: paddlefx.Node, symbol_table: dict[str, tvm.relay.Var], inputs: list
    ):
        import tvm.relay

        if node.target.__name__ in ["add", "sub", "mul", "truediv"]:
            left = symbol_table[str(node.args[0])]
            right = symbol_table[str(node.args[1])]
            tvm_relay_func = getattr(tvm.relay, node.target.__name__)
            symbol_table[str(node.name)] = tvm_relay_func(left, right)
        else:
            raise NotImplementedError(f"Unsupported function: {node.target.__name__}")

    def compile_output(
        self, node: paddlefx.Node, symbol_table: dict[str, tvm.relay.Var], inputs: list
    ):
        ret = symbol_table[str(node.args[0][0])]
        symbol_table["output"] = ret
