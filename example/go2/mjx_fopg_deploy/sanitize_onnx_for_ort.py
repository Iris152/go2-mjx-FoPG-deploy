#!/usr/bin/env python3
"""Sanitize exported ONNX graphs so they can load in ONNX Runtime.

Fixes currently implemented:
1. Expm1(x) -> Exp(x) - 1
2. PreventGradient(x) -> Identity(x)

Usage:
  python sanitize_onnx_for_ort.py in.onnx out.onnx
"""

from __future__ import annotations

import argparse
import itertools
from typing import Dict, Iterable

import onnx
from onnx import TensorProto, helper
from onnx.numpy_helper import from_array
import numpy as np


def _collect_existing_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    graph = model.graph
    for node in graph.node:
        if node.name:
            names.add(node.name)
        names.update(x for x in node.input if x)
        names.update(x for x in node.output if x)
    for init in graph.initializer:
        names.add(init.name)
    for vi in itertools.chain(graph.input, graph.output, graph.value_info):
        names.add(vi.name)
    return names


def _elem_type_map(model: onnx.ModelProto) -> Dict[str, int]:
    type_map: Dict[str, int] = {}
    graph = model.graph
    for vi in itertools.chain(graph.input, graph.output, graph.value_info):
        t = vi.type.tensor_type
        if t.HasField("elem_type"):
            type_map[vi.name] = t.elem_type
    for init in graph.initializer:
        type_map[init.name] = init.data_type
    return type_map


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 1
    while True:
        cand = f"{base}_{idx}"
        if cand not in used:
            used.add(cand)
            return cand
        idx += 1


def _scalar_one(elem_type: int) -> np.ndarray:
    if elem_type == TensorProto.DOUBLE:
        return np.array(1.0, dtype=np.float64)
    return np.array(1.0, dtype=np.float32)


def sanitize_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int, int]:
    used_names = _collect_existing_names(model)
    elem_types = _elem_type_map(model)
    graph = model.graph

    new_nodes = []
    new_initializers = list(graph.initializer)
    replaced_expm1 = 0
    replaced_pg = 0

    for node in graph.node:
        if node.op_type == "Expm1":
            replaced_expm1 += 1
            in_name = node.input[0]
            out_name = node.output[0]
            elem_type = elem_types.get(in_name, elem_types.get(out_name, TensorProto.FLOAT))

            exp_out = _unique_name(f"{out_name}__exp", used_names)
            one_name = _unique_name(f"{out_name}__one", used_names)

            exp_node = helper.make_node(
                "Exp",
                [in_name],
                [exp_out],
                name=_unique_name((node.name or "Expm1") + "_Exp", used_names),
            )
            sub_node = helper.make_node(
                "Sub",
                [exp_out, one_name],
                list(node.output),
                name=_unique_name((node.name or "Expm1") + "_Sub", used_names),
            )
            tensor = from_array(_scalar_one(elem_type), name=one_name)
            new_initializers.append(tensor)
            new_nodes.extend([exp_node, sub_node])
            continue

        if node.op_type == "PreventGradient":
            replaced_pg += 1
            identity_node = helper.make_node(
                "Identity",
                list(node.input),
                list(node.output),
                name=_unique_name((node.name or "PreventGradient") + "_Identity", used_names),
            )
            new_nodes.append(identity_node)
            continue

        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model, replaced_expm1, replaced_pg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model")
    parser.add_argument("output_model")
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    model, n_expm1, n_pg = sanitize_model(model)
    print(f"Replaced {n_expm1} Expm1 node(s).")
    print(f"Replaced {n_pg} PreventGradient node(s).")
    onnx.checker.check_model(model)
    onnx.save(model, args.output_model)
    print(f"Saved sanitized model to: {args.output_model}")


if __name__ == "__main__":
    main()
