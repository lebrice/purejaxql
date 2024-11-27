from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import load_file, save_file


import os


def save_params(params: dict, filename: str | os.PathLike) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)  # type: ignore


def load_params(filename: str | os.PathLike) -> dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")
