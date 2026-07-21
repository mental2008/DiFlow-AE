from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    _GenericAlias,
    get_args,
    get_origin,
)


def type_to_string(t):
    # Handle typing generics (List[...], Dict[...], etc)
    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        args_str = ",".join(type_to_string(arg) for arg in args)
        # Preserve the original format (typing.List or list)
        if isinstance(t, _GenericAlias) and t.__module__ == "typing":
            return f"typing.{origin.__name__.capitalize()}[{args_str}]"
        return f"{origin.__module__}.{origin.__name__}[{args_str}]"

    # Handle regular types
    if isinstance(t, type):
        return f"{t.__module__}.{t.__qualname__}"

    # Handle special cases like typing.Any, typing.Union, etc
    return str(t)


def string_to_type(type_string):
    # Handle generic types
    if "[" in type_string:
        base_type, args = type_string.split("[", 1)
        args = args.rstrip("]")

        # Convert base type
        module_name, type_name = base_type.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_name)
        base = getattr(module, type_name)

        # Convert argument types
        arg_types = [string_to_type(arg.strip()) for arg in args.split(",")]

        return base[tuple(arg_types) if len(arg_types) > 1 else arg_types[0]]

    # Handle regular types
    module_name, type_name = type_string.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, type_name)


class SourceType(Enum):
    INPUT = "input"
    NODE = "node"


@dataclass
class NodeIO:
    name: str
    data_type: type
    source_type: Optional[SourceType] = None
    source_node: Optional[str] = None  # TODO (Lingyun): Is it optional?
    size: Optional[list[int]] = None  # Used for pre-allocating output tensors
    lazy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": type_to_string(self.data_type),
            "source_type": self.source_type.value if self.source_type else None,
            "source_node": self.source_node,
            "size": self.size,
            "lazy": self.lazy,
        }

    @classmethod
    def from_dict(cls, io_dict: Dict[str, Any]) -> "NodeIO":
        return cls(
            name=io_dict["name"],
            data_type=string_to_type(io_dict["data_type"]),
            source_type=(
                SourceType(io_dict["source_type"]) if io_dict["source_type"] else None
            ),
            source_node=io_dict["source_node"],
            size=io_dict["size"],
            lazy=io_dict["lazy"],
        )


class DiffusionModelInputs(NamedTuple):
    latents: NodeIO
    prompt_embeds: NodeIO
    pooled_prompt_embeds: Optional[NodeIO] = None
    negative_prompt_embeds: Optional[NodeIO] = None
    negative_pooled_prompt_embeds: Optional[NodeIO] = None

    # these are for Flux 1.0 Dev
    # Note: This is not the same as the value used in the classifer-free guidance.
    guidance_scale: Optional[NodeIO] = None

    # these are for SDXL & FLUX
    height: Optional[NodeIO] = None
    width: Optional[NodeIO] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latents": self.latents.to_dict(),
            "prompt_embeds": self.prompt_embeds.to_dict(),
            "pooled_prompt_embeds": (
                self.pooled_prompt_embeds.to_dict()
                if self.pooled_prompt_embeds is not None
                else None
            ),
            "negative_prompt_embeds": (
                self.negative_prompt_embeds.to_dict()
                if self.negative_prompt_embeds is not None
                else None
            ),
            "negative_pooled_prompt_embeds": (
                self.negative_pooled_prompt_embeds.to_dict()
                if self.negative_pooled_prompt_embeds is not None
                else None
            ),
            "guidance_scale": (
                self.guidance_scale.to_dict()
                if self.guidance_scale is not None
                else None
            ),
            "height": (self.height.to_dict() if self.height is not None else None),
            "width": (self.width.to_dict() if self.width is not None else None),
        }

    @classmethod
    def from_dict(cls, io_dict: Dict[str, Any]) -> "DiffusionModelInputs":
        return cls(
            latents=NodeIO.from_dict(io_dict["latents"]),
            prompt_embeds=NodeIO.from_dict(io_dict["prompt_embeds"]),
            pooled_prompt_embeds=(
                NodeIO.from_dict(io_dict["pooled_prompt_embeds"])
                if io_dict["pooled_prompt_embeds"] is not None
                else None
            ),
            negative_prompt_embeds=(
                NodeIO.from_dict(io_dict["negative_prompt_embeds"])
                if io_dict["negative_prompt_embeds"] is not None
                else None
            ),
            negative_pooled_prompt_embeds=(
                NodeIO.from_dict(io_dict["negative_pooled_prompt_embeds"])
                if io_dict["negative_pooled_prompt_embeds"] is not None
                else None
            ),
            # these are for Flux 1.0 Dev
            guidance_scale=(
                NodeIO.from_dict(io_dict["guidance_scale"])
                if io_dict["guidance_scale"] is not None
                else None
            ),
            height=(
                NodeIO.from_dict(io_dict["height"])
                if io_dict["height"] is not None
                else None
            ),
            width=(
                NodeIO.from_dict(io_dict["width"])
                if io_dict["width"] is not None
                else None
            ),
        )


class SchedulerInputs(NamedTuple):
    num_inference_steps: NodeIO
    guidance_scale: Optional[NodeIO] = None
    # Suyi: for the img2img pipeline
    strength: Optional[NodeIO] = None
    seed: Optional[NodeIO] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_inference_steps": self.num_inference_steps.to_dict(),
            "guidance_scale": (
                self.guidance_scale.to_dict()
                if self.guidance_scale is not None
                else None
            ),
            # Suyi: for the img2img pipeline
            "strength": (
                self.strength.to_dict() if self.strength is not None else None
            ),
            # Suyi: for the img2img pipeline
            "seed": (self.seed.to_dict() if self.seed is not None else None),
        }

    @classmethod
    def from_dict(cls, io_dict: Dict[str, Any]) -> "SchedulerInputs":
        return cls(
            num_inference_steps=NodeIO.from_dict(io_dict["num_inference_steps"]),
            guidance_scale=(
                NodeIO.from_dict(io_dict["guidance_scale"])
                if io_dict["guidance_scale"] is not None
                else None
            ),
            # Suyi: for the img2img pipeline
            strength=(
                NodeIO.from_dict(io_dict["strength"])
                if io_dict["strength"] is not None
                else None
            ),
            # Suyi: for the img2img pipeline
            seed=(
                NodeIO.from_dict(io_dict["seed"])
                if io_dict["seed"] is not None
                else None
            ),
        )


class AdapterInputs(NamedTuple):
    controlnet_cond: NodeIO
    conditioning_scale: NodeIO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controlnet_cond": self.controlnet_cond.to_dict(),
            "conditioning_scale": self.conditioning_scale.to_dict(),
        }

    @classmethod
    def from_dict(cls, io_dict: Dict[str, Any]) -> "AdapterInputs":
        return cls(
            controlnet_cond=NodeIO.from_dict(io_dict["controlnet_cond"]),
            conditioning_scale=NodeIO.from_dict(io_dict["conditioning_scale"]),
        )
