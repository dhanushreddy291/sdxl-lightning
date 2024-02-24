# Prediction interface for Cog
import os
import subprocess
import time
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "model-cache"
REPO = "ByteDance/SDXL-Lightning"
UNET_CKPT = "sdxl_lightning_2step_unet.safetensors"
UNET_CACHE = "unet-cache"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DPM++2MSDE": KDPM2AncestralDiscreteScheduler,
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        unet = UNet2DConditionModel.from_config(
            BASE_MODEL,
            subfolder="unet",
            cache_dir=UNET_CACHE,
            local_files_only=True,
        ).to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(REPO, UNET_CKPT), device="cuda"))
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL,
            unet=unet,
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="21 years old girl,short cut,beauty,dusk,Ghibli style illustration",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="3d, cgi, render, bad quality, normal quality",
        ),
        width: int = Input(
            description="Width of output image. Recommended 1024 or 1280", default=1024
        ),
        height: int = Input(
            description="Height of output image. Recommended 1024 or 1280", default=1024
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=4,
            default=1,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance. Recommended 7-8",
            ge=0,
            le=50,
            default=0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        pipe = self.pipe
        pipe.scheduler = SCHEDULERS[scheduler].from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
        }

        output = pipe(**common_args)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
