# Prediction interface for Cog
import os
import subprocess
import time
from typing import List

import numpy as np
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
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "model-cache"
REPO = "ByteDance/SDXL-Lightning"
UNET_CKPT = "sdxl_lightning_2step_unet.safetensors"
UNET_CACHE = "unet-cache"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "feature-extractor"


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


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
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

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
    
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
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=False,
        ),
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

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
