import os
import shutil
import json
import random
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
BASE_PATH = ""
CHECKPOINT = "CinematicRedmond.safetensors"
BASE_URL = "https://weights.replicate.delivery/default/comfy-ui/checkpoints"
BASE_PATH = "ComfyUI/models/checkpoints"

with open("text-to-image-api.json", "r") as file:
    WORKFLOW_JSON = file.read()

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SCHEDULERS = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.weights_downloader.download(
            CHECKPOINT, f"{BASE_URL}/{CHECKPOINT}.tar", BASE_PATH
        )

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        workflow["4"]["inputs"]["ckpt_name"] = CHECKPOINT

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = kwargs["prompt"]

        image_dimensions = workflow["5"]["inputs"]
        image_dimensions["width"] = kwargs["width"]
        image_dimensions["height"] = kwargs["height"]
        image_dimensions["batch_size"] = kwargs["num_outputs"]

        sampler = workflow["3"]["inputs"]
        sampler["steps"] = kwargs["steps"]
        sampler["sampler_name"] = kwargs["sampler_name"]
        sampler["scheduler"] = kwargs["scheduler"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["guidance_scale"]

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        prompt: str = Input(default="a photo of an astronaut riding a unicorn"),
        negative_prompt: str = Input(
            description="The negative prompt to guide image generation.",
            default="ugly, disfigured, low quality, blurry, nsfw",
        ),
        steps: int = Input(
            description="Number of diffusion steps", ge=1, le=100, default=17
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0,
            le=30.0,
        ),
        seed: int = Input(default=None),
        width: int = Input(default=768),
        height: int = Input(default=768),
        num_outputs: int = Input(
            description="Number of outputs", ge=1, le=10, default=1
        ),
        sampler_name: str = Input(
            choices=SAMPLERS,
            default="euler",
        ),
        scheduler: str = Input(
            choices=SCHEDULERS,
            default="normal",
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images.", default=False
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(WORKFLOW_JSON)
        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            num_outputs=num_outputs,
            sampler_name=sampler_name,
            scheduler=scheduler,
            disable_safety_checker=disable_safety_checker,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        print(f"Contents of {OUTPUT_DIR}:")
        files.extend(self.log_and_collect_files(OUTPUT_DIR))

        return files
