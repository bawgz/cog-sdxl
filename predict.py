from typing import List, Optional
import os
import time
import subprocess

import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline

SDXL_MODEL_CACHE = "./sdxl-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("weights: ", weights)
        
        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)
            
        self.pipe = DiffusionPipeline.from_pretrained("./sdxl-cache", torch_dtype=torch.float16).to("cuda")

        self.pipe.load_lora_weights("./trained-model-luk/", weight_name="lora.safetensors", adapter_name="LUK")
        self.pipe.load_lora_weights("./trained-model-tok/", weight_name="lora.safetensors", adapter_name="TOK")

        # pipe.load_textual_inversion("./trained-model-tok/", weight_name="embeddings.pti", token="TOK")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        lora_scale2: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""

        print(f"Prompt: {prompt}")

        self.pipe.set_adapters(["LUK", "TOK"], adapter_weights=[lora_scale, lora_scale2])

        output = self.pipe(prompt, num_inference_steps=num_inference_steps, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0))

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths