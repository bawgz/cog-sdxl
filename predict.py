from typing import List, Optional

import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline



class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("weights: ", weights)

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

        pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

        pipe.load_lora_weights("./trained-model/", weight_name="lora.safetensors", adapter_name="LUK")
        pipe.load_lora_weights("./trained-model-tok/", weight_name="lora.safetensors", adapter_name="TOK")

        pipe.set_adapters(["LUK", "TOK"], adapter_weights=[lora_scale, lora_scale2])

        output = pipe(prompt, num_inference_steps=num_inference_steps, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0))

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths