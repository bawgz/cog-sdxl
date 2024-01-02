from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline



class Predictor(BasePredictor):
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
        )
    ) -> List[Path]:
        """Run a single prediction on the model."""

        print(f"Prompt: {prompt}")

        pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

        pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

        output = pipe(prompt, num_inference_steps=num_inference_steps, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0))

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths