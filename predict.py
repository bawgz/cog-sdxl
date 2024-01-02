import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weights import WeightsDownloadCache

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

from dataset_and_utils import TokenEmbeddingsHandler

SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
# SAFETY_CACHE = "./safety-cache"
# FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
# SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


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
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights, pipe):
        from no_init import no_init_or_tensor

        # weights can be a URLPath, which behaves in unexpected ways
        if weights is None:
            weights = "./trained-model-luk"
            print("Weights is None, setting to ", weights)
        else:
            print("Weights is not None, setting to ", str(weights))
            weights = str(weights)
        if self.tuned_weights == weights:
            print("skipping loading .. weights already loaded")
            return

        self.tuned_weights = weights

        local_weights_cache = self.weights_cache.ensure(weights)

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        print("Loading Unet LoRA")

        unet = pipe.unet

        tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

        unet_lora_attn_procs = {}
        name_rank_map = {}
        for tk, tv in tensors.items():
            # up is N, d
            if tk.endswith("up.weight"):
                proc_name = ".".join(tk.split(".")[:-3])
                r = tv.shape[1]
                name_rank_map[proc_name] = r

        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            with no_init_or_tensor():
                module = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=name_rank_map[name],
                )
            unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

        unet.set_attn_processor(unet_lora_attn_procs)
        unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        print("Setting tuned_model to True")
        self.tuned_model = True

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""

        start = time.time()
        print("Setting up predictor...")
        self.tuned_model = False
        self.tuned_weights = None

        self.weights_cache = WeightsDownloadCache()

        # print("Loading safety checker...")
        # if not os.path.exists(SAFETY_CACHE):
        #     download_weights(SAFETY_URL, SAFETY_CACHE)
        # self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     SAFETY_CACHE, torch_dtype=torch.float16
        # ).to("cuda")

        # self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.txt2img_pipe2 = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.is_lora = False

        if not os.path.exists("./trained-model-luk"):
          download_weights("https://replicate.delivery/pbxt/K8l70F8kIPrIy6GDcoMok2k2C7EJSeWL3kQ4V52LKhsBqhe8/trained_model_luk.tar", "./trained-model-luk")

        # if not os.path.exists("./trained-model-tok"):
        #   download_weights("https://replicate.delivery/pbxt/K7ku1HCBJMUchwXERHHSMi4Vkm3W75Qox5Rt5nKG7kGYmgkf/trained_model.tar", "trained-model-tok")

        self.load_trained_weights("./trained-model-luk", self.txt2img_pipe2)

        # self.txt2img_pipe.load_lora_weights("./trained-model-luk", weight_name="lora.safetensors", adapter_name="LUK")
        # self.txt2img_pipe.load_lora_weights("./trained-model-tok", weight_name="lora.safetensors", adapter_name="TOK")

        # self.txt2img_pipe.to("cuda")
        self.txt2img_pipe2.to("cuda")

        # print("Loading SDXL img2img pipeline...")
        # self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        # )
        # self.img2img_pipe.to("cuda")

        # print("Loading SDXL inpaint pipeline...")
        # self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        # )
        # self.inpaint_pipe.to("cuda")

        # print("Loading SDXL refiner pipeline...")
        # FIXME(ja): should the vae/text_encoder_2 be loaded from SDXL always?
        #            - in the case of fine-tuned SDXL should we still?
        # FIXME(ja): if the answer to above is use VAE/Text_Encoder_2 from fine-tune
        #            what does this imply about lora + refiner? does the refiner need to know about

        if not os.path.exists(REFINER_MODEL_CACHE):
            download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        print("Loading refiner pipeline...")
        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_MODEL_CACHE,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            vae=self.txt2img_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")
        print("setup took: ", time.time() - start)
        # self.txt2img_pipe.__class__.encode_prompt = new_encode_prompt

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
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
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        # apply_watermark: bool = Input(
        #     description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
        #     default=True,
        # ),
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
        # replicate_weights: str = Input(
        #     description="Replicate LoRA weights to use. Leave blank to use the default weights.",
        #     default=None,
        # ),
        # disable_safety_checker: bool = Input(
        #     description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
        #     default=False
        # )
    ) -> List[Path]:
        print("Running prediction...")
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        print("setting addapters")
        self.txt2img_pipe.set_adapters(["LUK", "TOK"], adapter_weights=[lora_scale, lora_scale2])
        
        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        print("tuned_model: ", self.tuned_model)

        prompt2 = prompt

        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt2 = prompt2.replace(k, v)

        print("prompt2: ", prompt2)
        # print(f"Prompt: {prompt}")

        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        # pipe = self.txt2img_pipe
        pipe2 = self.txt2img_pipe2

        # pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        pipe2.scheduler = SCHEDULERS[scheduler].from_config(pipe2.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        # common_args = {
        #     "prompt": [prompt] * num_outputs,
        #     "negative_prompt": [negative_prompt] * num_outputs,
        #     "guidance_scale": guidance_scale,
        #     "generator": generator,
        #     "num_inference_steps": num_inference_steps,
        # }

        common_args2 = {
            "prompt": [prompt2] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        print("Is LoRA: ", self.is_lora)
        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output_paths = []

        output2 = pipe2(**common_args2, **sdxl_kwargs)
        
        for i, image in enumerate(output2.images):
            output_path = f"/tmp/out-1{i}.png"
            image.save(output_path)
            print("Saved image to: ", output_path)
            output_paths.append(Path(output_path))

        # output = pipe(**common_args, **sdxl_kwargs)

        # for i, image in enumerate(output.images):
        #     output_path = f"/tmp/out-0{i}.png"
        #     image.save(output_path)
        #     print("Saved image to: ", output_path)
        #     output_paths.append(Path(output_path))
        
        return output_paths
