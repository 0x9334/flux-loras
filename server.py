import logging 
import sys

logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

import torch
import time
from datetime import datetime
from diffusers import FluxPipeline
from fastapi import FastAPI
import os
from threading import Lock
import base64
from io import BytesIO
import uvicorn
from PIL import Image
import traceback
from pydantic import BaseModel, ConfigDict
import os
from stream import StderrCapturing, StdoutCapturing
from safetensors.torch import load_file as load_lora_safe_tensor_file
from mmgp import offload, profile_type


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))

def sha256_of_file(path):
    if not os.path.exists(path):
        return None

    import hashlib
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)

    return sha256.hexdigest()

def safetensor2device(tensor: dict, device):
    return {k: v.to(device) for k, v in tensor.items()}

loras_cfg = [
    {
        'path': os.path.join(get_script_dir(), 'lora', 'Flux_woman_naked-v1_Sevenof9.safetensors'),
        'template_prompt': "{}, naked",
        'best_strength': 0.95,
        'name': 'Naked woman LoRA',
        'key': 'naked_woman_lora'
    }, 
    {
        'path': os.path.join(get_script_dir(), 'lora', 'Flux.saggy breasts-v1.2-step00000450.safetensors'),
        'template_prompt': "{}, saggy breasts",
        'best_strength': 0.85,
        'name': 'Saggy breasts LoRA',
        'key': 'saggy_breasts_lora'
    }, 
    {
        'path': os.path.join(get_script_dir(), 'lora', 'frazetta_flux_v2-000150.safetensors'),
        'template_prompt': "Frank Frazetta fantasy oil painting of {}",
        'best_strength': 1,
        'name': 'Frank Frazetta Style Oil Painting LoRA',
        'key': 'frazetta_lora'
    }
]

setup_logs = {}
pipeline_guard = Lock()
inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipeline = None

CACHE_DIR = os.path.join(get_script_dir(), '.cache')
FLUX_PRETRAINED = os.environ.get("FLUX_PRETRAINED", "black-forest-labs/FLUX.1-dev")

with StderrCapturing() as stderr, StdoutCapturing() as stdout:
    setup_logs['started_at'] = datetime.now().isoformat()
    try:

        os.makedirs(CACHE_DIR, exist_ok=True)

        start = time.time()

        with StdoutCapturing() as _stdout:

            pipeline = FluxPipeline.from_pretrained(
                FLUX_PRETRAINED,
                torch_dtype=torch.bfloat16,
            ).to("cpu")

        print(f"The base pipeline has been initialized in: {time.time() - start} seconds")
        print("Loading LoRAs")

        for cfg in loras_cfg:
            assert os.path.exists(cfg['path']), f"LoRA file not found: {cfg['path']}"

            pipeline.load_lora_weights(load_lora_safe_tensor_file(cfg['path']), cfg['key'])
            print(f"Loaded LoRA: {cfg['key']}")

        print("Moving pipeline to {}".format(inference_device))
        offload.profile(pipeline, profile_type.HighRAM_HighVRAM)

        print("Pipeline ready")
        setup_logs['status'] = 'succeeded'

    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        setup_logs['status'] = 'failed'

    finally:
        setup_logs['completed_at'] = datetime.now().isoformat()

setup_logs['logs'] = '\n'.join(stdout) # + '\n'.join(stderr)

# if activ[k] = True --> use lora k
def render_prompt_for_lora(prompt: str, activ: dict):
    for cfg in loras_cfg:
        if activ.get(cfg['key'], False):
            prompt = cfg['template_prompt'].format(prompt)
            
    return prompt

def reset_adapters(use_lora_cfg: dict):
    global pipeline
    adapter_name, adapter_strength = [], []
    for cfg in loras_cfg:
        if use_lora_cfg.get(cfg['key'], False):
            adapter_name.append(cfg['key'])
            adapter_strength.append(cfg['best_strength'])
        else:
            adapter_name.append(cfg['key'])
            adapter_strength.append(0.0)

    pipeline.set_adapters(adapter_name, adapter_weights=adapter_strength)
    
def generate(
    prompt="A blue jay standing on a large basket of rainbow macarons, disney style", 
    guidance_scale=3.5,
    num_inference_steps=20, 
    seed=0,
    height=1024,
    width=1024,
    lora_cfg={},
):
    global pipeline, inference_device, pipeline_guard

    with pipeline_guard:
        print("Fusing LoRA with settings: ", lora_cfg)
        reset_adapters(lora_cfg)

        if os.environ.get("DEBUG") != "1":
            logger.info(f"Generating image with prompt: {prompt}; Cfg: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}, seed={seed}")
            formated_prompt = render_prompt_for_lora(prompt, lora_cfg)
            logger.info(f"Formatted prompt: {formated_prompt}")
            
            resp = pipeline(
                prompt=formated_prompt,
                guidance_scale=guidance_scale,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                joint_attention_kwargs={"scale": 1.75}, 
                generator=torch.Generator("cpu").manual_seed(seed),                
                height=height,
                width=width
            ).images[0]
        else:
            logger.info("DEBUG mode enabled, returning red image")
            resp = Image.new("RGB", (512, 512), (255, 0, 0))

    return resp

app = FastAPI()

@app.get("/lora-cfgs")
async def get_lora_cfgs():
    return [
        {
            'key': cfg['key'],
            'name': cfg['name']
        } for cfg in loras_cfg
    ]

@app.get("/health-check")
async def get_lora_cfgs():
    global pipeline_guard, setup_logs
    ready = not pipeline_guard.locked()

    resp = {
        'status': 'READY' if ready else 'BUSY',
        'setup': setup_logs
    }

    return resp

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    guidance_scale: float = 3.5
    num_inference_steps: int = 20
    seed: int = 0
    height: int = 1024
    width: int = 1024
    lora_cfg: dict = {
        entry['key']: True 
        for entry in loras_cfg
    }

# TODO: Implement this
class OpenAIImageGenerationRequest(BaseModel):
    prompt: str
    model: str = ''
    n: int = 1
    quality: str = 'hd' # 'hd'
    response_format: str = 'url' # 'url'
    size: str = '1024x1024' # '1024x1024'
    style: str = 'vivid' # 'None'
    user: str = '' # 'user'
    
# TODO: Implement this
class OpenAIImageGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    guidance_scale: float = 3.5
    num_inference_steps: int = 20
    seed: int = 0
    height: int = 1024
    width: int = 1024
    lora_cfg: dict = {
        entry['key']: True 
        for entry in loras_cfg
    }


@app.post("/generate")
async def generate_api(request: GenerateRequest):
    try:
        image = generate(
            prompt=request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            height=request.height,
            width=request.width,
            lora_cfg=request.lora_cfg
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    bio = BytesIO()
    image.save(bio, format="PNG")
    b64_image = base64.b64encode(bio.getvalue()).decode("utf-8")

    return {"image": b64_image}

# TODO: Implement this
@app.post("/v1/generate")
async def generate_api_v1(request: GenerateRequest):
    try:
        image = generate(
            prompt=request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            height=request.height,
            width=request.width,
            lora_cfg=request.lora_cfg
        )
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    bio = BytesIO()
    image.save(bio, format="PNG")
    b64_image = base64.b64encode(bio.getvalue()).decode("utf-8")

    return {"image": b64_image}

if __name__ == '__main__':
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="debug",
        timeout_keep_alive=60
    )