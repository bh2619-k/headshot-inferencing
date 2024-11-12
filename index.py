import runpod
import sys
import asyncio
import torch
from diffusers import FluxPipeline
import boto3
import base64
import io


def download_file_from_s3(bucket_name, file_key, local_file_path):
    """
    Download a file from a S3 bucket to a local file path.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    file_key (str): The key/path of the file in the S3 bucket.
    local_file_path (str): The local file path to save the downloaded file.

    Returns:
    None
    """
    s3 = boto3.client("s3")

    try:
        s3.download_file(bucket_name, file_key, local_file_path)
        print(f"File downloaded successfully: {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")


def inferencing(job):
    """
    Generate the images of the trained instance.

    Parameters:
    s3_lora_bucket (str): The name of the S3 bucket that includes lora weight.
    s3_lora_object (str): The key/path of the lora weight in the S3 bucket.
    prompt (str): The prompt to generate an image from.
    guidance_scale (float): A scaling factor that influences the adherence to the prompt.
    height (int): The height of the generated images in pixels.
    width (int): The width of the generated images in pixels.
    num_inference_steps (int): The number of steps for the diffusion process.

    Returns:
    list: A list of base64 encoded images generated from the provided prompt.
    """
    job_input = job["input"]
    s3_lora_bucket = job_input["s3_lora_bucket"]
    s3_lora_object = job_input["s3_lora_object"]
    prompt = job_input["prompt"]
    guidance_scale = job_input["guidance_scale"]
    height = job_input["height"]
    width = job_input["width"]
    num_inference_steps = job_input["num_inference_steps"]

    download_file_from_s3(s3_lora_bucket, s3_lora_object, "./pytorch_lora_weights.safetensors")


    try:
        # Create FluxPipeline and move to cuda
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, use_safetensors=True
        ).to("cuda")

        # Load lora weights
        pipe.load_lora_weights("./pytorch_lora_weights.safetensors")

        # Inference
        images = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(1641421826),
        ).images

        # all_images.extend(images)

        base64_images = []
        for image in images:
            bytes = io.BytesIO()
            image.save(bytes, format="PNG")
            base64_image = base64.b64encode(bytes.getvalue()).decode("utf-8")
            base64_images.append(base64_image)

        return base64_images

    except Exception as e:
        print(f"Error during inference: {e}")
        return []

if __name__ == "__main__":
    # Use WindowsSelectorEventLoopPolicy on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Start the serverless handler
    runpod.serverless.start({"handler": inferencing})
