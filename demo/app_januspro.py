import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time
import argparse


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_config", action="store_true", help="Clean unused config parameters")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-1B", help="Path to the model")
    return parser.parse_args()


# Initialize model and processor at startup
def initialize_models(model_path, clean_config=False):
    print(f"Loading model from {model_path}...")
    start_time = time.time()

    config = AutoConfig.from_pretrained(model_path)

    # Clean config if specified
    if clean_config:
        print("Cleaning unused config parameters...")
        config_dict = config.to_dict()
        for param in ["mask_prompt", "image_tag", "add_special_token", "ignore_id", "num_image_tokens", "sft_format"]:
            config_dict.pop(param, None)
        config = config.__class__(**config_dict)

    language_config = config.language_config
    language_config._attn_implementation = 'eager'

    print("Loading model weights...")
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                language_config=language_config,
                                                trust_remote_code=True)

    cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {cuda_device}")

    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    else:
        vl_gpt = vl_gpt.to(torch.float16)

    print("Loading processor...")
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = vl_chat_processor.tokenizer

    # If using LlamaTokenizerFast
    if hasattr(tokenizer, 'legacy'):
        tokenizer.legacy = False

    # Warmup the model with a small input to initialize CUDA kernels
    if torch.cuda.is_available():
        print("Warming up model...")
        dummy_input = torch.ones((1, 10), dtype=torch.long).cuda()
        dummy_embeds = vl_gpt.language_model.get_input_embeddings()(dummy_input)
        _ = vl_gpt.language_model.model(inputs_embeds=dummy_embeds)
        torch.cuda.synchronize()

    load_time = time.time() - start_time
    print(f"Models loaded in {load_time:.2f} seconds")

    return vl_gpt, vl_chat_processor, tokenizer, cuda_device


# Get command line arguments and initialize models
args = parse_args()
vl_gpt, vl_chat_processor, tokenizer, cuda_device = initialize_models(args.model_path, args.clean_config)


@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Convert the image to RGB mode (3 channels)
    pil_img = Image.fromarray(image)
    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [np.array(pil_img)],  # Use the converted image
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    print(f"Processing input for understanding...")
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[pil_img], force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    print(f"Generating response...")
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Preparing tokens for image generation...")
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    print(f"Starting token generation loop...")
    pkv = None
    for i in range(image_token_num_per_image):
        if i > 0 and i % 100 == 0:
            print(f"Generated {i}/{image_token_num_per_image} tokens")

        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                  use_cache=True,
                                                  past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    print("Decoding image tokens...")
    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                  shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches


def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img


@torch.inference_mode()
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0):
    # Clear CUDA cache and avoid tracking gradients
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set the seed for reproducible results
    if seed is not None:
        print(f"Setting seed to {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    width = 384
    height = 384
    parallel_size = 5

    print(f"Generating images for prompt: {prompt}")
    start_time = time.time()

    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                           sft_format=vl_chat_processor.sft_format,
                                                                           system_prompt='')
        text = text + vl_chat_processor.image_start_tag

        input_ids = torch.LongTensor(tokenizer.encode(text)).to(cuda_device)
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        gen_time = time.time() - start_time
        print(f"Image generation completed in {gen_time:.2f} seconds")

        return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]


# Gradio interface
print("Setting up Gradio interface...")
with gr.Blocks() as demo:
    gr.Markdown(value="# Janus Pro - Multimodal AI Model")

    with gr.Tab("Multimodal Understanding"):
        with gr.Row():
            image_input = gr.Image()
            with gr.Column():
                question_input = gr.Textbox(label="Question")
                und_seed_input = gr.Number(label="Seed", precision=0, value=42)
                top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
                temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")

        understanding_button = gr.Button("Chat")
        understanding_output = gr.Textbox(label="Response")

        examples_inpainting = gr.Examples(
            label="Multimodal Understanding examples",
            examples=[
                [
                    "explain this meme",
                    "images/doge.png",
                ],
                [
                    "Convert the formula into latex code.",
                    "images/equation.png",
                ],
            ],
            inputs=[question_input, image_input],
        )

    with gr.Tab("Text-to-Image Generation"):
        with gr.Row():
            cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
            t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

        prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
        seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

        generation_button = gr.Button("Generate Images")

        image_output = gr.Gallery(label="Generated Images", columns=2, rows=3, height=400)

        examples_t2i = gr.Examples(
            label="Text to image generation examples.",
            examples=[
                "Master shifu racoon wearing drip attire as a street gangster.",
                "The face of a beautiful girl",
                "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                "A glass of red wine on a reflective surface.",
                "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
            ],
            inputs=prompt_input,
        )

    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )

    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )

print("Starting Gradio server...")
demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860, show_error=True)