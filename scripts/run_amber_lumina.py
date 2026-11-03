#!/usr/bin/env python3
"""AMBER benchmark evaluation for Lumina-DiMOO (baseline + VIGOR)."""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LUMINA_ROOT = ROOT / "Lumina-DiMOO"
if str(LUMINA_ROOT) not in sys.path:
    sys.path.insert(0, str(LUMINA_ROOT))

from config import SPECIAL_TOKENS
from generators.text_understanding_generator import generate_text_understanding
from utils.generation_utils import add_gumbel_noise, get_num_transfer_tokens
from model import LLaDAForMultiModalGeneration
from transformers import AutoTokenizer
from utils.image_utils import (
    add_break_line, calculate_vq_params, encode_img_with_breaks,
    generate_crop_size_list, var_center_crop,
)
from utils.prompt_utils import generate_multimodal_understanding_prompt
from diffusers import VQModel


@torch.no_grad()
def generate_text_understanding_visual_ablation(
    model, prompt, *, steps, gen_length, block_length, temperature,
    mask_id, code_start, probe_interval=1,
    img_start=-1, img_end=-1,
):
    x = prompt.clone()
    has_img_span = img_start >= 0 and img_end > img_start

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        cached_logits_noimg = None
        block_mask_index = (
            x[:, code_start + num_block * block_length : code_start + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            current_block_masks = mask_index[0, code_start + num_block * block_length : code_start + (num_block + 1) * block_length]
            if not current_block_masks.any():
                break

            logits_img = model(x, infer=True).logits
            logits_with_noise = add_gumbel_noise(logits_img, temperature=temperature)
            x0_img = torch.argmax(logits_with_noise, dim=-1)
            p_img = F.softmax(logits_img.to(torch.float64), dim=-1)
            conf_img = torch.gather(p_img, dim=-1, index=x0_img.unsqueeze(-1)).squeeze(-1)
            x0_p = conf_img

            def _noimg_forward():
                nonlocal cached_logits_noimg
                is_probe = (i % probe_interval == 0 or cached_logits_noimg is None)
                if is_probe:
                    x_noimg = x.clone()
                    if has_img_span:
                        x_noimg[:, img_start:img_end] = mask_id
                    cached_logits_noimg = model(x_noimg, infer=True).logits

            _noimg_forward()
            p_noimg = F.softmax(cached_logits_noimg.to(torch.float64), dim=-1)
            conf_noimg = torch.gather(p_noimg, dim=-1, index=x0_img.unsqueeze(-1)).squeeze(-1)
            eps = 1e-8
            log_ratio = torch.log(conf_img.clamp(min=eps)) - torch.log(conf_noimg.clamp(min=eps))
            x0_p = (conf_img - conf_noimg) + log_ratio

            x0_p[:, code_start + (num_block + 1) * block_length :] = -np.inf
            x0_img = torch.where(mask_index, x0_img, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0_img, dtype=torch.bool, device=x0_img.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i]))
                transfer_index[j, select_index] = True
            x[transfer_index] = x0_img[transfer_index]

    return x


def _normalize_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    if "yes" in t[:10]:
        return "Yes"
    if "no" in t[:10]:
        return "No"
    return "No"


@torch.no_grad()
def run_amber(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.query_json, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if args.max_samples and args.max_samples > 0:
        queries = queries[:args.max_samples]

    if args.num_shards > 1:
        queries = queries[args.shard_id :: args.num_shards]

    print(f"[shard {args.shard_id}/{args.num_shards}] {len(queries)} queries on {args.device}")
    print(f"Mode: {'VIGOR (paper score, mask ablation)' if not args.baseline else 'baseline'}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vae_ckpt = args.vae_ckpt or args.checkpoint
    vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    print("Model loaded.")

    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    outputs: List[dict] = []
    for idx, item in enumerate(tqdm(queries, desc="AMBER-Lumina")):
        qid = int(item["id"])
        img_name = item["image"]
        question = item["query"]
        image_path = os.path.join(args.image_dir, img_name)

        is_generative = qid <= 1004
        gen_length = args.gen_length_g if is_generative else args.gen_length_d
        steps = max(1, gen_length // 2)
        block_length = gen_length

        try:
            image = Image.open(image_path).convert("RGB")
            crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
            image = var_center_crop(image, crop_size_list=crop_size_list)
            image_w, image_h = image.size
            _, _, tg_h, tg_w = calculate_vq_params(image_h, image_w, vae_scale)
            input_img_token = encode_img_with_breaks(image, vqvae=vqvae)
            img_token = add_break_line(input_img_token, tg_h, tg_w, new_number=NEW_LINE)

            input_prompt = generate_multimodal_understanding_prompt(question)
            input_ids = tokenizer(input_prompt)["input_ids"]
            prefix_before_img = len(input_ids[:-1])
            input_token = input_ids[:-1] + img_token + input_ids[-1:]
            _img_start = prefix_before_img
            _img_end = _img_start + len(img_token)
            code_start = len(input_token) + 1
            input_token = input_token + [BOA] + gen_length * [MASK] + [EOA]
            input_ids_tensor = torch.tensor(input_token, device=device).unsqueeze(0)

            if args.baseline:
                out_new = generate_text_understanding(
                    model, input_ids_tensor,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=args.temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=MASK,
                    code_start=code_start,
                )
            else:
                out_new = generate_text_understanding_visual_ablation(
                    model, input_ids_tensor,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=args.temperature,
                    mask_id=MASK,
                    code_start=code_start,
                    probe_interval=args.probe_interval,
                    img_start=_img_start,
                    img_end=_img_end,
                )

            response = tokenizer.batch_decode(
                out_new[:, code_start:-1], skip_special_tokens=True
            )[0].strip().replace("\n", " ").strip()

        except Exception as e:
            print(f"  [ERROR] id={qid} {img_name}: {e}")
            response = "No" if not is_generative else "unknown"

        if not is_generative:
            response = _normalize_yesno(response)

        outputs.append({"id": qid, "response": response})

        if (idx + 1) % 200 == 0:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    n_gen = sum(1 for o in outputs if o["id"] <= 1004)
    n_disc = len(outputs) - n_gen
    print(f"Done: {len(outputs)} total ({n_gen} generative, {n_disc} discriminative)")
    print(f"Saved: {args.output_json}")


def main():
    p = argparse.ArgumentParser("AMBER Lumina-DiMOO evaluation")
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "ckpt" / "Lumina-DiMOO"))
    p.add_argument("--vae_ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_json", type=str, required=True)
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--gen_length_g", type=int, default=64, help="gen_length for generative tasks")
    p.add_argument("--gen_length_d", type=int, default=4, help="gen_length for discriminative tasks")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=0, help=">0 to limit samples")
    p.add_argument("--baseline", action="store_true", help="Standard decoding without VIGOR")
    p.add_argument("--probe_interval", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0, help="Shard index for multi-GPU")
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()
    run_amber(args)


if __name__ == "__main__":
    main()
