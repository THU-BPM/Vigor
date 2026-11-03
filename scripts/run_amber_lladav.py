#!/usr/bin/env python3
"""AMBER benchmark evaluation for LLaDA-V (baseline + VIGOR)."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LLADA_V_ROOT = ROOT / "LLaDA-V" / "train"
if str(LLADA_V_ROOT) not in sys.path:
    sys.path.insert(0, str(LLADA_V_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


@torch.no_grad()
def generate_with_embeds_visual_ablation(
    model, inputs_embeds, image_positions, *,
    steps, gen_length, block_length, temperature,
    mask_id=126336, probe_interval=1,
):
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    masked_embed = model.model.embed_tokens(torch.tensor([mask_id], device=inputs_embeds.device))
    total_len = inputs_embeds.shape[1] + gen_length

    x_embeds = masked_embed.repeat(1, total_len, 1).to(inputs_embeds.device)
    x_embeds[:, : inputs_embeds.shape[1]] = inputs_embeds.clone()
    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=inputs_embeds.device)

    prompt_len = inputs_embeds.shape[1]
    img_positions_full = None
    if image_positions is not None and image_positions.numel() > 0:
        img_positions_full = image_positions

    for num_block in range(num_blocks):
        cached_logits_noimg = None
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = torch.all(
            torch.abs(x_embeds[:, block_start:block_end] - masked_embed) < 1e-5, dim=2,
        )
        num_transfer_tokens = model.get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = torch.all(torch.abs(x_embeds - masked_embed) < 1e-5, dim=2)
            current_block_masks = mask_index[0, block_start:block_end]
            if not current_block_masks.any():
                break

            out = model.model(inputs_embeds=x_embeds)
            logits_img = model.lm_head(out[0]).float()
            logits_with_noise = model.add_gumbel_noise(logits_img, temperature=temperature)
            x0_img = torch.argmax(logits_with_noise, dim=-1)
            p_img = F.softmax(logits_img.to(torch.float64), dim=-1)
            conf_img = torch.gather(p_img, dim=-1, index=x0_img.unsqueeze(-1)).squeeze(-1)
            x0_p = conf_img

            def _noimg_forward():
                nonlocal cached_logits_noimg
                is_probe = (i % probe_interval == 0 or cached_logits_noimg is None)
                if is_probe:
                    x_embeds_noimg = x_embeds.clone()
                    if img_positions_full is not None:
                        x_embeds_noimg[:, img_positions_full] = masked_embed
                    out_noimg = model.model(inputs_embeds=x_embeds_noimg)
                    cached_logits_noimg = model.lm_head(out_noimg[0]).float()

            _noimg_forward()
            p_noimg = F.softmax(cached_logits_noimg.to(torch.float64), dim=-1)
            conf_noimg = torch.gather(p_noimg, dim=-1, index=x0_img.unsqueeze(-1)).squeeze(-1)
            eps = 1e-8
            log_ratio = torch.log(conf_img.clamp(min=eps)) - torch.log(conf_noimg.clamp(min=eps))
            x0_p = (conf_img - conf_noimg) + log_ratio

            x0_p[:, block_end:] = -np.inf
            x0_embeds = model.model.embed_tokens(x0_img)
            x0_embeds = torch.where(mask_index.unsqueeze(-1).expand_as(x_embeds), x0_embeds, x_embeds)
            x0_img = torch.where(mask_index, x0_img, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0_img, dtype=torch.bool, device=x0_img.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i]))
                transfer_index[j, select_index] = True
            x_embeds[transfer_index] = x0_embeds[transfer_index]
            x[transfer_index] = x0_img[transfer_index]

    return x[:, prompt_len : prompt_len + gen_length]


def _infer_image_positions(input_ids, inputs_embeds):
    pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
    if pos.numel() != 1:
        return None
    p = int(pos[0].item())
    replaced_len = inputs_embeds.shape[1] - (input_ids.shape[1] - 1)
    if replaced_len <= 0:
        return None
    return torch.arange(p, p + replaced_len, device=inputs_embeds.device, dtype=torch.long)


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

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.checkpoint, None, args.model_name,
        device_map=args.device,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()
    print("Model loaded.")

    MASK_ID = 126336

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    outputs: List[dict] = []
    for idx, item in enumerate(tqdm(queries, desc="AMBER-LLaDA-V")):
        qid = int(item["id"])
        img_name = item["image"]
        question = item["query"]
        image_path = os.path.join(args.image_dir, img_name)

        is_generative = qid <= 1004
        gen_length = args.gen_length_g if is_generative else args.gen_length_d
        if is_generative and (args.steps > 0 or args.block_length > 0):
            steps = args.steps if args.steps > 0 else max(1, gen_length // 2)
            block_length = args.block_length if args.block_length > 0 else gen_length
        else:
            steps = max(1, gen_length // 2)
            block_length = gen_length

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [x.to(dtype=torch.float16, device=device) for x in image_tensor]

            conv = copy.deepcopy(conv_templates[args.conv_template])
            conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            position_ids = None
            attention_mask = torch.ones_like(input_ids, device=device)
            _, position_ids, attention_mask, _, inputs_embeds, _ = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    None, None, image_tensor, ["image"],
                    image_sizes=[image.size],
                )

            if args.baseline:
                out_tokens = model.generate_with_embeds(
                    inputs_embeds,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=args.temperature,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=MASK_ID,
                )
            else:
                image_positions = _infer_image_positions(input_ids, inputs_embeds)
                out_tokens = generate_with_embeds_visual_ablation(
                    model, inputs_embeds, image_positions,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=args.temperature,
                    mask_id=MASK_ID,
                    probe_interval=args.probe_interval,
                )

            response = tokenizer.batch_decode(
                out_tokens, skip_special_tokens=True
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
    p = argparse.ArgumentParser("AMBER LLaDA-V evaluation")
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "ckpt" / "LLaDA-V"))
    p.add_argument("--model_name", type=str, default="llava_llada")
    p.add_argument("--conv_template", type=str, default="llava_llada")
    p.add_argument("--device", type=str, default="cuda:0")
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
    p.add_argument("--steps", type=int, default=0, help="Override steps (0=auto: gen_length//2)")
    p.add_argument("--block_length", type=int, default=0, help="Override block_length (0=auto: gen_length)")
    args = p.parse_args()
    run_amber(args)


if __name__ == "__main__":
    main()
