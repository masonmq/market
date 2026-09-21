import json
import re
from pathlib import Path

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


# Paths needs to edit

MODEL_PATH = "C:/Your/Path/To/Model/Qwen2.5-VL-7B-Instruct"

IMAGE_PATH = "C:/Your/Path/To/image.jpg"

EXAMPLES_PATH = "C:/Your/Path/To/FewShot_Examples.md"

PROMPT_TEMPLATE_PATH = "C:/Your/Path/To/prompt.md"


# Metadata paste here
TARGET_POST = r"""

""".strip()


# Helper functions
def read_text(path: str) -> str:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    return path_obj.read_text(encoding="utf-8")


def get_local_image_path(path: str) -> str:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {path_obj}")
    return str(path_obj.resolve())


def build_prompt(prompt_template_path: str, examples_path: str, target_post: str) -> str:
    template = read_text(prompt_template_path).strip()
    examples = read_text(examples_path).strip()
    target_post = target_post.strip()

    examples_block = f"""
--------------------------------------------------
# ACTUAL REFERENCE EXAMPLES

The following are the 25 labeled reference examples that must be used for calibration.

{examples}

--------------------------------------------------
# TARGET POST

The following is the target post to predict.

{target_post}
""".strip()

    # Case 1: template explicitly has placeholders.
    if "{examples}" in template or "{target_post}" in template:
        prompt = template
        prompt = prompt.replace("{examples}", examples)
        prompt = prompt.replace("{target_post}", target_post)

    # Case 2: new prompt has no placeholders.
    else:
        marker = "# OUTPUT FORMAT"
        if marker in template:
            before, after = template.split(marker, 1)
            prompt = before.rstrip() + "\n\n" + examples_block + "\n\n" + marker + after
        else:
            prompt = template + "\n\n" + examples_block

    # Only useful if some old prompt still uses doubled braces.
    prompt = prompt.replace("{{", "{").replace("}}", "}")

    return prompt


def extract_json_from_response(text: str) -> dict | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


# Load model and processor
print("Loading Qwen2.5-VL model...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
)


min_pixels = 128 * 28 * 28
max_pixels = 512 * 28 * 28

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

print("Model loaded.")


# Build prompt and messages
Agent_Instruction_Prompt = build_prompt(
    prompt_template_path=PROMPT_TEMPLATE_PATH,
    examples_path=EXAMPLES_PATH,
    target_post=TARGET_POST,
)

print("Prompt length:", len(Agent_Instruction_Prompt))

image_path_for_qwen = get_local_image_path(IMAGE_PATH)
print("Image path passed to Qwen:", image_path_for_qwen)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path_for_qwen,
            },
            {
                "type": "text",
                "text": Agent_Instruction_Prompt,
            },
        ],
    }
]


# Prepare inputs
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

print("Input token shape:", inputs.input_ids.shape)


# Run inference
print("Running inference...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

raw_response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]


# Print output
print(raw_response)

parsed_json = extract_json_from_response(raw_response)
