"""Local Qwen recipe generation helpers."""

import re

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'


def parse_recipe(text, ingredients, finished, generated_tokens=0):
    """Convert the model's titled, numbered response to the app recipe format."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = ''
    steps = []
    in_steps = False

    for line in lines:
        clean = re.sub(r'^#{1,6}\s*', '', line).strip().strip('*').strip()
        title_match = re.match(r'^(?:title|recipe)\s*:\s*(.+)$', clean, re.IGNORECASE)
        if title_match and not title:
            title = title_match.group(1).strip().strip('*')
            continue
        if re.match(r'^(?:steps|instructions)\s*:?\s*$', clean, re.IGNORECASE):
            in_steps = True
            continue
        if not title and not re.match(r'^(?:ingredients|steps|instructions)\b', clean, re.IGNORECASE):
            title = clean.strip('*')
            continue
        if in_steps or (title and re.match(r'^\d+[.)]\s+', clean)):
            in_steps = True
            step = re.sub(r'^(?:[-*]\s*|\d+[.)]\s*)', '', clean).strip()
            if step:
                steps.append(step)

    return {
        'title': title,
        'ingredients': ingredients,
        'steps': steps,
        'ingredient_coverage': 0.0,
        'mentioned_ingredients': [],
        'generated_tokens': generated_tokens,
        'finished': finished,
    }


def load_qwen(model_id=MODEL_ID, adapter_path=None):
    """Load Qwen on the available device; downloads model weights on first use."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map='auto',
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_qwen_recipe(model, tokenizer, ingredients, max_new_tokens=320):
    import torch

    ingredients_text = ', '.join(ingredients)
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a careful home-cooking assistant. Give practical, safe, '
                'concise recipe instructions. Use the requested ingredients and '
                'do not invent unavailable ingredients beyond basic salt, pepper, '
                'oil, or water. Never claim an unsafe cooking temperature.'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'Create one recipe using these ingredients: {ingredients_text}.\n'
                'Reply in this exact format:\n'
                'Title: <short recipe title>\n'
                'Steps:\n'
                '1. <first concise instruction>\n'
                '2. <next concise instruction>\n'
                'Include 4 to 7 clear steps. Do not include an ingredient list or extra commentary.'
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors='pt')
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: value.to(input_device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs['input_ids'].shape[1]
    generated_ids = output[0, prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return parse_recipe(
        response,
        ingredients,
        finished=tokenizer.eos_token_id in generated_ids.tolist(),
        generated_tokens=len(generated_ids),
    )
