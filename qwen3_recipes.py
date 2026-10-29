"""Prompts, loading, and generation for the Qwen3-0.6B LoRA model from MiniChefGPT_Qwen3_Training.ipynb.

Training and inference must use the same prompt, so both the notebook and the app import it from here.
"""
import json
import re
from pathlib import Path

import torch

from minichefgpt import clean_text, ingredient_coverage

BASE_MODEL = 'Qwen/Qwen3-0.6B'
# Kept short because every prompt token costs training time on a CPU.
SYSTEM_PROMPT = ('You are MiniChefGPT, a home cooking assistant. Write one sensible recipe that uses '
                 'the given ingredients, with a short title and numbered steps.')


def user_message(ingredients):
    if not ingredients:
        raise ValueError('Provide at least one ingredient.')
    return 'Create a recipe using these ingredients: ' + ', '.join(ingredients)


def assistant_answer(row):
    return 'Title: ' + row['title'] + '\nSteps:\n' + '\n'.join(
        f'{number}. {step}' for number, step in enumerate(row['steps'], 1))


def prompt_text(tokenizer, ingredients):
    """Chat prompt ending where the assistant's answer starts; Qwen3 thinking is turned off."""
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM_PROMPT},
         {'role': 'user', 'content': user_message(ingredients)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse_recipe(text, ingredients, generated_tokens, finished):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    title_match = re.search(r'(?im)^\s*title\s*:\s*(.+)$', text)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = text.splitlines()[0].strip() if text else ''
    # Untrained Qwen3 sometimes formats titles as markdown or quotes.
    title = title.strip().strip('*#"').strip()
    parts = re.split(r'(?im)^\s*steps\s*:\s*$', text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else '\n'.join(text.splitlines()[1:])
    steps = [re.sub(r'^\s*\d+[.)]\s*', '', line).strip()
             for line in body.splitlines() if re.match(r'^\s*\d+[.)]', line)]
    steps = [step for step in steps if step]
    # The same score as MiniChefGPT and GPT-2: requested ingredients mentioned in the steps.
    coverage, matched = ingredient_coverage(ingredients, ' '.join(steps))
    return {'title': title, 'ingredients': ingredients, 'steps': steps,
            'ingredient_coverage': coverage, 'mentioned_ingredients': matched,
            'generated_tokens': generated_tokens, 'finished': finished, 'raw_text': text}


def load_qwen3(adapter, device='cpu', merge=True, dtype=None):
    """Load the base model named in the adapter, apply the LoRA adapter, and merge it for faster inference.

    `adapter` is a local folder or a Hugging Face id. `dtype=torch.bfloat16` roughly halves CPU memory
    (about 2 GB peak instead of 3.5 GB) with the same recipes; the default is 32-bit on CPU.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = str(adapter)
    config_path = Path(adapter) / 'adapter_config.json'
    base = json.loads(config_path.read_text(encoding='utf-8'))['base_model_name_or_path'] \
        if config_path.is_file() else BASE_MODEL
    if dtype is None:
        dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, adapter)
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def end_token_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, (list, tuple)) else [ids])
    return ids | {tokenizer.eos_token_id}


@torch.inference_mode()
def generate_qwen3_recipe(model, tokenizer, ingredients, candidates=8, seed=42, max_new_tokens=320,
                          temperature=0.7, top_p=0.8, top_k=20, no_repeat_ngram_size=5):
    """Sample several drafts and keep the one that mentions the most requested ingredients."""
    if isinstance(ingredients, str):
        ingredients = [part.strip() for part in ingredients.split(',') if part.strip()]
    ingredients = [clean_text(item) for item in ingredients if clean_text(item)]
    if not ingredients or len(ingredients) > 30:
        raise ValueError('Provide between 1 and 30 ingredient names')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    prompt = tokenizer(prompt_text(tokenizer, ingredients), return_tensors='pt',
                       add_special_tokens=False).to(device)
    stop = end_token_ids(model, tokenizer)
    torch.manual_seed(seed)
    outputs = model.generate(**prompt, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                             max_new_tokens=max_new_tokens, no_repeat_ngram_size=no_repeat_ngram_size,
                             num_return_sequences=candidates, eos_token_id=sorted(stop),
                             pad_token_id=tokenizer.pad_token_id)
    model.train(was_training)
    recipes = []
    for sequence in outputs:
        new_tokens = sequence[prompt['input_ids'].shape[1]:].tolist()
        end = next((i for i, token in enumerate(new_tokens) if token in stop), None)
        finished = end is not None
        if finished:
            new_tokens = new_tokens[:end]
        recipes.append(parse_recipe(tokenizer.decode(new_tokens, skip_special_tokens=True), ingredients,
                                    len(new_tokens) + finished, finished))
    return max(recipes, key=lambda r: (r['ingredient_coverage'], bool(r['steps']), r['finished']))
