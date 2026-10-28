"""Recipe generation with the GPT-2 model fine-tuned in MiniChefGPT_GPT2_finetune.ipynb."""
import re

import torch

from minichefgpt import clean_text, ingredient_coverage


def gpt2_prompt(ingredients):
    return 'Ingredients: ' + ', '.join(ingredients) + '\nTitle:'


def parse_generation(text, ingredients, generated_tokens, finished):
    title, _, body = text.partition('\n')
    steps = [re.sub(r'^\s*\d+\.\s*', '', line).strip()
             for line in body.splitlines() if re.match(r'^\s*\d+\.', line)]
    steps = [step for step in steps if step]
    coverage, matched = ingredient_coverage(ingredients, ' '.join(steps))
    return {'title': title.strip(), 'ingredients': ingredients, 'steps': steps,
            'ingredient_coverage': coverage, 'mentioned_ingredients': matched,
            'generated_tokens': generated_tokens, 'finished': finished}


def load_gpt2(source, device='cpu'):
    """Load a fine-tuned model from a local folder or a Hugging Face model id."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source).to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_gpt2_recipe(model, tokenizer, ingredients, candidates=8, seed=42, max_new_tokens=256,
                         temperature=0.7, top_p=0.9, top_k=50, no_repeat_ngram_size=5):
    """Sample several drafts and keep the one that mentions the most requested ingredients."""
    if isinstance(ingredients, str):
        ingredients = [part.strip() for part in ingredients.split(',') if part.strip()]
    ingredients = [clean_text(item) for item in ingredients if clean_text(item)]
    if not ingredients or len(ingredients) > 30:
        raise ValueError('Provide between 1 and 30 ingredient names')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    prompt = tokenizer(gpt2_prompt(ingredients), return_tensors='pt').to(device)
    torch.manual_seed(seed)
    outputs = model.generate(**prompt, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                             max_new_tokens=max_new_tokens, no_repeat_ngram_size=no_repeat_ngram_size,
                             num_return_sequences=candidates, pad_token_id=tokenizer.eos_token_id)
    model.train(was_training)
    recipes = []
    for sequence in outputs:
        new_tokens = sequence[prompt.input_ids.shape[1]:].tolist()
        finished = tokenizer.eos_token_id in new_tokens
        if finished:
            new_tokens = new_tokens[:new_tokens.index(tokenizer.eos_token_id)]
        recipes.append(parse_generation(tokenizer.decode(new_tokens), ingredients,
                                        len(new_tokens) + finished, finished))
    return max(recipes, key=lambda r: (r['ingredient_coverage'], bool(r['steps']), r['finished']))
