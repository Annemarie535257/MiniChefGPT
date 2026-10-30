"""Small Streamlit interface for the saved MiniChefGPT recipe model."""
from pathlib import Path
import json
import logging
import os
import re
import secrets
import sys
import threading

ROOT = Path(__file__).resolve().parent
if (ROOT / '.runtime').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime'))

import streamlit as st

CHECKPOINT = ROOT / 'models' / 'refactored' / 'minichefgpt_best.pt'
# A local folder from MiniChefGPT_GPT2_finetune.ipynb, or a Hugging Face model id set by environment.
GPT2_MODEL = os.environ.get('MINICHEF_GPT2_MODEL', '')
GPT2_DIR = ROOT / 'models' / 'gpt2' / 'best'
MINICHEF, GPT2 = 'MiniChefGPT', 'GPT-2 fine-tuned'
QWEN_ADAPTER = ROOT / 'models' / 'qwen-recipe-lora'
QWEN = 'Qwen2.5-1.5B-Instruct (local)'
QWEN_FINETUNED = 'Qwen recipe fine-tune (local)'
MODEL_DESCRIPTIONS = {
    MINICHEF: 'Small Transformer trained from scratch on 9,000 recipes (1M parameters).',
    GPT2: 'Pretrained GPT-2 fine-tuned on the same recipes (124M parameters). Slower, more fluent.',
    QWEN: 'Pretrained Qwen2.5-1.5B-Instruct model. Requires several GB of storage and substantial RAM.',
    QWEN_FINETUNED: 'Qwen2.5 fine-tuned on the recipe data with a LoRA adapter.',
}
LOGGER = logging.getLogger(__name__)
EXAMPLES = {
    'Chicken dinner': 'chicken, garlic, onion, tomato',
    'Vegetable pasta': 'pasta, tomato, garlic, basil',
}


def parse_ingredients(text):
    """Accept commas or line breaks, ignoring duplicate ingredient names."""
    ingredients, seen = [], set()
    for item in re.split(r'[,\n\r]+', text):
        item = re.sub(r'^\s*[-•]\s+', '', item).strip()
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            ingredients.append(item)
    if not ingredients:
        raise ValueError('Add at least one ingredient to get started.')
    if len(ingredients) > 30:
        raise ValueError('Please use 30 ingredients or fewer.')
    if len(text) > 1000:
        raise ValueError('That list is too long. Try fewer or shorter ingredient names.')
    return ingredients


@st.cache_resource(show_spinner=False)
def load_resources(checkpoint_path, checkpoint_version):
    """Cache by file version, with a lock for shared inference across sessions."""
    import torch
    from minichefgpt import load_model

    torch.set_num_threads(2)
    model, tokenizer, _ = load_model(checkpoint_path, device='cpu')
    return model, tokenizer, threading.Lock()


def gpt2_source():
    """Return where the fine-tuned GPT-2 model can be loaded from, or None."""
    if GPT2_MODEL:
        return GPT2_MODEL
    if (GPT2_DIR / 'config.json').is_file() and (GPT2_DIR / 'model.safetensors').is_file():
        return str(GPT2_DIR)
    return None


def qwen_adapter_metadata():
    metadata_path = QWEN_ADAPTER / 'minichef_adapter.json'
    config_path = QWEN_ADAPTER / 'adapter_config.json'
    if not metadata_path.is_file() or not config_path.is_file():
        return None
    return json.loads(metadata_path.read_text(encoding='utf-8'))


@st.cache_resource(show_spinner=False)
def load_gpt2_resources(source, version):
    """Cache the fine-tuned GPT-2 model by source and file version."""
    import torch
    from gpt2_recipes import load_gpt2

    torch.set_num_threads(2)
    model, tokenizer = load_gpt2(source, device='cpu')
    return model, tokenizer, threading.Lock()


def generate(model_name, ingredients):
    seed = secrets.randbelow(2**31)
    if model_name == GPT2:
        source = gpt2_source()
        version = (GPT2_DIR / 'model.safetensors').stat().st_mtime_ns if source == str(GPT2_DIR) else 0
        model, tokenizer, lock = load_gpt2_resources(source, version)
        from gpt2_recipes import generate_gpt2_recipe
        with lock:
            return generate_gpt2_recipe(model, tokenizer, ingredients, candidates=4, seed=seed)
    if model_name in (QWEN, QWEN_FINETUNED):
        from qwen_recipe import MODEL_ID, generate_qwen_recipe

        if model_name == QWEN_FINETUNED:
            metadata = qwen_adapter_metadata()
            if metadata is None:
                raise ValueError('The Qwen recipe adapter is missing. Run python qwen_finetune.py first.')
            model_id = metadata['model_id']
            adapter_path = str(QWEN_ADAPTER)
            adapter_version = (QWEN_ADAPTER / 'adapter_config.json').stat().st_mtime_ns
        else:
            model_id, adapter_path, adapter_version = MODEL_ID, None, None
        model, tokenizer, lock = load_qwen_resources(model_id, adapter_path, adapter_version)
        with lock:
            return generate_qwen_recipe(model, tokenizer, ingredients)
    model, tokenizer, lock = load_resources(str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns)
    from minichefgpt import generate_recipe
    with lock:
        return generate_recipe(model, tokenizer, ingredients, candidates=8, seed=seed)


@st.cache_resource(show_spinner=False)
def load_qwen_resources(model_id, adapter_path=None, adapter_version=None):
    """Cache the local Qwen model and serialize inference across sessions."""
    from qwen_recipe import load_qwen

    model, tokenizer = load_qwen(model_id, adapter_path=adapter_path)
    return model, tokenizer, threading.Lock()


def use_example(name):
    st.session_state['ingredients_input'] = EXAMPLES[name]
    st.session_state.pop('recipe', None)
    st.session_state.pop('recipe_model', None)


def recipe_text(recipe, model_name=MINICHEF):
    lines = [recipe['title'].strip() or 'Recipe idea', f'Generated by {model_name}', '', 'Your ingredients:']
    lines.extend(f'- {item}' for item in recipe['ingredients'])
    lines.extend(['', 'Steps:'])
    lines.extend(f'{number}. {step}' for number, step in enumerate(recipe['steps'], 1))
    if not recipe['finished']:
        lines.extend(['', 'This draft ended before the recipe was complete.'])
    lines.extend(['', 'Experimental recipe draft. Review ingredients, quantities, and cooking instructions before use.'])
    return '\n'.join(lines)


def main():
    st.set_page_config(page_title='MiniChefGPT', page_icon='🥕', layout='wide')
    st.markdown('''
<style>
.block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 2rem; }
h1 { font-family: Georgia, serif; letter-spacing: -0.04em; }
h2, h3 { letter-spacing: -0.02em; }
[data-testid="stForm"] { border: 0; padding: 0; }
.empty-recipe { padding: 4.5rem 1rem; text-align: center; color: #637469; }
.empty-recipe .symbol { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-recipe strong { display: block; color: #294636; font-size: 1.15rem; margin-bottom: 0.5rem; }
</style>
''', unsafe_allow_html=True)
    st.title('MiniChefGPT')
    st.write('What’s in your kitchen? Add a few ingredients and get a recipe idea.')
    st.caption('Experimental recipe drafts. Review ingredients, quantities, and cooking instructions before use.')
    st.write('')

    input_column, output_column = st.columns([1, 1.35], gap='large')
    with input_column:
        with st.container(border=True):
            model_name = st.radio(
                'Model', [MINICHEF, GPT2, QWEN, QWEN_FINETUNED],
                key='model_choice', horizontal=True,
            )
            st.caption(MODEL_DESCRIPTIONS[model_name])
            gpt2_missing = model_name == GPT2 and gpt2_source() is None
            if gpt2_missing:
                st.info('The fine-tuned GPT-2 model is not available here. Run '
                        'MiniChefGPT_GPT2_finetune.ipynb to create models/gpt2/best, '
                        'or set MINICHEF_GPT2_MODEL to a Hugging Face model id.')
            qwen_metadata = qwen_adapter_metadata() if model_name == QWEN_FINETUNED else None
            qwen_adapter_missing = model_name == QWEN_FINETUNED and qwen_metadata is None
            if qwen_adapter_missing:
                st.info('The Qwen recipe adapter is not available here. Run python qwen_finetune.py first.')
            minichef_missing = model_name == MINICHEF and not CHECKPOINT.is_file()
            if minichef_missing:
                st.info('The MiniChefGPT checkpoint is missing. Include models/refactored/minichefgpt_best.pt.')
            st.subheader('Your ingredients')
            st.caption('Try an example, or use what you have.')
            first, second = st.columns(2)
            first.button('Chicken dinner', on_click=use_example, args=('Chicken dinner',), use_container_width=True)
            second.button('Vegetable pasta', on_click=use_example, args=('Vegetable pasta',), use_container_width=True)
            with st.form('recipe_form'):
                text = st.text_area('Ingredients', key='ingredients_input', height=160,
                                    max_chars=1000, placeholder='chicken, garlic, onion, tomato',
                                    help='Separate ingredients with commas or put each on a new line.')
                submitted = st.form_submit_button('Generate recipe', type='primary', use_container_width=True,
                                                  disabled=gpt2_missing or qwen_adapter_missing or minichef_missing)
            if submitted:
                st.session_state.pop('recipe', None)
                st.session_state.pop('recipe_model', None)
                try:
                    ingredients = parse_ingredients(text)
                    with st.spinner(f'Creating your recipe with {model_name}…'):
                        recipe = generate(model_name, ingredients)
                    st.session_state['recipe'] = recipe
                    st.session_state['recipe_model'] = model_name
                except ValueError as error:
                    if 'context' in str(error).lower():
                        st.warning('That ingredient list is too long. Try fewer or shorter ingredient names.')
                    else:
                        st.warning(str(error))
                except Exception:
                    LOGGER.exception('Recipe generation failed')
                    st.error('Something went wrong while creating the recipe. Please try again.')

    with output_column:
        with st.container(border=True):
            recipe = st.session_state.get('recipe')
            if recipe is None:
                st.markdown('''
<div class="empty-recipe">
<div class="symbol">🍽️</div>
<strong>Your recipe will appear here</strong>
Add ingredients, then select Generate recipe.
</div>
''', unsafe_allow_html=True)
            else:
                recipe_model = st.session_state.get('recipe_model', MINICHEF)
                st.subheader(recipe['title'].strip().capitalize() or 'Recipe idea')
                st.caption(f'Generated by {recipe_model} · Your ingredients: ' + ', '.join(recipe['ingredients']))
                st.markdown('**Steps**')
                if recipe['steps']:
                    for number, step in enumerate(recipe['steps'], 1):
                        st.markdown(f'{number}. {step}')
                else:
                    st.info('No instruction steps were produced. Try generating another recipe.')
                if not recipe['finished']:
                    st.info('This draft ended before the recipe was complete. Try generating another recipe.')
                st.divider()
                st.download_button('Download recipe', data=recipe_text(recipe, recipe_model),
                                   file_name='minichefgpt_recipe.txt', mime='text/plain',
                                   use_container_width=True)


if __name__ == '__main__':
    main()
