"""Streamlit interface for the MiniChefGPT, fine-tuned GPT-2, and fine-tuned Qwen3 recipe models."""
from pathlib import Path
import logging
import os
import re
import secrets
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
if (ROOT / '.runtime').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime'))

import streamlit as st

CHECKPOINT = ROOT / 'models' / 'refactored' / 'minichefgpt_best.pt'
# Each model loads from an environment setting if given, else the local folder from its notebook,
# else the copy published on Hugging Face.
GPT2_MODEL = os.environ.get('MINICHEF_GPT2_MODEL', '')
GPT2_DIR = ROOT / 'models' / 'gpt2' / 'best'
GPT2_HUB_ID = 'Annemarie535257/minichef-gpt2'
QWEN3_ADAPTER = os.environ.get('MINICHEF_QWEN3_ADAPTER', '')
QWEN3_DIR = ROOT / 'models' / 'qwen3' / 'adapter'
QWEN3_HUB_ID = 'Annemarie535257/qwen_model'
# 'auto' adapts to the server's memory; 'off' hides Qwen3; 'on' skips the memory check.
QWEN3_SETTING = os.environ.get('MINICHEF_QWEN3', 'auto').strip().lower()
# Measured peaks for Qwen3 on CPU: about 3.5 GB in 32-bit and 2.0 GB in 16-bit (bfloat16).
FULL_MEMORY_GB = 6.0       # below this, Qwen3 loads in 16-bit and only one large model stays in memory
QWEN3_MIN_MEMORY_GB = 2.4  # below this, Qwen3 is turned off
CPU_THREADS = int(os.environ.get('MINICHEF_THREADS', min(4, os.cpu_count() or 2)))

MINICHEF, GPT2, QWEN3 = 'MiniChefGPT', 'GPT-2 fine-tuned', 'Qwen3 fine-tuned'
MODEL_NAMES = [MINICHEF, GPT2, QWEN3]
MODEL_DESCRIPTIONS = {
    MINICHEF: 'Small Transformer trained from scratch on 9,000 recipes (1M parameters). Fastest.',
    GPT2: 'Pretrained GPT-2 fine-tuned on the same recipes (124M parameters). Slower, more fluent.',
    QWEN3: ('Qwen3-0.6B chat model with a LoRA adapter trained on the same recipes (600M parameters). Slowest; '
            'the first request downloads the 1.5 GB base model.'),
}
MISSING_MESSAGES = {
    GPT2: ('The fine-tuned GPT-2 model is not available here. Run MiniChefGPT_GPT2_finetune.ipynb to create '
           'models/gpt2/best, or set MINICHEF_GPT2_MODEL to a Hugging Face model id.'),
    QWEN3: ('The fine-tuned Qwen3 model is not available here. Run MiniChefGPT_Qwen3_Training.ipynb to create '
            'models/qwen3/adapter, or set MINICHEF_QWEN3_ADAPTER to a Hugging Face adapter id.'),
}
# Drafts ranked per request; the larger models are slower per draft.
CANDIDATES = {MINICHEF: 8, GPT2: 4, QWEN3: 2}
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

    torch.set_num_threads(CPU_THREADS)
    model, tokenizer, _ = load_model(checkpoint_path, device='cpu')
    return model, tokenizer, threading.Lock()


def gpt2_source():
    """Return where the fine-tuned GPT-2 model can be loaded from, or None."""
    if GPT2_MODEL:
        return GPT2_MODEL
    weights = GPT2_DIR / 'model.safetensors'
    # A Git LFS pointer is a tiny text file; the real weights are about 500 MB.
    if (GPT2_DIR / 'config.json').is_file() and weights.is_file() and weights.stat().st_size > 1_000_000:
        return str(GPT2_DIR)
    return GPT2_HUB_ID


@st.cache_resource(show_spinner=False)
def load_gpt2_resources(source, version):
    """Cache the fine-tuned GPT-2 model by source and file version."""
    import torch
    from gpt2_recipes import load_gpt2

    torch.set_num_threads(CPU_THREADS)
    model, tokenizer = load_gpt2(source, device='cpu')
    return model, tokenizer, threading.Lock()


def memory_limit_gb():
    """Smallest of the machine's RAM and any container (cgroup) memory limit, or None if unknown."""
    limits = []
    try:
        limits.append(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (AttributeError, ValueError, OSError):
        pass
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = Path(path).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            limits.append(int(value))
    return min(limits) / 1024**3 if limits else None


def low_memory():
    """True on small servers such as Streamlit Community Cloud."""
    memory = memory_limit_gb()
    return memory is not None and memory < FULL_MEMORY_GB


def qwen3_memory_blocked():
    """Return the memory limit in GB when it is too small even for 16-bit Qwen3, otherwise None."""
    if QWEN3_SETTING == 'on':
        return None
    memory = memory_limit_gb()
    return memory if memory is not None and memory < QWEN3_MIN_MEMORY_GB else None


def qwen3_source():
    """Return where the Qwen3 LoRA adapter can be loaded from, or None. The base model downloads separately."""
    if QWEN3_SETTING == 'off' or qwen3_memory_blocked() is not None:
        return None
    if QWEN3_ADAPTER:
        return QWEN3_ADAPTER
    weights = QWEN3_DIR / 'adapter_model.safetensors'
    if (QWEN3_DIR / 'adapter_config.json').is_file() and weights.is_file() and weights.stat().st_size > 1_000_000:
        return str(QWEN3_DIR)
    return QWEN3_HUB_ID


@st.cache_resource(show_spinner=False)
def load_qwen3_resources(source, version, half_precision):
    """Cache Qwen3 with its adapter merged, by source, file version, and precision."""
    import torch
    from qwen3_recipes import load_qwen3

    torch.set_num_threads(CPU_THREADS)
    model, tokenizer = load_qwen3(source, device='cpu', dtype=torch.bfloat16 if half_precision else None)
    return model, tokenizer, threading.Lock()


def keep_one_large_model(model_name):
    """On small servers, unload the other large model before loading this one."""
    if not low_memory():
        return
    other = {GPT2: load_qwen3_resources, QWEN3: load_gpt2_resources}.get(model_name)
    if other is not None:
        other.clear()
        import gc
        gc.collect()


def model_available(model_name):
    if model_name == GPT2:
        return gpt2_source() is not None
    if model_name == QWEN3:
        return qwen3_source() is not None
    return CHECKPOINT.is_file()


def missing_message(model_name):
    if model_name == QWEN3:
        if QWEN3_SETTING == 'off':
            return 'Qwen3 is turned off on this server (MINICHEF_QWEN3 = "off").'
        memory = qwen3_memory_blocked()
        if memory is not None:
            return (f'Qwen3 needs about 2 GB of memory, and this server has {memory:.1f} GB, so it is turned off '
                    'here to keep the app from crashing. Run the app on your own computer to use Qwen3.')
    return MISSING_MESSAGES[model_name]


def file_version(source, local_dir, weights):
    return (local_dir / weights).stat().st_mtime_ns if source == str(local_dir) else 0


def generate(model_name, ingredients):
    seed = secrets.randbelow(2**31)
    candidates = CANDIDATES[model_name]
    keep_one_large_model(model_name)
    if model_name == GPT2:
        source = gpt2_source()
        model, tokenizer, lock = load_gpt2_resources(source, file_version(source, GPT2_DIR, 'model.safetensors'))
        from gpt2_recipes import generate_gpt2_recipe
        with lock:
            return generate_gpt2_recipe(model, tokenizer, ingredients, candidates=candidates, seed=seed)
    if model_name == QWEN3:
        source = qwen3_source()
        model, tokenizer, lock = load_qwen3_resources(
            source, file_version(source, QWEN3_DIR, 'adapter_model.safetensors'), low_memory())
        from qwen3_recipes import generate_qwen3_recipe
        with lock:
            return generate_qwen3_recipe(model, tokenizer, ingredients, candidates=candidates, seed=seed)
    model, tokenizer, lock = load_resources(str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns)
    from minichefgpt import generate_recipe
    with lock:
        return generate_recipe(model, tokenizer, ingredients, candidates=candidates, seed=seed)


def timed_generate(model_name, ingredients):
    start = time.perf_counter()
    recipe = generate(model_name, ingredients)
    return recipe, time.perf_counter() - start


def clear_results():
    for key in ('recipe', 'recipe_model', 'recipe_seconds'):
        st.session_state.pop(key, None)


def use_example(name):
    st.session_state['ingredients_input'] = EXAMPLES[name]
    clear_results()


def recipe_text(recipe, model_name=MINICHEF):
    lines = [recipe['title'].strip() or 'Recipe idea', f'Generated by {model_name}', '', 'Your ingredients:']
    lines.extend(f'- {item}' for item in recipe['ingredients'])
    lines.extend(['', 'Steps:'])
    lines.extend(f'{number}. {step}' for number, step in enumerate(recipe['steps'], 1))
    if not recipe['finished']:
        lines.extend(['', 'This draft ended before the recipe was complete.'])
    lines.extend(['', 'Experimental recipe draft. Review ingredients, quantities, and cooking instructions before use.'])
    return '\n'.join(lines)


def percent(value):
    return '—' if value is None else f'{value:.0%}'


def recipe_stats(recipe, seconds):
    parts = []
    if recipe.get('ingredient_coverage') is not None:
        parts.append(f"Ingredients mentioned in steps: {percent(recipe['ingredient_coverage'])}")
    if seconds is not None:
        parts.append(f'{seconds:.1f} s')
    return ' · '.join(parts)


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

    if not CHECKPOINT.is_file():
        st.error('The recipe model is missing. Include models/refactored/minichefgpt_best.pt in the deployed app files.')
        st.stop()

    input_column, output_column = st.columns([1, 1.35], gap='large')
    with input_column:
        with st.container(border=True):
            model_name = st.radio('Model', MODEL_NAMES, key='model_choice', horizontal=True)
            st.caption(MODEL_DESCRIPTIONS[model_name])
            model_missing = not model_available(model_name)
            if model_missing:
                st.info(missing_message(model_name))
            st.subheader('Your ingredients')
            st.caption('Try an example, or use what you have.')
            first, second = st.columns(2)
            first.button('Chicken dinner', on_click=use_example, args=('Chicken dinner',), width='stretch')
            second.button('Vegetable pasta', on_click=use_example, args=('Vegetable pasta',), width='stretch')
            with st.form('recipe_form'):
                text = st.text_area('Ingredients', key='ingredients_input', height=160,
                                    max_chars=1000, placeholder='chicken, garlic, onion, tomato',
                                    help='Separate ingredients with commas or put each on a new line.')
                submitted = st.form_submit_button('Generate recipe', type='primary', width='stretch',
                                                  disabled=model_missing)
            if submitted:
                clear_results()
                try:
                    ingredients = parse_ingredients(text)
                    with st.spinner(f'Creating your recipe with {model_name}…'):
                        recipe, seconds = timed_generate(model_name, ingredients)
                    st.session_state['recipe'] = recipe
                    st.session_state['recipe_model'] = model_name
                    st.session_state['recipe_seconds'] = seconds
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
                stats = recipe_stats(recipe, st.session_state.get('recipe_seconds'))
                if stats:
                    st.caption(stats)
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
                                   width='stretch')


if __name__ == '__main__':
    main()
