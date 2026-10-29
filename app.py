"""Streamlit interface for the MiniChefGPT, fine-tuned GPT-2, and fine-tuned Qwen3 recipe models."""
from pathlib import Path
import json
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
# Local folders from the fine-tuning notebooks, or Hugging Face ids set by environment.
GPT2_MODEL = os.environ.get('MINICHEF_GPT2_MODEL', '')
GPT2_DIR = ROOT / 'models' / 'gpt2' / 'best'
QWEN3_ADAPTER = os.environ.get('MINICHEF_QWEN3_ADAPTER', '')
QWEN3_DIR = ROOT / 'models' / 'qwen3' / 'adapter'
CPU_THREADS = int(os.environ.get('MINICHEF_THREADS', min(4, os.cpu_count() or 2)))

MINICHEF, GPT2, QWEN3, COMPARE = 'MiniChefGPT', 'GPT-2 fine-tuned', 'Qwen3 fine-tuned', 'Compare all models'
MODEL_NAMES = [MINICHEF, GPT2, QWEN3]
MODEL_DESCRIPTIONS = {
    MINICHEF: 'Small Transformer trained from scratch on 9,000 recipes (1M parameters). Fastest.',
    GPT2: 'Pretrained GPT-2 fine-tuned on the same recipes (124M parameters). Slower, more fluent.',
    QWEN3: 'Qwen3-0.6B chat model with a LoRA adapter trained on the same recipes (600M parameters). Slowest.',
    COMPARE: 'Run the same ingredients through every available model and compare them side by side.',
}
MISSING_MESSAGES = {
    GPT2: ('The fine-tuned GPT-2 model is not available here. Run MiniChefGPT_GPT2_finetune.ipynb to create '
           'models/gpt2/best, or set MINICHEF_GPT2_MODEL to a Hugging Face model id.'),
    QWEN3: ('The fine-tuned Qwen3 model is not available here. Run MiniChefGPT_Qwen3_Training.ipynb to create '
            'models/qwen3/adapter, or set MINICHEF_QWEN3_ADAPTER to a Hugging Face adapter id.'),
}
# Drafts ranked per request; the larger models are slower per draft.
CANDIDATES = {MINICHEF: 8, GPT2: 4, QWEN3: 2}
RESULT_DIRS = {MINICHEF: ROOT / 'models' / 'refactored', GPT2: ROOT / 'models' / 'gpt2',
               QWEN3: ROOT / 'models' / 'qwen3'}
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
    if (GPT2_DIR / 'config.json').is_file() and (GPT2_DIR / 'model.safetensors').is_file():
        return str(GPT2_DIR)
    return None


@st.cache_resource(show_spinner=False)
def load_gpt2_resources(source, version):
    """Cache the fine-tuned GPT-2 model by source and file version."""
    import torch
    from gpt2_recipes import load_gpt2

    torch.set_num_threads(CPU_THREADS)
    model, tokenizer = load_gpt2(source, device='cpu')
    return model, tokenizer, threading.Lock()


def qwen3_source():
    """Return where the Qwen3 LoRA adapter can be loaded from, or None. The base model downloads separately."""
    if QWEN3_ADAPTER:
        return QWEN3_ADAPTER
    if (QWEN3_DIR / 'adapter_config.json').is_file() and (QWEN3_DIR / 'adapter_model.safetensors').is_file():
        return str(QWEN3_DIR)
    return None


@st.cache_resource(show_spinner=False)
def load_qwen3_resources(source, version):
    """Cache Qwen3 with its adapter merged, by source and file version."""
    import torch
    from qwen3_recipes import load_qwen3

    torch.set_num_threads(CPU_THREADS)
    model, tokenizer = load_qwen3(source, device='cpu')
    return model, tokenizer, threading.Lock()


def model_available(model_name):
    if model_name == GPT2:
        return gpt2_source() is not None
    if model_name == QWEN3:
        return qwen3_source() is not None
    return CHECKPOINT.is_file()


def file_version(source, local_dir, weights):
    return (local_dir / weights).stat().st_mtime_ns if source == str(local_dir) else 0


def generate(model_name, ingredients):
    seed = secrets.randbelow(2**31)
    candidates = CANDIDATES[model_name]
    if model_name == GPT2:
        source = gpt2_source()
        model, tokenizer, lock = load_gpt2_resources(source, file_version(source, GPT2_DIR, 'model.safetensors'))
        from gpt2_recipes import generate_gpt2_recipe
        with lock:
            return generate_gpt2_recipe(model, tokenizer, ingredients, candidates=candidates, seed=seed)
    if model_name == QWEN3:
        source = qwen3_source()
        model, tokenizer, lock = load_qwen3_resources(
            source, file_version(source, QWEN3_DIR, 'adapter_model.safetensors'))
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
    seconds = time.perf_counter() - start
    st.session_state.setdefault('history', []).append({
        'model': model_name, 'coverage': recipe.get('ingredient_coverage'),
        'seconds': seconds, 'finished': recipe['finished'], 'steps': len(recipe['steps'])})
    return recipe, seconds


def clear_results():
    for key in ('recipe', 'recipe_model', 'recipe_seconds', 'comparison'):
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


@st.cache_data(show_spinner=False)
def saved_benchmarks():
    """Summarize each model's recorded outputs on the same 12 unseen test recipes and 4 custom inputs."""
    rows = []
    for name in MODEL_NAMES:
        folder = RESULT_DIRS[name]
        try:
            heldout = json.loads((folder / 'heldout_generation.json').read_text(encoding='utf-8'))
            custom = list(json.loads((folder / 'custom_recipes.json').read_text(encoding='utf-8')).values())
        except (OSError, ValueError):
            continue
        if not heldout or not custom:
            continue
        rows.append({
            'Model': name,
            'Unseen recipes: ingredients mentioned': percent(sum(r['ingredient_coverage'] for r in heldout) / len(heldout)),
            'Unseen recipes: finished': percent(sum(bool(r['finished']) for r in heldout) / len(heldout)),
            'Unseen recipes: average steps': round(sum(len(r['steps']) for r in heldout) / len(heldout), 1),
            'Custom inputs: ingredients mentioned': percent(sum(r['ingredient_coverage'] for r in custom) / len(custom)),
        })
    return rows


def session_summary(history):
    rows = []
    for name in MODEL_NAMES:
        runs = [item for item in history if item['model'] == name]
        if not runs:
            continue
        scored = [item['coverage'] for item in runs if item['coverage'] is not None]
        rows.append({
            'Model': name, 'Recipes': len(runs),
            'Ingredients mentioned': percent(sum(scored) / len(scored) if scored else None),
            'Finished': percent(sum(item['finished'] for item in runs) / len(runs)),
            'Average steps': round(sum(item['steps'] for item in runs) / len(runs), 1),
            'Average time (s)': round(sum(item['seconds'] for item in runs) / len(runs), 1),
        })
    return rows


def show_comparison(results):
    st.subheader('Model comparison')
    st.caption('Same ingredients, one request per model. Each model ranks its own drafts '
               f"({', '.join(f'{name}: {CANDIDATES[name]}' for name in MODEL_NAMES)}).")
    columns = st.columns(len(results), gap='medium')
    for column, result in zip(columns, results):
        with column, st.container(border=True):
            st.markdown(f"**{result['model']}**")
            if 'error' in result:
                st.info(result['error'])
                continue
            recipe = result['recipe']
            st.markdown(f"#### {recipe['title'].strip().capitalize() or 'Recipe idea'}")
            st.caption(recipe_stats(recipe, result['seconds']))
            if recipe['steps']:
                st.markdown('\n'.join(f'{number}. {step}' for number, step in enumerate(recipe['steps'], 1)))
            else:
                st.info('No instruction steps were produced.')
            if not recipe['finished']:
                st.caption('This draft ended before the recipe was complete.')


def show_performance():
    st.divider()
    st.subheader('How the models perform')
    benchmarks = saved_benchmarks()
    if benchmarks:
        st.markdown('**Saved benchmark**')
        st.caption('Recorded in each training notebook: the same 12 unseen test recipes (1 draft each) and '
                   '4 custom ingredient lists (best of 8 drafts). "Ingredients mentioned" counts requested '
                   'ingredients named in the steps; it does not check whether a recipe is correct.')
        st.dataframe(benchmarks, hide_index=True, width='stretch')
    history = st.session_state.get('history', [])
    if history:
        st.markdown('**Your tests in this session**')
        st.dataframe(session_summary(history), hide_index=True, width='stretch')


def main():
    st.set_page_config(page_title='MiniChefGPT', page_icon='🥕', layout='wide')
    st.markdown('''
<style>
.block-container { max-width: 1180px; padding-top: 3rem; padding-bottom: 2rem; }
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
            model_name = st.radio('Model', MODEL_NAMES + [COMPARE], key='model_choice', horizontal=True)
            st.caption(MODEL_DESCRIPTIONS[model_name])
            model_missing = model_name != COMPARE and not model_available(model_name)
            if model_missing:
                st.info(MISSING_MESSAGES[model_name])
            elif model_name == COMPARE:
                unavailable = [name for name in MODEL_NAMES if not model_available(name)]
                if unavailable:
                    st.caption('Not available here, so skipped: ' + ', '.join(unavailable) + '.')
            st.subheader('Your ingredients')
            st.caption('Try an example, or use what you have.')
            first, second = st.columns(2)
            first.button('Chicken dinner', on_click=use_example, args=('Chicken dinner',), width='stretch')
            second.button('Vegetable pasta', on_click=use_example, args=('Vegetable pasta',), width='stretch')
            with st.form('recipe_form'):
                text = st.text_area('Ingredients', key='ingredients_input', height=160,
                                    max_chars=1000, placeholder='chicken, garlic, onion, tomato',
                                    help='Separate ingredients with commas or put each on a new line.')
                submitted = st.form_submit_button('Compare models' if model_name == COMPARE else 'Generate recipe',
                                                  type='primary', width='stretch', disabled=model_missing)
            if submitted:
                clear_results()
                try:
                    ingredients = parse_ingredients(text)
                    if model_name == COMPARE:
                        results = []
                        for name in MODEL_NAMES:
                            if not model_available(name):
                                results.append({'model': name, 'error': 'Not available here.'})
                                continue
                            try:
                                with st.spinner(f'Creating a recipe with {name}…'):
                                    recipe, seconds = timed_generate(name, ingredients)
                                results.append({'model': name, 'recipe': recipe, 'seconds': seconds})
                            except ValueError:
                                results.append({'model': name, 'error': 'This ingredient list is too long for this model.'})
                            except Exception:
                                LOGGER.exception('Recipe generation failed for %s', name)
                                results.append({'model': name, 'error': 'Something went wrong with this model.'})
                        st.session_state['comparison'] = results
                    else:
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

    comparison = st.session_state.get('comparison')
    with output_column:
        with st.container(border=True):
            recipe = st.session_state.get('recipe')
            if comparison is not None:
                st.subheader('Comparison summary')
                st.dataframe([{
                    'Model': result['model'],
                    'Title': result['recipe']['title'].strip().capitalize() if 'recipe' in result else '—',
                    'Ingredients mentioned': percent(result['recipe'].get('ingredient_coverage')) if 'recipe' in result else '—',
                    'Steps': len(result['recipe']['steps']) if 'recipe' in result else '—',
                    'Time (s)': round(result['seconds'], 1) if 'recipe' in result else '—',
                } for result in comparison], hide_index=True, width='stretch')
                st.caption('Each model’s full recipe is shown below.')
            elif recipe is None:
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

    if comparison is not None:
        show_comparison(comparison)
    show_performance()


if __name__ == '__main__':
    main()
