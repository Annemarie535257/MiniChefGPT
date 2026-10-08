"""Small Streamlit interface for the saved MiniChefGPT recipe model."""
from pathlib import Path
import logging
import re
import secrets
import sys
import threading

ROOT = Path(__file__).resolve().parent
if (ROOT / '.runtime').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime'))

import streamlit as st

CHECKPOINT = ROOT / 'models' / 'refactored' / 'minichefgpt_best.pt'
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


def use_example(name):
    st.session_state['ingredients_input'] = EXAMPLES[name]
    st.session_state.pop('recipe', None)


def recipe_text(recipe):
    lines = [recipe['title'].strip() or 'Recipe idea', '', 'Your ingredients:']
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

    if not CHECKPOINT.is_file():
        st.error('The recipe model is missing. Include models/refactored/minichefgpt_best.pt in the deployed app files.')
        st.stop()

    input_column, output_column = st.columns([1, 1.35], gap='large')
    with input_column:
        with st.container(border=True):
            st.subheader('Your ingredients')
            st.caption('Try an example, or use what you have.')
            first, second = st.columns(2)
            first.button('Chicken dinner', on_click=use_example, args=('Chicken dinner',), use_container_width=True)
            second.button('Vegetable pasta', on_click=use_example, args=('Vegetable pasta',), use_container_width=True)
            with st.form('recipe_form'):
                text = st.text_area('Ingredients', key='ingredients_input', height=160,
                                    max_chars=1000, placeholder='chicken, garlic, onion, tomato',
                                    help='Separate ingredients with commas or put each on a new line.')
                submitted = st.form_submit_button('Generate recipe', type='primary', use_container_width=True)
            if submitted:
                st.session_state.pop('recipe', None)
                try:
                    ingredients = parse_ingredients(text)
                    with st.spinner('Creating your recipe…'):
                        model, tokenizer, lock = load_resources(str(CHECKPOINT), CHECKPOINT.stat().st_mtime_ns)
                        from minichefgpt import generate_recipe
                        with lock:
                            recipe = generate_recipe(model, tokenizer, ingredients,
                                                     candidates=8, seed=secrets.randbelow(2**31))
                    st.session_state['recipe'] = recipe
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
                st.subheader(recipe['title'].strip().capitalize() or 'Recipe idea')
                st.caption('Your ingredients: ' + ', '.join(recipe['ingredients']))
                st.markdown('**Steps**')
                if recipe['steps']:
                    for number, step in enumerate(recipe['steps'], 1):
                        st.markdown(f'{number}. {step}')
                else:
                    st.info('No instruction steps were produced. Try generating another recipe.')
                if not recipe['finished']:
                    st.info('This draft ended before the recipe was complete. Try generating another recipe.')
                st.divider()
                st.download_button('Download recipe', data=recipe_text(recipe),
                                   file_name='minichefgpt_recipe.txt', mime='text/plain',
                                   use_container_width=True)


if __name__ == '__main__':
    main()
