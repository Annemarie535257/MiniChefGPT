"""Exercise the Streamlit interface without loading or training a model."""

from pathlib import Path
import threading
import tempfile
import unittest
from unittest import mock

import app as recipe_app
import gpt2_recipes
import minichefgpt as chef
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / 'streamlit_app.py'


class RecipeAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.recipe = {
            'title': 'garlic pasta',
            'ingredients': ['pasta', 'tomato', 'garlic', 'basil'],
            'steps': ['Cook the pasta.', 'Add tomato, garlic, and basil.'],
            'finished': True,
        }
        self.load_patch = mock.patch.object(chef, 'load_model', return_value=(object(), object(), chef.Config()))
        self.generate_patch = mock.patch.object(chef, 'generate_recipe', return_value=self.recipe)
        self.load_model = self.load_patch.start()
        self.generate_recipe = self.generate_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.addCleanup(self.generate_patch.stop)
        self.qwen_load_patch = mock.patch.object(
            recipe_app, 'load_qwen_resources',
            return_value=(object(), object(), threading.Lock()),
        )
        self.qwen_generate_patch = mock.patch(
            'qwen_recipe.generate_qwen_recipe', return_value=self.recipe,
        )
        self.qwen_load = self.qwen_load_patch.start()
        self.qwen_generate = self.qwen_generate_patch.start()
        self.addCleanup(self.qwen_load_patch.stop)
        self.addCleanup(self.qwen_generate_patch.stop)
        self.addCleanup(st.cache_resource.clear)
        self.gpt2_recipe = dict(self.recipe, title='tomato basil pasta')
        patches = [
            mock.patch.object(recipe_app, 'gpt2_source', return_value='fine-tuned-gpt2'),
            mock.patch.object(gpt2_recipes, 'load_gpt2', return_value=(object(), object())),
            mock.patch.object(gpt2_recipes, 'generate_gpt2_recipe', return_value=self.gpt2_recipe),
        ]
        self.gpt2_source, self.load_gpt2, self.generate_gpt2 = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)

    def open_app(self, model=recipe_app.MINICHEF):
        app = AppTest.from_file(str(APP), default_timeout=10).run()
        self.assertFalse(app.exception)
        if model:
            app.radio[0].set_value(model).run()
            self.assertFalse(app.exception)
        return app

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertFalse(app.exception)
        return app

    def test_empty_submission_does_not_load_or_generate(self):
        app = self.open_app()
        app.text_area[0].input('  ,\n  ')
        self.click(app, 'Generate recipe')
        self.assertIn('Add at least one ingredient', app.warning[0].value)
        self.load_model.assert_not_called()
        self.generate_recipe.assert_not_called()
        self.qwen_generate.assert_not_called()

    def test_qwen_option_generates_without_the_legacy_checkpoint(self):
        app = self.open_app(model=recipe_app.QWEN)
        app.text_area[0].input('carrot, water')
        self.click(app, 'Generate recipe')
        self.qwen_load.assert_called_once()
        self.qwen_generate.assert_called_once()
        self.assertEqual(self.qwen_generate.call_args.args[2], ['carrot', 'water'])

    def test_fine_tuned_qwen_loads_its_recorded_base_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)
            (adapter / 'minichef_adapter.json').write_text(
                '{"model_id": "Qwen/Qwen2.5-0.5B-Instruct"}', encoding='utf-8',
            )
            (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
            with mock.patch.object(recipe_app, 'QWEN_ADAPTER', adapter):
                app = self.open_app(model=recipe_app.QWEN_FINETUNED)
                app.text_area[0].input('carrot, water')
                self.click(app, 'Generate recipe')
        self.qwen_load.assert_called_once()
        self.assertEqual(self.qwen_load.call_args.args[:2], (
            'Qwen/Qwen2.5-0.5B-Instruct', str(adapter),
        ))
        self.qwen_generate.assert_called_once()

    def test_commas_lines_and_duplicate_names_reach_model_as_one_list(self):
        app = self.open_app()
        app.text_area[0].input('pasta,\n tomato\n- Garlic,\nPASTA, tomato,\n• basil')
        self.click(app, 'Generate recipe')
        self.generate_recipe.assert_called_once()
        self.assertEqual(self.generate_recipe.call_args.args[2], ['pasta', 'tomato', 'Garlic', 'basil'])

    def test_example_selection_fills_input_and_clears_previous_recipe(self):
        app = self.open_app()
        app.session_state['recipe'] = self.recipe
        self.click(app, 'Chicken dinner')
        self.assertEqual(app.text_area[0].value, 'chicken, garlic, onion, tomato')
        self.assertNotIn('recipe', app.session_state)
        self.click(app, 'Vegetable pasta')
        self.assertEqual(app.text_area[0].value, 'pasta, tomato, garlic, basil')
        self.load_model.assert_not_called()
        self.generate_recipe.assert_not_called()

    def test_recipe_and_download_survive_reruns_without_regeneration(self):
        app = self.open_app()
        app.text_area[0].input('pasta, tomato, garlic, basil')
        self.click(app, 'Generate recipe')
        self.assertIn('Garlic pasta', [heading.value for heading in app.subheader])
        self.assertIn('1. Cook the pasta.', [text.value for text in app.markdown])
        self.assertIn('2. Add tomato, garlic, and basil.', [text.value for text in app.markdown])
        self.assertEqual(app.session_state['recipe'], self.recipe)
        self.assertEqual(app.download_button[0].label, 'Download recipe')
        app.download_button[0].click().run()
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['recipe'], self.recipe)
        self.assertIn('Garlic pasta', [heading.value for heading in app.subheader])
        self.load_model.assert_called_once()
        self.generate_recipe.assert_called_once()

    def test_missing_checkpoint_shows_setup_error(self):
        original = Path.is_file
        with mock.patch.object(Path, 'is_file', autospec=True,
                               side_effect=lambda path: False if path.name == 'minichefgpt_best.pt' else original(path)):
            app = self.open_app()
        self.assertIn('MiniChefGPT checkpoint is missing', app.info[0].value)
        submit = next(button for button in app.button if button.label == 'Generate recipe')
        self.assertTrue(submit.disabled)
        self.load_model.assert_not_called()
        self.generate_recipe.assert_not_called()

    def test_generation_validation_error_is_shown_without_stale_recipe(self):
        app = self.open_app()
        app.session_state['recipe'] = self.recipe
        app.text_area[0].input('very long ingredient names')
        self.generate_recipe.side_effect = ValueError('Ingredient list is too long for this model context')
        self.click(app, 'Generate recipe')
        self.assertIn('Try fewer or shorter ingredient names', app.warning[0].value)
        self.assertNotIn('recipe', app.session_state)
        self.assertEqual(len(app.download_button), 0)

    def test_gpt2_choice_uses_fine_tuned_model(self):
        app = self.open_app()
        app.radio[0].set_value('GPT-2 fine-tuned').run()
        app.text_area[0].input('pasta, tomato, garlic, basil')
        self.click(app, 'Generate recipe')
        self.generate_gpt2.assert_called_once()
        self.assertEqual(self.generate_gpt2.call_args.args[2], ['pasta', 'tomato', 'garlic', 'basil'])
        self.load_gpt2.assert_called_once_with('fine-tuned-gpt2', device='cpu')
        self.generate_recipe.assert_not_called()
        self.assertIn('Tomato basil pasta', [heading.value for heading in app.subheader])
        self.assertTrue(any('Generated by GPT-2 fine-tuned' in caption.value for caption in app.caption))
        self.assertEqual(app.session_state['recipe_model'], 'GPT-2 fine-tuned')

    def test_missing_gpt2_model_disables_generation(self):
        self.gpt2_source.return_value = None
        app = self.open_app()
        app.radio[0].set_value('GPT-2 fine-tuned').run()
        self.assertIn('fine-tuned GPT-2 model is not available', app.info[0].value)
        submit = next(button for button in app.button if button.label == 'Generate recipe')
        self.assertTrue(submit.disabled)
        self.load_gpt2.assert_not_called()


if __name__ == '__main__':
    unittest.main()
