"""Exercise the Streamlit interface without loading or training a model."""

from pathlib import Path
import unittest
from unittest import mock

import app as recipe_app
import gpt2_recipes
import qwen3_recipes
import minichefgpt as chef
import streamlit as st
import torch
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
        self.addCleanup(st.cache_resource.clear)
        self.gpt2_recipe = dict(self.recipe, title='tomato basil pasta')
        patches = [
            mock.patch.object(recipe_app, 'gpt2_source', return_value='fine-tuned-gpt2'),
            mock.patch.object(gpt2_recipes, 'load_gpt2', return_value=(object(), object())),
            mock.patch.object(gpt2_recipes, 'generate_gpt2_recipe', return_value=self.gpt2_recipe),
        ]
        self.gpt2_source, self.load_gpt2, self.generate_gpt2 = [patch.start() for patch in patches]
        self.qwen3_recipe = dict(self.recipe, title='basil tomato spaghetti', ingredient_coverage=0.75)
        qwen3_patches = [
            mock.patch.object(recipe_app, 'qwen3_source', return_value='fine-tuned-qwen3'),
            mock.patch.object(qwen3_recipes, 'load_qwen3', return_value=(object(), object())),
            mock.patch.object(qwen3_recipes, 'generate_qwen3_recipe', return_value=self.qwen3_recipe),
        ]
        self.qwen3_source, self.load_qwen3, self.generate_qwen3 = [patch.start() for patch in qwen3_patches]
        patches += qwen3_patches
        # Pretend to be a large machine unless a test says otherwise.
        memory_patch = mock.patch.object(recipe_app, 'memory_limit_gb', return_value=None)
        self.memory = memory_patch.start()
        patches.append(memory_patch)
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
        self.generate_gpt2.assert_not_called()
        self.generate_qwen3.assert_not_called()

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

    def test_qwen3_choice_uses_adapter_model(self):
        app = self.open_app()
        app.radio[0].set_value('Qwen3 fine-tuned').run()
        app.text_area[0].input('pasta, tomato, garlic, basil')
        self.click(app, 'Generate recipe')
        self.generate_qwen3.assert_called_once()
        self.assertEqual(self.generate_qwen3.call_args.args[2], ['pasta', 'tomato', 'garlic', 'basil'])
        self.load_qwen3.assert_called_once_with('fine-tuned-qwen3', device='cpu', dtype=None)
        self.generate_recipe.assert_not_called()
        self.generate_gpt2.assert_not_called()
        self.assertIn('Basil tomato spaghetti', [heading.value for heading in app.subheader])
        self.assertTrue(any('Ingredients mentioned in steps: 75%' in caption.value for caption in app.caption))

    def test_missing_qwen3_model_disables_generation(self):
        self.qwen3_source.return_value = None
        app = self.open_app()
        app.radio[0].set_value('Qwen3 fine-tuned').run()
        self.assertIn('fine-tuned Qwen3 model is not available', app.info[0].value)
        self.assertTrue(next(b for b in app.button if b.label == 'Generate recipe').disabled)

    def test_offers_all_models_without_comparison_or_benchmark_sections(self):
        app = self.open_app()
        self.assertEqual(list(app.radio[0].options),
                         ['MiniChefGPT', 'GPT-2 fine-tuned', 'Qwen3 fine-tuned'])
        app.text_area[0].input('pasta, basil')
        self.click(app, 'Generate recipe')
        headings = [heading.value for heading in app.subheader]
        self.assertNotIn('How the models perform', headings)
        self.assertNotIn('Model comparison', headings)
        self.assertEqual(len(app.dataframe), 0)

    def test_small_server_loads_qwen3_in_16_bit_and_keeps_one_large_model(self):
        self.memory.return_value = 2.7  # about Streamlit Community Cloud's limit
        app = self.open_app()

        def generate_with(model_name):
            app.radio[0].set_value(model_name).run()
            app.text_area[0].input('pasta, basil')
            self.click(app, 'Generate recipe')

        generate_with('GPT-2 fine-tuned')
        generate_with('Qwen3 fine-tuned')
        self.load_qwen3.assert_called_once_with('fine-tuned-qwen3', device='cpu', dtype=torch.bfloat16)
        generate_with('GPT-2 fine-tuned')
        # GPT-2 was unloaded to make room for Qwen3, so it loads again.
        self.assertEqual(self.load_gpt2.call_count, 2)


class Qwen3MemoryGateTests(unittest.TestCase):
    """Qwen3 adapts to the server's memory and falls back to the Hugging Face copy."""

    def setUp(self):
        self.folder = mock.MagicMock()
        self.folder.__truediv__.return_value.is_file.return_value = True
        self.folder.__truediv__.return_value.stat.return_value.st_size = 40_000_000
        self.folder.__str__.return_value = 'models/qwen3/adapter'
        for patch in (mock.patch.object(recipe_app, 'QWEN3_DIR', self.folder),
                      mock.patch.object(recipe_app, 'QWEN3_ADAPTER', '')):
            patch.start()
            self.addCleanup(patch.stop)

    def check(self, setting, memory):
        with mock.patch.object(recipe_app, 'QWEN3_SETTING', setting),                 mock.patch.object(recipe_app, 'memory_limit_gb', return_value=memory):
            return recipe_app.qwen3_source(), recipe_app.missing_message('Qwen3 fine-tuned'), recipe_app.low_memory()

    def test_streamlit_cloud_sized_server_keeps_qwen3_in_low_memory_mode(self):
        source, _, low = self.check('auto', 2.7)
        self.assertEqual(source, 'models/qwen3/adapter')
        self.assertTrue(low)

    def test_very_small_server_turns_qwen3_off_with_explanation(self):
        source, message, _ = self.check('auto', 1.0)
        self.assertIsNone(source)
        self.assertIn('this server has 1.0 GB', message)

    def test_large_or_unknown_memory_uses_full_precision(self):
        for memory in (15.6, None):
            source, _, low = self.check('auto', memory)
            self.assertEqual(source, 'models/qwen3/adapter')
            self.assertFalse(low)

    def test_settings_override_the_memory_check(self):
        self.assertEqual(self.check('on', 1.0)[0], 'models/qwen3/adapter')
        source, message, _ = self.check('off', 64.0)
        self.assertIsNone(source)
        self.assertIn('turned off', message)

    def test_missing_local_folder_or_lfs_pointer_uses_hugging_face(self):
        self.folder.__truediv__.return_value.stat.return_value.st_size = 130  # a Git LFS pointer file
        self.assertEqual(self.check('auto', 2.7)[0], 'Annemarie535257/qwen_model')
        self.folder.__truediv__.return_value.is_file.return_value = False
        self.assertEqual(self.check('auto', None)[0], 'Annemarie535257/qwen_model')

    def test_gpt2_falls_back_to_hugging_face(self):
        missing = mock.MagicMock()
        missing.__truediv__.return_value.is_file.return_value = False
        with mock.patch.object(recipe_app, 'GPT2_DIR', missing), mock.patch.object(recipe_app, 'GPT2_MODEL', ''):
            self.assertEqual(recipe_app.gpt2_source(), 'Annemarie535257/minichef-gpt2')

if __name__ == '__main__':
    unittest.main()
