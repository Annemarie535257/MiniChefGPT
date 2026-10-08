"""Exercise the Streamlit interface without loading or training a model."""

from pathlib import Path
import unittest
from unittest import mock

import minichefgpt as chef
import streamlit as st
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / 'app.py'


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

    def open_app(self):
        app = AppTest.from_file(str(APP), default_timeout=10).run()
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
        self.assertIn('recipe model is missing', app.error[0].value)
        self.assertEqual(len(app.text_area), 0)
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


if __name__ == '__main__':
    unittest.main()
