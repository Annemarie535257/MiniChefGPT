"""Checks for turning fine-tuned GPT-2 text into a recipe, without loading GPT-2."""

import unittest

import gpt2_recipes


class GPT2RecipeTests(unittest.TestCase):
    def test_prompt_matches_fine_tuning_format(self):
        self.assertEqual(gpt2_recipes.gpt2_prompt(['pasta', 'basil']), 'Ingredients: pasta, basil\nTitle:')

    def test_parse_generation_reads_title_and_numbered_steps(self):
        text = ' basil pasta\nSteps:\n1. boil the pasta.\n2. toss with basil.\n\n3.  \nnote without a number'
        recipe = gpt2_recipes.parse_generation(text, ['pasta', 'basil', 'garlic'], 20, True)
        self.assertEqual(recipe['title'], 'basil pasta')
        self.assertEqual(recipe['steps'], ['boil the pasta.', 'toss with basil.'])
        self.assertAlmostEqual(recipe['ingredient_coverage'], 2 / 3)
        self.assertEqual(recipe['mentioned_ingredients'], ['pasta', 'basil'])
        self.assertTrue(recipe['finished'])

    def test_parse_generation_without_steps(self):
        recipe = gpt2_recipes.parse_generation(' soup', ['carrot'], 2, False)
        self.assertEqual(recipe['title'], 'soup')
        self.assertEqual(recipe['steps'], [])
        self.assertEqual(recipe['ingredient_coverage'], 0.0)


if __name__ == '__main__':
    unittest.main()
