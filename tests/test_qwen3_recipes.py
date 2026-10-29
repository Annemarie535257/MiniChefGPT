"""Checks for the Qwen3 prompt and answer parsing, without loading Qwen3."""

import unittest

import qwen3_recipes


class FakeChatTokenizer:
    def apply_chat_template(self, messages, **options):
        self.messages, self.options = messages, options
        return 'PROMPT'


class Qwen3RecipeTests(unittest.TestCase):
    def test_prompt_uses_system_and_user_messages_without_thinking(self):
        tokenizer = FakeChatTokenizer()
        self.assertEqual(qwen3_recipes.prompt_text(tokenizer, ['pasta', 'basil']), 'PROMPT')
        self.assertEqual([m['role'] for m in tokenizer.messages], ['system', 'user'])
        self.assertEqual(tokenizer.messages[1]['content'], 'Create a recipe using these ingredients: pasta, basil')
        self.assertEqual(tokenizer.options, {'tokenize': False, 'add_generation_prompt': True,
                                             'enable_thinking': False})

    def test_answer_format_matches_training_rows(self):
        row = {'title': 'basil pasta', 'steps': ['boil the pasta.', 'toss with basil.']}
        self.assertEqual(qwen3_recipes.assistant_answer(row),
                         'Title: basil pasta\nSteps:\n1. boil the pasta.\n2. toss with basil.')

    def test_parse_recipe_reads_title_steps_and_scores_steps_only(self):
        text = '<think>\n\n</think>\nTitle: Garlic Pasta\nSteps:\n1. boil the pasta.\n2) add garlic.\nEnjoy!'
        recipe = qwen3_recipes.parse_recipe(text, ['pasta', 'garlic', 'basil'], 30, True)
        self.assertEqual(recipe['title'], 'Garlic Pasta')
        self.assertEqual(recipe['steps'], ['boil the pasta.', 'add garlic.'])
        self.assertAlmostEqual(recipe['ingredient_coverage'], 2 / 3)
        self.assertEqual(recipe['mentioned_ingredients'], ['pasta', 'garlic'])
        self.assertNotIn('<think>', recipe['raw_text'])

    def test_parse_recipe_strips_markdown_from_title(self):
        recipe = qwen3_recipes.parse_recipe('Title: **Potato Egg Stew**\nSteps:\n1. boil.', ['potato'], 9, True)
        self.assertEqual(recipe['title'], 'Potato Egg Stew')

    def test_title_mentions_do_not_count_as_ingredient_use(self):
        recipe = qwen3_recipes.parse_recipe('Title: basil soup\nSteps:\n1. simmer water.', ['basil'], 9, True)
        self.assertEqual(recipe['ingredient_coverage'], 0.0)

    def test_parse_recipe_without_headings(self):
        recipe = qwen3_recipes.parse_recipe('Tomato salad\n1. slice the tomato.', ['tomato'], 8, False)
        self.assertEqual(recipe['title'], 'Tomato salad')
        self.assertEqual(recipe['steps'], ['slice the tomato.'])
        self.assertFalse(recipe['finished'])


if __name__ == '__main__':
    unittest.main()
