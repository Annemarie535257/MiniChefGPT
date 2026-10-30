"""Tests for Qwen response parsing and local generation."""

import unittest

import torch
from torch import nn

import qwen_recipe


class QwenRecipeTests(unittest.TestCase):
    def test_parse_recipe_extracts_title_and_numbered_steps(self):
        recipe = qwen_recipe.parse_recipe(
            'Title: **Tomato Pasta**\nSteps:\n1. Boil the pasta.\n2. Add tomatoes.',
            ['pasta', 'tomato'],
            finished=True,
            generated_tokens=18,
        )
        self.assertEqual(recipe['title'], 'Tomato Pasta')
        self.assertEqual(recipe['steps'], ['Boil the pasta.', 'Add tomatoes.'])
        self.assertEqual(recipe['ingredients'], ['pasta', 'tomato'])
        self.assertTrue(recipe['finished'])
        self.assertEqual(recipe['generated_tokens'], 18)

    def test_generation_uses_chat_prompt_and_reports_end_marker(self):
        class FakeTokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.assert_generation_prompt = kwargs['add_generation_prompt']
                return 'encoded prompt'

            def __call__(self, prompt, return_tensors):
                self.prompt = prompt
                return {'input_ids': torch.tensor([[1, 2]])}

            def decode(self, tokens, skip_special_tokens):
                self.skip_special_tokens = skip_special_tokens
                return 'Title: Quick soup\nSteps:\n1. Simmer the vegetables.'

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(3, 2)

            def get_input_embeddings(self):
                return self.embedding

            def generate(self, input_ids, **kwargs):
                self.generation_settings = kwargs
                return torch.tensor([[1, 2, 4, 9]])

        tokenizer, model = FakeTokenizer(), FakeModel()
        recipe = qwen_recipe.generate_qwen_recipe(model, tokenizer, ['carrot', 'water'])
        self.assertEqual(tokenizer.prompt, 'encoded prompt')
        self.assertTrue(tokenizer.assert_generation_prompt)
        self.assertEqual(tokenizer.messages[1]['role'], 'user')
        self.assertEqual(recipe['title'], 'Quick soup')
        self.assertEqual(recipe['steps'], ['Simmer the vegetables.'])
        self.assertTrue(recipe['finished'])
        self.assertEqual(recipe['generated_tokens'], 2)
        self.assertEqual(model.generation_settings['max_new_tokens'], 320)


if __name__ == '__main__':
    unittest.main()
