"""Tests for recipe SFT formatting, splitting, and padding."""

import unittest

import qwen_finetune


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = '|'.join(message['role'] + ':' + message['content'] for message in messages)
        if add_generation_prompt:
            rendered += '|assistant:'
        return rendered

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {
            'input_ids': list(text.encode('utf-8')),
            'offset_mapping': [(index, index + 1) for index in range(len(text))],
        }


class QwenFinetuneTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'title': 'Soup', 'ingredients': ['carrot'], 'steps': ['Boil it.']},
            {'title': 'Soup', 'ingredients': ['onion'], 'steps': ['Cook it.']},
            {'title': 'Pasta', 'ingredients': ['pasta'], 'steps': ['Boil pasta.']},
            {'title': 'Salad', 'ingredients': ['lettuce'], 'steps': ['Chop it.']},
            {'title': 'Toast', 'ingredients': ['bread'], 'steps': ['Toast it.']},
            {'title': 'Rice', 'ingredients': ['rice'], 'steps': ['Cook rice.']},
            {'title': 'Curry', 'ingredients': ['beans'], 'steps': ['Simmer it.']},
            {'title': 'Cake', 'ingredients': ['flour'], 'steps': ['Bake it.']},
            {'title': 'Stew', 'ingredients': ['potato'], 'steps': ['Simmer it.']},
            {'title': 'Omelet', 'ingredients': ['eggs'], 'steps': ['Fry it.']},
        ]

    def test_title_groups_are_not_split_across_train_and_validation(self):
        train, validation = qwen_finetune.split_recipe_rows(self.rows, seed=7)
        train_titles = {row['title'] for row in train}
        validation_titles = {row['title'] for row in validation}
        self.assertFalse(train_titles & validation_titles)
        self.assertEqual(len(train) + len(validation), len(self.rows))

    def test_supervised_dataset_masks_prompt_and_collates_padding(self):
        dataset = qwen_finetune.RecipeSFTDataset(
            self.rows[:2], FakeTokenizer(), max_length=2048,
        )
        self.assertEqual(len(dataset), 2)
        tokens, labels = dataset[0]
        self.assertIn(-100, labels)
        self.assertTrue(any(label != -100 for label in labels))
        batch = qwen_finetune.collate_batch(
            [dataset[0], dataset[1]], pad_token_id=0,
        )
        self.assertEqual(tuple(batch['input_ids'].shape), (2, max(len(dataset[0][0]), len(dataset[1][0]))))
        self.assertEqual(batch['attention_mask'].sum().item(), sum(len(dataset[i][0]) for i in range(2)))
        self.assertTrue((batch['labels'][batch['attention_mask'] == 0] == -100).all())


if __name__ == '__main__':
    unittest.main()
