"""Focused checks for ingredient conditioning, causal learning, and saved models."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import minichefgpt as chef

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class MiniChefGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        chef.seed_everything(7)
        cls.short_recipe = {
            'id': 1,
            'title': 'carrot soup',
            'ingredients': ['carrot', 'onion', 'water'],
            'steps': ['chop the carrot and onion.', 'simmer in water until tender.'],
        }
        cls.long_recipe = {
            'id': 2,
            'title': 'vegetable dinner',
            'ingredients': ['carrot', 'onion', 'water'],
            'steps': [f'cook the carrot and onion for {i} minutes.' for i in range(80)],
        }
        cls.tokenizer = chef.train_tokenizer([cls.short_recipe, cls.long_recipe], 384)
        cls.config = chef.Config(
            context_length=64, d_model=16, num_heads=4, num_layers=2,
            d_ff=32, dropout=0.0, cpu_threads=2,
        )

    def new_model(self):
        return chef.MiniChefGPT(self.config, self.tokenizer.get_vocab_size())

    def test_attention_and_model_do_not_read_future_tokens(self):
        attention = chef.CausalAttention(16, 4, dropout=0.0).eval()
        hidden = torch.randn(2, 12, 16)
        changed = hidden.clone()
        changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 10
        with torch.no_grad():
            original = attention(hidden)
            perturbed = attention(changed)
        torch.testing.assert_close(original[:, :6], perturbed[:, :6])
        self.assertFalse(torch.allclose(original[:, 6:], perturbed[:, 6:]))

        model = self.new_model().eval()
        tokens = torch.randint(0, self.tokenizer.get_vocab_size(), (2, 12))
        changed_tokens = tokens.clone()
        changed_tokens[:, 6:] = (changed_tokens[:, 6:] + 1) % self.tokenizer.get_vocab_size()
        with torch.no_grad():
            original = model(tokens)
            perturbed = model(changed_tokens)
        torch.testing.assert_close(original[:, :6], perturbed[:, :6])

    def test_ingredient_prompt_and_padding_do_not_contribute_loss(self):
        dataset = chef.RecipeDataset([self.short_recipe], self.tokenizer, 64)
        prompt = self.tokenizer.encode(chef.recipe_prompt(self.short_recipe['ingredients'])).ids
        response = self.tokenizer.encode(chef.recipe_response(self.short_recipe)).ids
        x, labels = dataset.samples[0]
        self.assertEqual(labels[:len(prompt) - 1], [chef.IGNORE_INDEX] * (len(prompt) - 1))
        self.assertEqual(labels[len(prompt) - 1], response[0])
        self.assertEqual(x[:len(prompt)], prompt)

        # Add a shorter sample so a real batch contains ignored right padding.
        tiny = dict(self.short_recipe, title='soup', steps=['cook.'])
        dataset = chef.RecipeDataset([self.short_recipe, tiny], self.tokenizer, 64)
        x, y = dataset.batch(range(len(dataset)), self.tokenizer.token_to_id('<pad>'), 'cpu')
        self.assertTrue((y[-1] == chef.IGNORE_INDEX).any())
        model = self.new_model().eval()
        with torch.no_grad():
            loss_sum, count = model(x, y)
            logits = model(x)
            selected = y != chef.IGNORE_INDEX
            expected = F.cross_entropy(logits[selected], y[selected], reduction='sum')
        self.assertEqual(count.item(), int(selected.sum()))
        torch.testing.assert_close(loss_sum, expected)

    def test_long_windows_score_every_response_token_once(self):
        for row in (self.short_recipe, self.long_recipe):
            with self.subTest(title=row['title']):
                dataset = chef.RecipeDataset([row], self.tokenizer, 64)
                expected = self.tokenizer.encode(chef.recipe_response(row)).ids
                scored = [token for _, labels in dataset.samples
                          for token in labels if token != chef.IGNORE_INDEX]
                self.assertEqual(scored, expected)
                self.assertEqual(dataset.response_tokens, len(expected))
                self.assertEqual(scored.count(self.tokenizer.token_to_id('<eos>')), 1)
                for x, y in dataset.samples:
                    self.assertEqual(len(x), len(y))
                    self.assertLessEqual(len(x), 64)
                    self.assertTrue(any(token != chef.IGNORE_INDEX for token in y))
                if row is self.long_recipe:
                    self.assertGreater(len(dataset), 1)
                    self.assertEqual(dataset.long_recipes, 1)

    def test_tiny_context_windows_always_make_progress(self):
        class BoundedReads(list):
            """Fail promptly if a regression creates an endless slicing loop."""
            reads = 0

            def __getitem__(self, key):
                self.reads += 1
                if self.reads > 2 * len(self) + 2:
                    raise AssertionError('Response windows stopped making progress')
                return super().__getitem__(key)

        expected = list(range(8, 88)) + [6]

        class TinyTokenizer:
            def encode(self, text):
                if text.startswith('<bos>'):
                    return SimpleNamespace(ids=[2, 3, 7, 4])
                return SimpleNamespace(ids=BoundedReads(expected))

        dataset = chef.RecipeDataset([self.long_recipe], TinyTokenizer(), 8)
        scored = [token for _, labels in dataset.samples
                  for token in labels if token != chef.IGNORE_INDEX]
        self.assertEqual(scored, expected)
        self.assertGreater(len(dataset), 1)
        for tokens, labels in dataset.samples:
            self.assertLessEqual(len(tokens), 8)
            self.assertEqual(tokens[:4], [2, 3, 7, 4])
            self.assertTrue(any(token != chef.IGNORE_INDEX for token in labels))

    def test_split_keeps_titles_together_and_removes_duplicate_instructions(self):
        raw = []
        for title_index in range(40):
            for variant in range(2):
                raw.append({
                    'Name': f'dinner {title_index}',
                    'RecipeIngredientParts': repr(['carrot', 'water']),
                    'RecipeInstructions': repr([f'cook batch {title_index} variant {variant}.']),
                })
        raw.append(dict(raw[0], Name='same instructions under another title'))
        raw.append(dict(raw[0], Name='empty recipe', RecipeInstructions='[]'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(raw).to_csv(root / 'recipe_10000.csv', index=False)
            rows = chef.load_recipes(root)
        self.assertEqual(len(rows), 80)
        splits = chef.split_recipes(rows, seed=29)
        self.assertEqual(splits, chef.split_recipes(rows, seed=29))
        titles = {name: {row['title'] for row in values} for name, values in splits.items()}
        ids = {name: {row['id'] for row in values} for name, values in splits.items()}
        instructions = {name: {tuple(row['steps']) for row in values} for name, values in splits.items()}
        for first, second in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            self.assertFalse(titles[first] & titles[second])
            self.assertFalse(ids[first] & ids[second])
            self.assertFalse(instructions[first] & instructions[second])
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_evaluation_uses_all_response_tokens_and_restores_model_mode(self):
        dataset = chef.RecipeDataset([self.short_recipe, self.long_recipe], self.tokenizer, 64)
        model = self.new_model().eval()
        total_loss = 0.0
        with torch.no_grad():
            for index in range(len(dataset)):
                x, y = dataset.batch([index], self.tokenizer.token_to_id('<pad>'), 'cpu')
                loss, _ = model(x, y)
                total_loss += loss.item()
        model.train()
        metrics = chef.evaluate(model, dataset, 3, self.tokenizer.token_to_id('<pad>'), 'cpu')
        self.assertTrue(model.training)
        self.assertEqual(metrics['tokens'], dataset.response_tokens)
        self.assertAlmostEqual(metrics['loss'], total_loss / dataset.response_tokens, places=5)
        self.assertAlmostEqual(metrics['perplexity'], math.exp(metrics['loss']), places=5)

    def test_byte_tokenizer_supports_unseen_ingredient_characters(self):
        encoded = self.tokenizer.encode('dragonfruit jalapeño 김치').ids
        self.assertNotIn(self.tokenizer.token_to_id('<unk>'), encoded)

    def test_ingredient_coverage_handles_irregular_plurals(self):
        ingredients = ['potatoes', 'tomatoes', 'fresh basil leaves', 'berries', 'garlic']
        coverage, matched = chef.ingredient_coverage(
            ingredients, 'Dice a potato and tomato. Add a basil leaf and one berry.',
        )
        self.assertEqual(coverage, 0.8)
        self.assertEqual(matched, ingredients[:-1])
        coverage, matched = chef.ingredient_coverage(['eggs'], 'Bake the eggplant.')
        self.assertEqual(coverage, 0.0)
        self.assertEqual(matched, [])

    def test_negative_logit_repetition_penalty_discourages_repetition(self):
        class ToyTokenizer:
            vocabulary = chef.SPECIAL_TOKENS + ['title', 'carrot']

            def token_to_id(self, token):
                return self.vocabulary.index(token)

            def encode(self, text):
                return SimpleNamespace(ids=[2, 3, 8, 4])

            def decode(self, tokens):
                return ' '.join(self.vocabulary[token] for token in tokens)

        class NegativeLogitModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(context_length=64)

            def forward(self, tokens, last_only=False):
                logits = torch.full((1, 1, 9), -100.0)
                logits[0, 0, 7] = -1.0
                logits[0, 0, 8] = -1.1
                return logits

        model = NegativeLogitModel().train()
        recipe = chef.sample_recipe(
            model, ToyTokenizer(), ['carrot'], max_new_tokens=2,
            top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=2.0,
        )
        self.assertEqual(recipe['title'], 'title carrot')
        self.assertTrue(model.training)

    def test_checkpoint_round_trip_preserves_predictions_and_tokenizer(self):
        model = self.new_model().eval()
        tokens = torch.tensor([self.tokenizer.encode(chef.recipe_prompt(['carrot'])).ids])
        with torch.no_grad():
            expected = model(tokens)
        manifest = {'recipe_ids': {'train': [1], 'validation': [2], 'test': [3]}}
        payload = chef.model_payload(model, self.tokenizer, self.config, 5, 1.25, manifest)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'model.pt'
            chef.atomic_save(payload, checkpoint)
            self.assertFalse(checkpoint.with_suffix('.tmp').exists())
            loaded, tokenizer, config = chef.load_model(checkpoint)
            with torch.no_grad():
                actual = loaded(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tokenizer.to_str(), self.tokenizer.to_str())
        self.assertEqual(config, self.config)
        self.assertFalse(loaded.training)
        self.assertIs(loaded.head.weight, loaded.embedding.words.weight)

    def test_saved_model_preparation_reuses_tokenizer_and_rejects_data_drift(self):
        raw = [{
            'Name': f'carrot soup {index}',
            'RecipeIngredientParts': repr(['carrot', 'water']),
            'RecipeInstructions': repr([f'cook the carrot for {index + 1} minutes.']),
        } for index in range(40)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'recipe_10000.csv'
            pd.DataFrame(raw).to_csv(source, index=False)
            _, _, tokenizer, _, manifest, model = chef.prepare(
                self.config, root, tokenizer=self.tokenizer,
            )
            checkpoint = root / 'saved.pt'
            chef.atomic_save(
                chef.model_payload(model, tokenizer, self.config, 4, 2.0, manifest),
                checkpoint,
            )
            with mock.patch.object(chef, 'train_tokenizer', side_effect=AssertionError('Tokenizer retrained')):
                _, _, restored_tokenizer, _, restored_manifest, restored_model = chef.prepare_saved_model(
                    checkpoint, root,
                )
                self.assertEqual(restored_manifest, manifest)
                self.assertEqual(restored_tokenizer.to_str(), tokenizer.to_str())
                for name, parameter in model.state_dict().items():
                    torch.testing.assert_close(restored_model.state_dict()[name].cpu(), parameter.cpu())

                raw[0]['RecipeInstructions'] = repr(['simmer the onion until tender.'])
                pd.DataFrame(raw).to_csv(source, index=False)
                with self.assertRaisesRegex(ValueError, 'Dataset changed since training'):
                    chef.prepare_saved_model(checkpoint, root)


if __name__ == '__main__':
    unittest.main()
