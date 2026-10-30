"""LoRA fine-tuning for Qwen using the cleaned local recipe dataset."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'processed' / 'recipes_clean.jsonl'
DEFAULT_OUTPUT = ROOT / 'models' / 'qwen-recipe-lora'
DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
SYSTEM_PROMPT = (
    'You are a careful home-cooking assistant. Give practical, safe, concise '
    'recipe instructions. Use the requested ingredients and do not invent '
    'unavailable ingredients beyond basic salt, pepper, oil, or water.'
)


def load_recipe_rows(path):
    rows = []
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid recipe JSON on line {line_number}') from error
            if row.get('title') and row.get('ingredients') and row.get('steps'):
                rows.append(row)
    if len(rows) < 2:
        raise ValueError(f'At least two valid recipes are required in {path}')
    return rows


def split_recipe_rows(rows, seed=42, validation_fraction=0.1):
    titles = sorted({row['title'] for row in rows})
    random.Random(seed).shuffle(titles)
    validation_count = max(1, int(len(titles) * validation_fraction))
    validation_titles = set(titles[:validation_count])
    train = [row for row in rows if row['title'] not in validation_titles]
    validation = [row for row in rows if row['title'] in validation_titles]
    if not train or not validation:
        raise ValueError('Recipe data must contain at least two distinct titles')
    return train, validation


def recipe_messages(row):
    ingredients = ', '.join(str(item) for item in row['ingredients'])
    steps = '\n'.join(f'{index}. {step}' for index, step in enumerate(row['steps'], 1))
    answer = f"Title: {row['title']}\nSteps:\n{steps}"
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': (
                f'Create one recipe using these ingredients: {ingredients}.\n'
                'Reply with a title followed by numbered cooking steps.'
            ),
        },
    ], answer


class RecipeSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.samples = []
        skipped = 0
        for row in rows:
            messages, answer = recipe_messages(row)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                messages + [{'role': 'assistant', 'content': answer}],
                tokenize=False, add_generation_prompt=False,
            )
            if not full_text.startswith(prompt):
                raise ValueError('Tokenizer chat template changed between prompt and answer')
            prompt_end = len(prompt)
            encoded = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            full = encoded['input_ids']
            offsets = encoded['offset_mapping']
            if len(full) != len(offsets):
                raise ValueError('Tokenizer returned mismatched token and offset counts')
            response_start = next(
                (index for index, (start, end) in enumerate(offsets)
                 if end > prompt_end and start >= prompt_end),
                None,
            )
            if response_start is None:
                skipped += 1
                continue
            if response_start >= max_length:
                skipped += 1
                continue
            full = full[:max_length]
            labels = [-100] * min(response_start, len(full)) + full[response_start:]
            if not any(token != -100 for token in labels):
                skipped += 1
                continue
            self.samples.append((full, labels))
        if not self.samples:
            raise ValueError('No recipes fit max_length; increase --max-length')
        self.skipped = skipped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_batch(samples, pad_token_id):
    length = max(len(input_ids) for input_ids, _ in samples)
    input_ids = torch.full((len(samples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, (tokens, targets) in enumerate(samples):
        size = len(tokens)
        input_ids[index, :size] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[index, :size] = 1
        labels[index, :size] = torch.tensor(targets, dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_total, batch_count = 0.0, 0
    for batch in loader:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        loss_total += model(**batch).loss.item()
        batch_count += 1
    model.train()
    return loss_total / max(1, batch_count)


def fine_tune(args):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_recipe_rows(args.data)
    train_rows, validation_rows = split_recipe_rows(rows, args.seed)
    if args.max_train_recipes:
        train_rows = train_rows[:args.max_train_recipes]
    if args.max_validation_recipes:
        validation_rows = validation_rows[:args.max_validation_recipes]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    training = RecipeSFTDataset(train_rows, tokenizer, args.max_length)
    validation = RecipeSFTDataset(validation_rows, tokenizer, args.max_length)
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda samples: collate_batch(samples, pad_id),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda samples: collate_batch(samples, pad_id),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
    ))
    model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'model_id': args.model_id, 'data': str(args.data), 'seed': args.seed}
    (args.output_dir / 'minichef_adapter.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8',
    )
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for step, batch in enumerate(train_loader, 1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch + 1}, step {step}')
            group_size = min(
                args.gradient_accumulation,
                len(train_loader) - ((step - 1) % args.gradient_accumulation),
            )
            (loss / group_size).backward()
            accumulated += 1
            if accumulated == args.gradient_accumulation or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            if step % args.log_every == 0:
                print(f'Epoch {epoch + 1} | batch {step}/{len(train_loader)} | loss {loss.item():.4f}')
        validation_loss = evaluate(model, validation_loader, device)
        print(f'Epoch {epoch + 1} | validation loss {validation_loss:.4f}')
        if validation_loss < best_loss:
            best_loss = validation_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / 'metrics.json').write_text(
                json.dumps({
                    'epoch': epoch + 1,
                    'validation_loss': validation_loss,
                    'train_recipes': len(training),
                    'validation_recipes': len(validation),
                    'skipped_train_recipes': training.skipped,
                    'skipped_validation_recipes': validation.skipped,
                }, indent=2),
                encoding='utf-8',
            )
    print(f'Best validation loss: {best_loss:.4f}')
    print(f'LoRA adapter saved to {args.output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--model-id', default=DEFAULT_MODEL)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--max-train-recipes', type=int)
    parser.add_argument('--max-validation-recipes', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 8 or args.batch_size < 1:
        parser.error('epochs, batch-size, and max-length must be positive')
    if args.gradient_accumulation < 1 or args.log_every < 1:
        parser.error('gradient-accumulation and log-every must be positive')
    fine_tune(args)


if __name__ == '__main__':
    main()
