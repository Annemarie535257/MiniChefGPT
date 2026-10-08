"""Build, execute, or finish the self-contained MiniChefGPT notebook."""

import argparse
import base64
import contextlib
import copy
import io
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.runtime'))
import nbformat as nbf


DESCRIPTIONS = {
    'Setup': 'Set a reproducible seed and choose settings for this computer. The CPU profile has three Transformer blocks and a 256-token context. A larger GPU profile is available in the Python training script.',
    'Recipe data': 'Read ingredient names, titles, and instruction steps. Remove empty and duplicate recipes, normalize text, and split recipes into training, validation, and test sets. Recipes with the same title stay together. Ingredient quantities do not align reliably with ingredient names in this source, so they are not paired automatically.',
    'Subword tokenizer': 'Fit a byte-level subword tokenizer on training recipes only. Uncommon ingredient names can be represented without unknown whole words. Long recipes use multiple windows that repeat the ingredients; every response token is retained. Training also uses shuffled ingredient lists.',
    '1. Word and position embeddings': 'Mechanism 1: add a learned token embedding to a learned position embedding. Tokens are subword pieces rather than whole words. Both embeddings are updated during training.',
    '2. Layer normalization': 'Mechanism 2: normalize each token across its feature dimension before attention and before the feed-forward network.',
    '3. Causal attention': 'Mechanism 3: split queries, keys, and values into attention heads. Attention computes softmax(QKᵀ / √d) V with a causal mask, so a position cannot see future tokens. The PyTorch kernel implements this efficiently.',
    '4. Feed-forward network': 'Mechanism 4: apply Linear → GELU → Dropout → Linear independently to each token. This changes token features without mixing sequence positions.',
    '5. Residual connections': 'Mechanism 5: add each sublayer update to its input. Each block has two skip paths: x = x + Dropout(Attention(LayerNorm(x))), then x = x + Dropout(FFN(LayerNorm(x))).',
    'Recipe Transformer': 'Combine the five mechanisms into a causal language model. The output projection shares the token embedding weights. Training predicts the next title or instruction token; ingredient prompt and padding positions do not contribute to the loss.',
    'Training and evaluation': 'Use AdamW, learning-rate warmup and decay, gradient clipping, and validation-based checkpoint selection. Validation and test loss are ordinary, unweighted cross-entropy over every response token. Perplexity is exp(loss). The latest checkpoint includes optimizer and random states for resuming.',
    'Generate a recipe': 'Generate a short title followed by steps. Temperature, top-k/top-p sampling, and a sign-aware repetition penalty control decoding. Several candidates are ranked by requested-ingredient mentions. This score does not establish correctness or ingredient exclusivity.',
    'Prepare training': 'Prepare the datasets and define four custom ingredient tests. Reusing a checkpoint also reuses its tokenizer, configuration, and data split. The final test set is used after checkpoint selection. Save metrics and actual neural outputs, including whether generation reached the end marker.',
}


def code_cell(source, role, section=None):
    cell = nbf.v4.new_code_cell(source.strip())
    cell.metadata['minichefgpt'] = {'role': role}
    if section is not None:
        cell.metadata['minichefgpt']['section'] = section
    return cell


def build(minutes=30, steps=12000, resume=False):
    """Build notebook source without running training or writing any files."""
    source = (ROOT / 'minichefgpt.py').read_text(encoding='utf-8')
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    }
    cells = [nbf.v4.new_markdown_cell(
        '# MiniChefGPT\n\n'
        'Train a Transformer to turn ingredient lists into a title and recipe steps. '
        'This notebook explains five mechanisms, saves the best model, reloads it, '
        'and tests custom ingredients.\n\n'
        'Run from the project root with `requirements.txt`. '
        'Use the measured results and generated recipes below to assess quality.'
    )]
    for chunk in source.split('# %% ')[1:]:
        heading, code = chunk.split('\n', 1)
        heading = heading.strip()
        if heading == 'Prepare training':
            code = code.split('\ndef main():', 1)[0]
        cells.extend([
            nbf.v4.new_markdown_cell('## ' + heading + '\n\n' + DESCRIPTIONS[heading]),
            code_cell(code, 'definition', heading),
        ])
        if heading == 'Setup':
            cells.append(code_cell(
                f'config = Config(training_minutes={minutes!r}, max_steps={steps!r})\n'
                'TRAIN_MODEL = True\n'
                f'RESUME_TRAINING = {resume!r}\n'
                'OUTPUT_DIR = PROJECT_ROOT / "models/refactored"\n'
                'print(config)',
                'settings',
            ))
    cells.extend([
        nbf.v4.new_markdown_cell(
            '## Prepare the model\n\n'
            'Build the train-only tokenizer, datasets, and Transformer. When reusing a '
            'saved model, load its exact tokenizer and settings before building datasets. '
            'The saved data manifest verifies that the held-out split is unchanged.'
        ),
        code_cell('''
if TRAIN_MODEL:
    device, splits, tokenizer, datasets, manifest, model = prepare(config)
else:
    device, splits, tokenizer, datasets, manifest, model = prepare_saved_model(
        OUTPUT_DIR / "minichefgpt_best.pt")
    config = model.config
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Response tokens discarded:", manifest["truncated_response_tokens"])
''', 'prepare'),
        nbf.v4.new_markdown_cell(
            '## Train and save\n\n'
            'Train within the time budget above. The best validation checkpoint is '
            '`models/refactored/minichefgpt_best.pt`; the resumable checkpoint is '
            '`minichefgpt_last.pt`. Set `TRAIN_MODEL = False` to reuse the saved model. '
            'Set `RESUME_TRAINING = True` and increase `max_steps` to continue training.'
        ),
        code_cell('''
if TRAIN_MODEL:
    history, run = train_model(model, datasets, tokenizer, config, device,
                               OUTPUT_DIR, manifest, resume=RESUME_TRAINING)
else:
    history = json.loads((OUTPUT_DIR / "training_history.json").read_text(encoding="utf-8"))
    run = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))["run"]
print("Best checkpoint:", OUTPUT_DIR / "minichefgpt_best.pt")
''', 'train'),
        nbf.v4.new_markdown_cell(
            '## Test the saved model\n\n'
            'Reload the checkpoint from disk, evaluate all held-out test response tokens, '
            'and test chicken dinner, pasta, an omelet, and vegetables. Also generate recipes for 12 unseen test ingredient lists. These are actual '
            'neural outputs; no retrieved recipe or template is substituted.'
        ),
        code_cell('''
metrics, custom_recipes = finish_run(model, tokenizer, config, device, datasets,
                                   manifest, history, run, OUTPUT_DIR, test_rows=splits['test'])
''', 'test'),
        nbf.v4.new_markdown_cell(
            '## Training results\n\n'
            'Compare training and validation loss at each evaluation checkpoint. '
            'Test perplexity measures token prediction. Inspect custom recipes to see '
            'whether the model follows the requested ingredients.'
        ),
        code_cell('''
import matplotlib.pyplot as plt
summary = pd.DataFrame([{"split": name, **metrics[name]} for name in ("validation", "test")])
print(summary.to_string(index=False))
print("Best training step:", metrics["run"]["best_step"])
print("Held-out generation:", metrics["generation_benchmark"])
print("Custom ingredient coverage:", "{:.0%}".format(metrics["mean_custom_ingredient_coverage"]))
plot_data = pd.DataFrame(history)
plt.figure(figsize=(8, 4))
plt.plot(plot_data["step"], plot_data["train_loss"], label="Training")
plt.plot(plot_data["step"], plot_data["validation_loss"], label="Validation")
plt.xlabel("Training step")
plt.ylabel("Response-token cross-entropy")
plt.title("MiniChefGPT training")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
plt.show()
plt.close()
''', 'plot'),
        nbf.v4.new_markdown_cell(
            '## Your ingredients\n\n'
            'Edit this list and rerun the cell. Loading a saved model does not require '
            'retraining. The result includes a title, steps, ingredient mentions, '
            'and whether the recipe reached its end marker.'
        ),
        code_cell('''
recipe_model, recipe_tokenizer, saved_config = load_model(OUTPUT_DIR / "minichefgpt_best.pt", device)
my_ingredients = ["chicken", "garlic", "onion", "tomato"]
my_recipe = generate_recipe(recipe_model, recipe_tokenizer, my_ingredients, seed=123)
print_recipe(my_recipe)
''', 'custom'),
        nbf.v4.new_markdown_cell(
            '## What the results mean\n\n'
            'A small model trained from scratch can learn recipe language without '
            'consistently planning a usable meal. Ingredient mentions do not prove '
            'that all requested ingredients are used or that extra ingredients are absent. '
            'The source data sometimes mentions ingredients missing from its own list. '
            'Assess quality using the saved outputs as well as held-out loss. '
            'Longer training and a larger GPU model are supported; measure improvements '
            'on validation data and new ingredient tests. Earlier weighted-loss results '
            'are not directly comparable to this unweighted, response-only evaluation.'
        ),
    ])
    notebook.cells = cells
    validate_source(notebook)
    return notebook


def validate_source(notebook):
    """Check the notebook schema and compile every code cell without executing it."""
    nbf.validate(notebook)
    for number, cell in enumerate(notebook.cells, 1):
        if cell.cell_type == 'code':
            compile(cell.source, f'MiniChefGPT_combined.ipynb:cell{number}', 'exec')


def write_notebook(notebook, path):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        nbf.write(notebook, stream)
    temporary.replace(path)


class Tee(io.TextIOBase):
    """Capture real cell output and save long-running progress about every 30 seconds."""
    def __init__(self, terminal, notebook, cell, path, name='stdout'):
        self.terminal, self.notebook, self.cell, self.path = terminal, notebook, cell, path
        self.name = name
        self.last_save = time.monotonic()

    def write(self, text):
        self.terminal.write(text)
        self.terminal.flush()
        if not text:
            return 0
        if (self.cell.outputs and self.cell.outputs[-1].output_type == 'stream'
                and self.cell.outputs[-1].name == self.name):
            self.cell.outputs[-1].text += text
        else:
            self.cell.outputs.append(nbf.v4.new_output('stream', name=self.name, text=text))
        if time.monotonic() - self.last_save >= 30:
            write_notebook(self.notebook, self.path)
            self.last_save = time.monotonic()
        return len(text)

    def flush(self):
        self.terminal.flush()


def training_output(notebook):
    """Find the real training cell in both the original and rebuilt notebook layouts."""
    heading = None
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## '):
            heading = cell.source.splitlines()[0][3:].strip()
        elif cell.cell_type == 'code':
            role = cell.metadata.get('minichefgpt', {}).get('role')
            if role == 'train' or heading == 'Train and save':
                return copy.deepcopy(cell.outputs)
    raise ValueError('The existing notebook has no training cell to preserve')


def execute(notebook, path, *, saved_training_outputs=None, saved_run=None):
    """Execute the notebook, or finish saved training without calling train_model."""
    validate_source(notebook)
    scope = {'__name__': '__notebook__', '__file__': str(ROOT / 'minichefgpt.py')}
    count = 0
    finishing = saved_training_outputs is not None
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        cell.execution_count = count
        cell.outputs = []
        role = cell.metadata.get('minichefgpt', {}).get('role')
        if finishing and role == 'train':
            scope['history'] = json.loads(
                (scope['OUTPUT_DIR'] / 'training_history.json').read_text(encoding='utf-8'))
            scope['run'] = copy.deepcopy(saved_run)
            cell.outputs = copy.deepcopy(saved_training_outputs)
            cell.metadata['minichefgpt']['preserved_training_output'] = True
            write_notebook(notebook, path)
            continue
        stdout = Tee(sys.stdout, notebook, cell, path)
        stderr = Tee(sys.stderr, notebook, cell, path, 'stderr')
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(cell.source, f'{path.name}:cell{count}', 'exec'), scope)
            if finishing and role == 'settings':
                # The original settings stay visible, while finalization only loads saved weights.
                scope['TRAIN_MODEL'] = False
            if role == 'plot':
                image_path = scope['OUTPUT_DIR'] / 'training_loss.png'
                cell.outputs.append(nbf.v4.new_output('display_data', data={
                    'image/png': base64.b64encode(image_path.read_bytes()).decode('ascii'),
                }, metadata={}))
        except Exception as error:
            cell.outputs.append(nbf.v4.new_output(
                'error', ename=type(error).__name__, evalue=str(error),
                traceback=traceback.format_exc().splitlines(),
            ))
            write_notebook(notebook, path)
            raise
        write_notebook(notebook, path)
    nbf.validate(notebook)
    print('Executed and saved:', path)
    return notebook


def finalize(path):
    """Repair source, preserve completed training output, and rerun saved-model results."""
    with path.open('r', encoding='utf-8') as stream:
        previous = nbf.read(stream, as_version=4)
    outputs = training_output(previous)
    if not outputs or any(output.output_type == 'error' for output in outputs):
        raise ValueError('Complete training successfully before finalizing its notebook')
    metrics_path = ROOT / 'models/refactored/metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if not (ROOT / 'models/refactored/minichefgpt_best.pt').is_file():
        raise FileNotFoundError('The saved best model is required for finalization')
    notebook = build(metrics['config']['training_minutes'], metrics['config']['max_steps'])
    # Preserve the exact configuration that produced the recorded training outputs.
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.metadata.get('minichefgpt', {}).get('role') == 'settings':
            cell.source = (
                'config = Config(**' + repr(metrics['config']) + ')\n'
                'TRAIN_MODEL = True\n'
                'RESUME_TRAINING = False\n'
                'OUTPUT_DIR = PROJECT_ROOT / "models/refactored"\n'
                'print(config)'
            )
    return execute(notebook, path, saved_training_outputs=outputs, saved_run=metrics['run'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=30)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--resume', action='store_true')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--build-only', action='store_true')
    action.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'MiniChefGPT_combined.ipynb'
    if args.finalize or not args.build_only:
        import matplotlib
        matplotlib.use('Agg')
    if args.finalize:
        finalize(path)
    else:
        notebook = build(args.minutes, args.steps, args.resume)
        write_notebook(notebook, path)
        if not args.build_only:
            execute(notebook, path)


if __name__ == '__main__':
    main()
