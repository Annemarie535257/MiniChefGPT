# MiniChefGPT

MiniChefGPT learns to turn ingredient names into a short recipe title and instruction steps. The main notebook shows the complete model and training workflow with simple section titles.

**Current status:** the refactor, training, and saved-model tests are complete. Recipe quality remains experimental: this CPU run is not yet a reliably high-performing recipe generator.

## Try the deployed model

Test the deployed MiniChefGPT model in Streamlit: [Open the MiniChefGPT app](https://minichefgpt.streamlit.app/)

Open **`MiniChefGPT_combined.ipynb`** and run its cells in order from the project root. It contains the actual training logs, held-out results, loss plot, and custom ingredient outputs from the latest run.

## Five Transformer mechanisms

1. **Word and position embeddings**: learned subword embeddings plus learned position embeddings.
2. **Layer normalization**: normalize each token before attention and before the feed-forward network.
3. **Causal attention**: four attention heads; each token can attend only to itself and earlier tokens.
4. **Feed-forward network**: Linear → GELU → Dropout → Linear for each token.
5. **Residual connections**: two skip connections per block, one around attention and one around the feed-forward network.

The CPU model has three Transformer blocks, width 128, feed-forward width 384, a 256-token context, a 4,096-piece byte-level vocabulary, and 1,053,440 trainable parameters. Input and output token weights are shared. The attention implementation uses PyTorch's scaled-dot-product kernel; its causal behavior is tested.

## Setup

Use Python 3.11 or newer with the training dependencies:

```powershell
python -m pip install -r requirements.txt
```

The development run uses project-local libraries in `.runtime/` when present. A normal installation can use a virtual environment and does not need that directory.

## Data and training

The pipeline reads `recipe_10000.csv` directly using `Name`, `RecipeIngredientParts`, and `RecipeInstructions`. It normalizes Unicode and HTML, removes empty rows and duplicate instructions, and keeps 9,979 recipes. Ingredient quantities do not consistently align with ingredient names in the source data, so they are not blindly paired.

Recipes are split before tokenizer fitting: 8,983 training, 502 validation, and 494 test recipes. Identical titles stay in the same split. The byte-level subword tokenizer is trained only on training recipes and can represent unseen ingredient names.

Each example has this structure:

```text
<bos><ingredients> chicken, garlic, onion, tomato
<title> chicken dinner
<step> first instruction
<step> next instruction
<eos>
```

Only title and instruction targets contribute to cross-entropy. Ingredient prompt tokens and right padding are ignored. Long recipes are divided into windows that repeat the ingredient prompt and retain recent response context. Every response token, including a single final end marker, is scored once per recipe view; later instructions are retained. Training includes a second shuffled ingredient order.

Training uses AdamW, learning-rate warmup and cosine decay, gradient clipping, and validation-based checkpoint selection. Validation and test metrics use ordinary, unweighted response-token cross-entropy over the entire split. Perplexity is `exp(loss)`. The test set is evaluated after selecting the best validation checkpoint. A separate free-generation check samples 12 unseen test ingredient lists with at most 12 ingredients and uses one candidate per input; it reports ingredient mentions, end markers, and nonempty steps. These are limited diagnostic metrics rather than proof of recipe correctness.

The default notebook runs training for about 30 minutes on CPU, plus validation, checkpoint writing, and generation. Set `TRAIN_MODEL = False` in the notebook to reuse the saved model. Set `RESUME_TRAINING = True` to continue from the latest checkpoint; increase `max_steps` when necessary.

The same implementation is available as a script:

```powershell
python minichefgpt.py --minutes 30 --steps 12000
python minichefgpt.py --resume --minutes 120 --steps 30000
```

Each resumed invocation gets a new time budget. The latest checkpoint stores optimizer and random states. Resuming requires the same data, tokenizer, and architecture. Keep the checkpoint and source together.

A larger profile is available for a CUDA GPU, with width 256, four blocks, context 384, and batch size 32:

```powershell
python minichefgpt.py --gpu-profile --minutes 0 --steps 30000
```

`--minutes 0` disables the time limit. The GPU profile saves separately under `models/gpu/`. `--output-dir` chooses another directory. Larger models and longer runs are options to evaluate; they do not guarantee better recipes.

## Streamlit interface

The deployment entry point is `streamlit_app.py`; it starts the interface defined in `app.py`. The interface loads `models/refactored/minichefgpt_best.pt`, accepts ingredients separated by commas or new lines, displays the generated title and steps, and downloads the recipe as text. Example buttons fill in chicken or pasta ingredients. The app caches the CPU model and serializes inference across sessions. It does not train or load the recipe dataset.

Run locally after installing `requirements.txt`:

```powershell
python -m streamlit run streamlit_app.py
```

For [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), push these files to your GitHub repository and select `streamlit_app.py` as the entry point:

- `streamlit_app.py`
- `app.py`
- `minichefgpt.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `models/refactored/minichefgpt_best.pt`

Use Python 3.11 to match the tested environment. The best checkpoint is about 4.2 MiB and is allowed through `.gitignore`; include it in the repository before deployment. The latest training checkpoint, raw dataset, notebook, local environments, and result JSON files are not needed for inference deployment. No API key is required. This prepares the app for deployment; it does not publish the app.

## Use your ingredients

Load the saved model without retraining:

```python
from minichefgpt import load_model, generate_recipe, print_recipe

model, tokenizer, config = load_model("models/refactored/minichefgpt_best.pt")
recipe = generate_recipe(
    model,
    tokenizer,
    ["chicken", "garlic", "onion", "tomato"],
    seed=123,
)
print_recipe(recipe)
```

A comma-separated ingredient string also works. The result is a dictionary containing `title`, `ingredients`, `steps`, `ingredient_coverage`, `mentioned_ingredients`, `generated_tokens`, and `finished` (whether the model generated `<eos>`).

Decoding uses temperature, top-k/top-p sampling, a sign-aware repetition penalty, and eight candidates ranked by ingredient mentions. The complete ingredient prompt is retained during long generation. All returned instruction text comes from the neural model; no retrieved recipe or recipe template replaces it.

## Saved files

- `MiniChefGPT_combined.ipynb`: self-contained notebook, with recorded outputs.
- `minichefgpt.py`: reusable model, data, training, and generation code.
- `build_notebook.py`: builds and executes the notebook from the same source.
- `models/refactored/minichefgpt_best.pt`: best validation checkpoint, including weights, tokenizer, configuration, and split manifest.
- `models/refactored/minichefgpt_last.pt`: resumable training state.
- `models/refactored/tokenizer.json`: standalone tokenizer.
- `models/refactored/metrics.json`: measured validation/test results and run settings.
- `models/refactored/training_history.json`: training and validation history.
- `models/refactored/custom_recipes.json`: actual custom-ingredient neural drafts.
- `models/refactored/heldout_generation.json`: neural outputs for 12 unseen test ingredient lists, sampled with a fixed seed and one candidate per input.
- `models/refactored/training_loss.png`: training plot.

The notebook defines all five mechanisms directly and uses the raw recipe data. Superseded models, component notebooks, and processed dataset copies have been removed.

## Results

The CPU run trained for approximately 30.0 minutes and completed 5,933 steps. The best validation checkpoint was step 5,933.

| Metric | Result |
| --- | ---: |
| Trainable parameters | 1,053,440 |
| Validation loss | 2.7660 |
| Validation perplexity | 15.90 |
| Test loss | 2.8143 |
| Test perplexity | 16.68 |
| Custom ingredient mentions, mean over 4 inputs, 8 candidates each | 56.2% |
| Unseen ingredient mentions, mean over 12 test inputs, 1 candidate each | 27.5% |
| End-marker rate on 12 unseen inputs | 100% |

| Custom input | Requested ingredients mentioned in steps | End marker |
| --- | ---: | --- |
| Chicken dinner | 25% | Yes |
| Vegetable pasta | 100% | Yes |
| Cheese omelet | 75% | Yes |
| Roasted vegetables | 25% | Yes |

The model learned recipe phrasing and recipe boundaries, but these drafts still omit ingredients, add unrelated ones, repeat phrases, and sometimes describe invalid actions. Ingredient overlap does not establish culinary correctness. **High-performing recipe generation has not been achieved by this run.** The notebook includes the actual outputs. Candidate sampling improves the four custom examples; the separate one-candidate benchmark exposes the weaker underlying ingredient adherence.

All 11 model checks and 6 Streamlit interface checks pass. The checkpoint was reloaded from disk before evaluating all 494 test recipes (64,382 response tokens).

## Quality limits

A small Transformer trained from scratch is an educational model, and lower perplexity alone does not establish useful recipe generation. Inspect the custom drafts for coherent steps, appropriate ingredients, and sensible quantities. Source recipes sometimes mention ingredients missing from their own ingredient list. Ingredient coverage checks simple word forms and measures mentions, not ingredient exclusivity or correctness.

The model may introduce unrelated ingredients, omit requested ones, end early, or produce incomplete instructions. The saved custom outputs expose those failures. A high-performing recipe assistant needs stronger data, more compute, and recipe-level evaluation beyond token loss; this run should be judged by its actual results.

## Checks

```powershell
python -m unittest discover -s tests -v
```

Checks cover future-token isolation, prompt/padding loss masking, complete response coverage across windows, split isolation, full held-out token accounting, unseen Unicode tokenization, negative-logit repetition penalties, and checkpoint/tokenizer round trips.

The Streamlit checks cover ingredient validation, example selection, recipe display, downloads, session persistence, missing checkpoints, and generation errors. A separate smoke check also generated a recipe through the interface using the real saved checkpoint.
