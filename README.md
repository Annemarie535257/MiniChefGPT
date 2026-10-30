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

For a larger GPT-2-inspired model trained from random initialization, use the compact six-layer profile:

```powershell
python minichefgpt.py --gpt2-profile --minutes 30 --steps 12000
```

This profile uses 256-wide embeddings, 8 attention heads, a 4x feed-forward layer, and a 512-token context. It is about 6 million parameters, not the 124-million-parameter GPT-2 Small, and it does not load pretrained GPT-2 weights. It saves separately under `models/gpt2/`; add `--minutes 0` to disable the time limit. A larger architecture trained from scratch may still perform worse on this relatively small recipe dataset, so compare held-out loss and inspect generated recipes before replacing the deployed checkpoint.

## Streamlit interface

The deployment entry point is `streamlit_app.py`; it starts the interface defined in `app.py`. The app offers MiniChefGPT, fine-tuned GPT-2, Qwen2.5-1.5B-Instruct, and a Qwen LoRA fine-tune. It accepts ingredients separated by commas or new lines, displays the generated title and steps, and downloads the recipe as text. Models are loaded only when selected for generation, cached, and protected by a lock for inference across sessions.

The pretrained Qwen option downloads model files from Hugging Face on first use (several GB of storage). CPU inference requires substantial RAM and can be slow; CUDA is used automatically when available. The fine-tuned Qwen option uses the adapter produced by the workflow below and loads its recorded base model. No API key is required.

### Fine-tune Qwen on the recipe data

The project includes a LoRA supervised fine-tuning script. It uses the cleaned local recipe JSONL, keeps matching recipe titles together in a deterministic train/validation split, and trains only small adapter weights. The default fine-tuning base is Qwen2.5-0.5B-Instruct to reduce memory use; the saved adapter records its base model so the app loads the matching model. Install dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python qwen_finetune.py
```

The first run downloads the selected base model. By default, training runs one epoch and saves the best validation adapter under `models/qwen-recipe-lora/`. After training, select **Qwen recipe fine-tune (local)** in the app. For a short smoke run, add `--max-train-recipes 100 --max-validation-recipes 20`. Use `--model-id Qwen/Qwen2.5-1.5B-Instruct` only on a machine with enough memory/compute. Fine-tuning on CPU can take a long time; a supported CUDA GPU is recommended.

Run locally after installing `requirements.txt`:

```powershell
python -m streamlit run streamlit_app.py
```

For [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), push these files to your GitHub repository and select `streamlit_app.py` as the entry point:

- `streamlit_app.py`
- `app.py`
- `qwen_recipe.py`
- `minichefgpt.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `models/refactored/minichefgpt_best.pt` (only needed for the MiniChefGPT baseline)

Use Python 3.11 to match the tested environment. The best checkpoint is about 4.2 MiB and is allowed through `.gitignore`; include it in the repository before deployment. The latest training checkpoint, raw dataset, notebook, local environments, and result JSON files are not needed for inference deployment. No API key is required. This prepares the app for deployment; it does not publish the app.

The fine-tuned GPT-2 folder (`models/gpt2/best/`, about 480 MB) is excluded by `.gitignore`, so a deployed app shows the GPT-2 option as unavailable until the model is provided. To enable it, upload that folder to a Hugging Face model repository and set the `MINICHEF_GPT2_MODEL` environment variable (or Streamlit secret exposed as an environment variable) to its model id. GPT-2 needs roughly 1 GB of memory alongside the small model.

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
- `qwen_recipe.py`: local Qwen loading and recipe generation.
- `qwen_finetune.py`: LoRA fine-tuning on the cleaned recipe dataset.
- `build_notebook.py`: builds and executes the notebook from the same source.
- `models/refactored/minichefgpt_best.pt`: best validation checkpoint, including weights, tokenizer, configuration, and split manifest.
- `models/refactored/minichefgpt_last.pt`: resumable training state.
- `models/refactored/tokenizer.json`: standalone tokenizer.
- `models/refactored/metrics.json`: measured validation/test results and run settings.
- `models/refactored/training_history.json`: training and validation history.
- `models/refactored/custom_recipes.json`: actual custom-ingredient neural drafts.
- `models/refactored/heldout_generation.json`: neural outputs for 12 unseen test ingredient lists, sampled with a fixed seed and one candidate per input.
- `models/refactored/training_loss.png`: training plot.
- `MiniChefGPT_GPT2_finetune.ipynb`: GPT-2 fine-tuning notebook, with recorded outputs and the comparison with MiniChefGPT.
- `gpt2_recipes.py`: loads the fine-tuned GPT-2 model and generates recipes.
- `models/gpt2/`: fine-tuning history, metrics, and generated recipes; `best/` holds the model weights (not committed).

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

All 11 model checks, 3 GPT-2 parsing checks, and 8 Streamlit interface checks pass. The checkpoint was reloaded from disk before evaluating all 494 test recipes (64,382 response tokens).

## GPT-2 fine-tuning

`MiniChefGPT_GPT2_finetune.ipynb` fine-tunes pretrained GPT-2 (124M parameters) on the same cleaned recipes and the same train/validation/test split. Recipes are written as plain text (`Ingredients: … / Title: … / Steps: 1. …`) ending with GPT-2's `<|endoftext|>` token; only title and step tokens contribute to the loss. Training runs on CPU for a 60-minute budget, with the cosine learning-rate decay tied to elapsed time so the schedule completes. Sequences are capped at 256 tokens (1,271 training recipes are truncated). The best model is saved to `models/gpt2/best/`, and results to `models/gpt2/metrics.json`, `custom_recipes.json`, and `heldout_generation.json`. `gpt2_recipes.py` provides the same loading and generation code for the app.

The recorded run completed 292 steps (2,336 recipes, about a quarter of the training set) in 61 minutes.

| Metric | MiniChefGPT | GPT-2 fine-tuned |
| --- | ---: | ---: |
| Parameters | 1,053,440 | 124,439,808 |
| Unseen ingredient mentions, 12 test inputs, 1 candidate each | 27.5% | 56.1% |
| Custom ingredient mentions, 4 inputs, 8 candidates each | 56.2% | 100% |
| End-marker rate on 12 unseen inputs | 100% | 100% |
| Test loss (GPT-2 tokens): before → after fine-tuning | — | 2.484 → 2.018 |

The two models use different tokenizers, so their perplexities are not comparable; the generation rows use the same inputs, seeds, and `ingredient_coverage` score. GPT-2 follows the requested ingredients much more often and writes more fluent steps, but its drafts still repeat actions, contradict themselves (for example, "remove from heat and simmer"), and sometimes add ingredients. Longer fine-tuning, ideally on a GPU, is the clearest next step.

## Quality limits

A small Transformer trained from scratch is an educational model, and lower perplexity alone does not establish useful recipe generation. Inspect the custom drafts for coherent steps, appropriate ingredients, and sensible quantities. Source recipes sometimes mention ingredients missing from their own ingredient list. Ingredient coverage checks simple word forms and measures mentions, not ingredient exclusivity or correctness.

The model may introduce unrelated ingredients, omit requested ones, end early, or produce incomplete instructions. The saved custom outputs expose those failures. A high-performing recipe assistant needs stronger data, more compute, and recipe-level evaluation beyond token loss; this run should be judged by its actual results.

## Checks

```powershell
python -m unittest discover -s tests -v
```

Checks cover future-token isolation, prompt/padding loss masking, complete response coverage across windows, split isolation, full held-out token accounting, unseen Unicode tokenization, negative-logit repetition penalties, and checkpoint/tokenizer round trips.

The Streamlit checks cover ingredient validation, example selection, recipe display, downloads, session persistence, missing checkpoints, generation errors, choosing the GPT-2 model, and a missing GPT-2 model. A separate smoke check also generated a recipe through the interface using the real saved checkpoint.
