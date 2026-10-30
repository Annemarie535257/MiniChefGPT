# MiniChefGPT: Fine-tuning GPT-2 and Qwen3 for Recipe Generation

Slide deck source. Each slide is separated by `---`. Speaker notes follow each slide.

---

## Slide 1 — From a small model to pretrained models

**Problem:** our original MiniChefGPT (a 1M-parameter Transformer trained from scratch) learned recipe *style*, but ignored the ingredients it was given.

**Idea:** start from models that already know English and food, then fine-tune them on the same recipes.

| | MiniChefGPT | GPT-2 | Qwen3-0.6B |
|---|---|---|---|
| Parameters | 1M | 124M | 600M |
| Starting point | Random | Pretrained | Pretrained chat model |
| How it was trained | Full training | Full fine-tuning | LoRA (10M trainable, 1.7%) |
| Training time (laptop CPU) | 30 min | 60 min | 60 min |
| Recipes seen in training | ~47,000 windows | 2,336 | 224 |

**Same data and same test for all three:** 9,979 Food.com recipes, the same train/validation/test split, the same 12 unseen test recipes and 4 custom ingredient lists.

> **Speaker notes:** Everything was trained on a laptop CPU with a one-hour budget, so GPT-2 and Qwen3 saw only a fraction of the training set (GPT-2 about a quarter, Qwen3 about 2.5%). Keeping the data split and test inputs identical is what makes the comparison fair.

---

## Slide 2 — Results: pretrained models follow the ingredients

**Main score:** the share of requested ingredients that the recipe steps actually use.

| Model | Unseen recipes (1 draft) | Custom inputs (best of 8) | Recipe finished properly |
|---|---:|---:|---:|
| MiniChefGPT | 27.5% | 56% | 100% |
| GPT-2 fine-tuned | 56.1% | 100% | 100% |
| **Qwen3 fine-tuned** | **62.6%** | **100%** | **100%** |

**Fine-tuning helped both models**
- GPT-2 test loss: 2.48 → **2.02** (60 minutes)
- Qwen3 validation loss: 2.73 → **1.93** after only 224 recipes

**Example — pasta, tomato, garlic, basil (Qwen3):**
*"tomato basil pasta"*: boil the water → cook the pasta until al dente → drain → toss with tomato, garlic and basil.

**Still not perfect:** repeated or contradictory steps (GPT-2: "remove from heat and simmer for 1 hour"), unrelated titles (Qwen3: "1997", "cream cheese frosting").

> **Speaker notes:** Qwen3 scored best on unseen recipes even though it trained on ten times fewer recipes than GPT-2. That shows how much a stronger pretrained model brings. Loss values cannot be compared between models because each uses a different tokenizer, so the shared ingredient score is the fair comparison. The score checks mentions, not whether a recipe is correct or safe.

---

## Slide 3 — Experiment: GPT-2 drafts, Qwen3 edits, RAG adds examples

**Pipeline tested**
1. **GPT-2** writes a draft recipe.
2. **Qwen3** edits the draft: removes repeated steps, fixes the order, keeps only the listed ingredients.
3. **RAG (retrieval-augmented generation):** the 3 most similar training recipes (TF-IDF search over 8,983 recipes) are added to Qwen3's prompt as examples.
4. **Checker:** flags ingredients that were not requested and asks Qwen3 to repair them (up to 2 attempts).

**Results (same GPT-2 drafts in every column)**

| Ingredients used in steps | GPT-2 alone | + Qwen3 edit | + Qwen3 edit + RAG |
|---|---:|---:|---:|
| 4 custom inputs | **100%** | 81% | 38% |
| 12 unseen recipes | **54%** | 18% | 21% |
| Passed the checker | **17%** | 8% | 8% |

**What we learned:** editing made recipes *worse*. Qwen3 dropped requested ingredients, added extras (salt, pepper, cheese, bread) and copied the reference recipes' format. It was trained to **write** recipes, not to **edit** them.

> **Speaker notes:** Part of the RAG drop is a formatting issue: some answers used bullet points, which the scorer did not read. Counting bullets too, both editing setups reach about 35% against 65% for GPT-2 alone, so the conclusion holds. Negative results are useful: they show the bottleneck is how the model was trained, not the idea of retrieval itself.

---

## Slide 4 — Next steps: using RAG to make Qwen3 reliable

**Why we are still optimistic:** fine-tuned Qwen3 is our best generator (62.6% on unseen recipes), after only 224 training recipes on a laptop CPU.

**How RAG can be used next**
- **Retrieve, then generate directly:** give Qwen3 similar recipes as context when it writes a recipe, instead of asking it to edit GPT-2's draft.
- **Train with retrieval:** fine-tune Qwen3 on prompts that already include retrieved examples, so it learns to use them for technique without copying their ingredients.
- **Better search:** semantic embeddings instead of keyword (TF-IDF) matching, retrieving by ingredients and cooking method.
- **Check and choose:** generate several drafts and pick the one that passes the ingredient checker.
- **More training:** a GPU and the full 9,000 recipes instead of 224.

**What we hope to achieve**
- Most requested ingredients used, with **no unrequested ones**
- Logical, non-repeating steps grounded in real Food.com methods
- A fast, small model that runs in our Streamlit app

> **Speaker notes:** The Streamlit app already offers all three models. Qwen3's adapter is published on Hugging Face (`Annemarie535257/qwen_model`) and loads in 16-bit to fit the app's memory. Recipes are still experimental drafts: ingredient scores do not check food safety, so a human should review any recipe before cooking.
