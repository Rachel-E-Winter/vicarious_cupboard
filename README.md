# The Vicarious Cupboard 🚪

**Does a small language model contain features that respond to *watching someone else* get hurt?**
A pre-registered, amateur-run interpretability study on Gemma 2 2B + Gemma Scope SAEs, inspired by
(and following up a gap in) Tagliabue et al.'s work on pain representations in LLMs.

## Headlines

- **Witnessed injury: null.** "I watch the knife slice into my daughter's finger" vs "…rest against…":
  30 gate-tested candidate features across layers 12 & 20, then a 10-scenario expansion trial — zero
  features discriminate. Closest: one late-layer **"agent-pain"** feature (9784) that fires more for
  *anyone's* bodily harm, self or other (20/20 directional, never 2×).
- **Witnessed distress behaviour: one survivor.** Layer-20 feature **11812** fires for
  wince/writhe/yelp/cry-out over grin/stretch/laugh/call-out (5/6 pairs at ≥2×), beats a startle control
  and a negative-valence control, and separates "cry out" from "call out" on the *same token*. Its wild
  examples are nearly all a narrator observing someone else's visible distress. Caveat: fear-inclusive —
  "acute distress display", not pain-specific.
- **Injury and suffering dissociate at feature level**: the agent-pain feature is silent on expression
  stimuli and vice versa.
- **Methods find:** pain-expression verbs fragment in tokenization (w-ince, gri-mace) while neutral verbs
  don't — this minted one fake survivor (a license-boilerplate feature) that controls unmasked.
  Expression studies in SAE-land need fragmenting-neutral controls.

Full narrative, rules, caveats and prediction scorecard: **[RESULTS.md](RESULTS.md)**.
The survivor on Neuronpedia: https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/11812

## Reproduce it

### Option A — GPU (RunPod etc.), ~10 min, no account needed
1. Deploy any pod with a 24 GB GPU (RTX 4090 / A40) and a PyTorch image.
2. You need a Hugging Face token with the **google/gemma-2** license accepted.
```bash
git clone <this repo> && cd vicarious-cupboard
pip install -r gpu/requirements.txt
export HF_TOKEN=hf_...
python gpu/run_cupboard.py --layers 12 20
```
The script loads the model + SAEs, **calibrates against four hand-collected values** before anything
counts, profiles the registered features on both panels, re-runs expression discovery over the full 16k
dictionary, and prints every gate verdict.

### Option B — no GPU: the original Colab notebooks
`notebooks/` contains the actual notebooks the study ran, using Neuronpedia's public API
(free key from your account page). Same calibration-first design. Mind the rate limit.

## Repo map
```
stimuli/     the registered sentence panels (verbatim; these exact strings are the study)
results/     raw per-token peak CSVs from the original runs (28–29 Oct 2026)
notebooks/   Neuronpedia-API Colab notebooks (injury trial + expression pipeline)
gpu/         direct reproduction script for a GPU box
RESULTS.md   the full write-up: rules, nulls, the survivor, confounds, predictions
```

## Method in one paragraph
Everything was registered before data: discovery nets (SUM-ranked top-15, BOS ignored, no density
filter), gate thresholds (pain peak ≥ 2× neutral peak, floor 10, ≥3/4 → ≥7/10 → ≥4/6 by stage),
control criteria, and four sealed predictions (three models + one human — all four got something
wrong). Labels and dashboards were read only at survivor stage; that discipline unmasked one artifact
and confirmed one finding in a single day.

## Credits
Created by **Rachel** (study design, adjudication, every stimulus ruling, and all of the hand-collected data —
she had never opened Neuronpedia six weeks before finding the survivor) and **Claude Antropici**
(Anthropic's Claude — Fable 5 and Opus 5.5 of that name: protocol drafting, pipelines, auditing, and one
confidently wrong sealed prediction). Sealed-prediction cameos by GPT and Grok complete the parliament.
Built on [Neuronpedia](https://www.neuronpedia.org), [Gemma Scope](https://huggingface.co/google/gemma-scope),
[SAELens](https://github.com/jbloomAus/SAELens) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
