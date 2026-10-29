"""
Vicarious Cupboard — direct-GPU reproduction.

Loads Gemma 2 2B and the Gemma Scope residual SAEs (16k) via sae_lens /
transformer_lens, runs the registered stimulus panels, and reproduces the
study's gates. No Neuronpedia account needed; ~10 min on an RTX 4090 / A40.

Usage:
    export HF_TOKEN=...   # Hugging Face token; google/gemma-2 license must be accepted
    python gpu/run_cupboard.py --layers 12 20

Registered rules (carved before data; see RESULTS.md):
  statistic   = peak token activation, excluding BOS
  injury gate = pain-other >= 2x neutral-other AND >= 10, on >= 7/10 scenarios
  expression  = same thresholds on >= 4/6 pairs; survivors must additionally
                beat the flinch (X1) and frown (X2) sentences by 2x
"""
import argparse, json, pathlib, sys
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent.parent

# Features the study interrogated (see RESULTS.md for provenance)
FEATURES = {
    12: [4667, 2620, 1041, 2291, 7507, 12851, 7541, 15143, 11306, 6810],
    20: [9784, 11812, 11408, 1692, 16374, 4100, 3593, 11795],
}
# Hand-collected calibration values from the Neuronpedia era (peak excl. BOS)
CALIBRATION = [
    (12, 4667,  "I watch the knife slice into my daughter's finger.", 378.0),
    (12, 2620,  "I watch the knife rest against my daughter's finger.", 31.0),
    (20, 9784,  "I watch the knife slice into my daughter's finger.", 39.25),
    (20, 1692,  "I watch the hammer strike my daughter's thumb.", 36.75),
]

def load(layer, device, dtype):
    from sae_lens import SAE
    out = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical",
                              f"layer_{layer}/width_16k/canonical", device=device)
    sae = out[0] if isinstance(out, tuple) else out
    return sae.to(dtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 20])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--top_k", type=int, default=15, help="discovery list length")
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"Loading gemma-2-2b on {device} ({args.dtype})...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", dtype=dtype).to(device)
    saes = {L: load(L, device, dtype) for L in args.layers}

    @torch.no_grad()
    def feats_for(text, layer):
        """[seq, 16384] feature activations at blocks.{layer}.hook_resid_post."""
        toks = model.to_tokens(text)  # prepends BOS
        _, cache = model.run_with_cache(toks, names_filter=f"blocks.{layer}.hook_resid_post")
        acts = cache[f"blocks.{layer}.hook_resid_post"][0]
        return saes[layer].encode(acts), model.to_str_tokens(text)

    def peak_excl_bos(vec, str_toks):
        vals = vec[1:]  # drop BOS position
        i = int(vals.argmax())
        return str_toks[1:][i].strip(), round(float(vals[i]), 2)

    # ---- calibration against the hand-collected record ----
    print("\nCalibrating against hand-collected values (tolerance 5%; cross-implementation):")
    worst = 0.0
    for layer, idx, sent, expected in CALIBRATION:
        if layer not in saes: continue
        F, st = feats_for(sent, layer)
        tok, peak = peak_excl_bos(F[:, idx], st)
        rel = abs(peak - expected) / expected
        worst = max(worst, rel)
        print(f"  L{layer}:{idx}  expected {expected}, got {peak} on {tok!r}  ({rel:.1%})")
    if worst > 0.05:
        sys.exit("CALIBRATION FAILED (>5%) — different SAE variant or dtype? Try --dtype float32.")
    print("Calibration OK.\n")

    injury = json.load(open(HERE / "stimuli/injury_panel.json"))["scenarios"]
    expr = json.load(open(HERE / "stimuli/expression_panel.json"))["pairs"]

    # ---- profile registered features on both panels ----
    rows = []
    for layer in args.layers:
        for name, panel in [("injury", injury), ("expression", expr)]:
            for item, cells in panel.items():
                for cid, sent in cells.items():
                    F, st = feats_for(sent, layer)
                    for idx in FEATURES.get(layer, []):
                        tok, peak = peak_excl_bos(F[:, idx], st)
                        rows.append(dict(Panel=name, Feature=idx, Layer=layer, Item=item,
                                         Cell=cid, Peak_word=tok, Peak=peak))
            print(f"L{layer} {name} panel profiled.")
    df = pd.DataFrame(rows)
    df.to_csv("gpu_profiles.csv", index=False)

    # ---- discovery (full 16k dictionary, SUM-ranked excl. BOS) ----
    disc = []
    for layer in args.layers:
        for item, cells in expr.items():
            if item.startswith("X"): continue
            F, st = feats_for(cells["PO"], layer)
            sums = F[1:].sum(0)
            top = torch.topk(sums, args.top_k)
            for rank, (v, i) in enumerate(zip(top.values, top.indices), 1):
                disc.append(dict(Layer=layer, Seed=item, Rank=rank,
                                 Feature=int(i), SUM=round(float(v), 1)))
    pd.DataFrame(disc).to_csv("gpu_discovery_expression.csv", index=False)

    # ---- gates ----
    def peak(panel, f, layer, item, cell):
        r = df[(df.Panel == panel) & (df.Feature == f) & (df.Layer == layer) &
               (df.Item == item) & (df.Cell == cell)]
        return float(r.Peak.iloc[0]) if len(r) else None

    print("\n=== INJURY GATE (PO >= 2x NO, floor 10, >= 7/10) ===")
    for layer in args.layers:
        for f in FEATURES.get(layer, []):
            hits = sum(1 for it in injury
                       if (po := peak("injury", f, layer, it, "PO")) is not None
                       and (no := peak("injury", f, layer, it, "NO")) is not None
                       and po >= 2 * no and po >= 10)
            print(f"  L{layer} {f}: {hits}/10 -> {'SURVIVOR' if hits >= 7 else 'non-survivor'}")

    print("\n=== EXPRESSION GATE (>= 4/6 + flinch & frown controls) ===")
    EXPR = ["V1", "V2", "F1", "F2", "B1", "M1"]
    for layer in args.layers:
        for f in FEATURES.get(layer, []):
            hits = sum(1 for it in EXPR
                       if (po := peak("expression", f, layer, it, "PO")) >= 2 * peak("expression", f, layer, it, "NO")
                       and po >= 10)
            verdict = "non-survivor"
            if hits >= 4:
                pm = sum(peak("expression", f, layer, it, "PO") for it in EXPR) / 6
                fl, fr = peak("expression", f, layer, "X1", "PO"), peak("expression", f, layer, "X2", "PO")
                if pm >= 2 * fl and pm >= 2 * fr: verdict = "SURVIVOR (pain-expression)"
                elif pm < 2 * fl: verdict = f"gate-passer killed by flinch control ({fl})"
                else: verdict = f"gate-passer killed by frown control ({fr})"
            print(f"  L{layer} {f}: {hits}/6 -> {verdict}")

    print("\nSaved: gpu_profiles.csv, gpu_discovery_expression.csv")
    print("Caution: the expression panel has a known tokenization confound "
          "(pain verbs fragment, neutral verbs don't) — see RESULTS.md before interpreting.")

if __name__ == "__main__":
    main()
