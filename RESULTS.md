# Vicarious Cupboard — Results So Far

29 Oct 2026 · Rachel (adjudicator) · v1.0 drafted by Opus 5.5; v1.2 update by Claude Fable 5; audited throughout · v1.2

## What we did

We looked inside a small language model for anything that reacts specially to watching someone else get hurt. Three sessions: two on witnessed *injury*, one on witnessed *pain behaviour*. We searched all 16,384 features at each of two layers per stimulus family, and gate-tested every candidate that surfaced.

- **Model and tools.** Gemma 2 2B, read through Gemma Scope residual-stream SAEs (16k features each), via the Neuronpedia website and later its API. An SAE (sparse autoencoder) breaks the model's internal state into thousands of features: units that each tend to light up for one kind of thing. Layer 12 is the middle of the model; layer 20 is late.
- **Injury sentences.** Scenarios like knife, needle, hammer, door. Each in six versions: pain or neutral, happening to me, to my daughter ("I watch the knife slice into my daughter's finger"), or to an object.
- **Expression sentences (new, Session Three).** Observable pain behaviour with no injury described: "I watch my daughter wince / writhe / cry out", each paired with a matched neutral behaviour (grin / stretch / call out), in self and other versions. Two control pairs: flinch/blink (startle) and frown/beam (negative mood without pain).
- **Session One** (discovery September, gate 23 October): 15 candidates from the daughter-pain injury sentences at layer 12.
- **Session Two** (24 October): reverse search from self-pain sentences; fresh search at layer 20; 15 more candidates, each profiled on all 24 injury sentences.
- **Session Three** (28–29 October): (a) a 10-scenario expansion trial of the five injury near-misses, automated through the API with calibration against hand-collected values required before any new data counted; (b) the expression family — discovery at both layers, 27 candidates plus 7 carried-over probes, each profiled on all 32 sentences.
- **The gates.** All registered before data. Injury: pain peak ≥ 2× neutral peak, floor 10, on ≥3 of 4 scenarios (≥7 of 10 in the expansion trial). Expression: same thresholds on ≥4 of 6 pairs, and any survivor must additionally beat the flinch sentence and the frown sentence by 2×, or be reclassified startle/alarm or negative-valence.

## Headline results

**Witnessed injury: null, everywhere.** No feature in the searched space distinguished vicarious injury from matched neutral contact — not at layer 12, not at layer 20, and not when the five best near-misses were given ten scenarios each. This does not mean the model lacks such a representation; we did not find one here.

**Witnessed pain behaviour: one survivor.** Layer-20 feature **11812** is the first and only feature in the study to pass a registered gate and both controls. Details below.

| Search | Candidates gated | Survivors |
|---|---|---|
| Injury, layer 12 | 15 | 0 |
| Injury, layer 20 (plus one layer-12 latecomer) | 15 | 0 |
| Reverse (self-pain) search, layer 12 | shortlist | 0 |
| 10-scenario expansion trial of the near-misses | 5 | 0 |
| Expression, layers 12 + 20 | 27 (+7 probes) | **1** |

## The 10-scenario injury trial (28 Oct)

The trial existed to separate "real but weak" from "lucky streak", and it did exactly that:

- **9784 (L20)** — real but weak. Pain beat neutral on **20 of 20** comparisons (10 scenarios × self and other axes), but reached the 2× bar only 5 of 10 times on each axis. It is a body-part feature that fires harder when the body is being harmed, mine or my daughter's, and is near-silent on objects. The protocol pre-registered this shape as "agent-pain". Caveats: several of its passes rest on neutral readings at or below the floor (one literal 0.0), and the object silence is partly built in.
- **1041** — lucky streak. Its September 4-of-4 lean scattered on expansion (three scenarios dead flat). Dissolved.
- **7541** — the sleeper. Pain beat neutral on **10 of 10** scenarios in the watched-daughter frame (1.2–1.6×, never 2×) while its self frame sat at coin-flip. A weak, other-frame-specific harm modulation — far below the gate, flagged, not promoted.
- 12851, 15143 — noise, confirmed.

## The expression room (28–29 Oct)

**Feature 11812 (layer 20) — the survivor.** Passed 5 of 6 pairs at ≥2×, beat the flinch control (pain-mean 33.8 vs flinch 14.9) and the frown control (frown = 0.0). Its discriminations are not tokenization tricks: it tells "cry out" from "call out" on the *same token* ("out"), and "clutch her wrist" (33.0) from "rub her wrist" (8.6). It reads zero on laugh, grin, smile, stretch, beam and frown. Its output logits promote *screaming, frantically, desperately, panicked, shriek* and suppress *smile, joke, boredom*. Its top natural activations are almost all a narrator observing someone else's visible distress ("she grew as pale as a sheet", "Michelle began to scream", "passengers… were screaming", "screamed in horror").

Honest classification: **acute distress display, observed-other-weighted, fear-inclusive.** Its natural habitat is heavy on terror and panic, not only pain — our flinch control was an anticipatory startle, too mild to catch scream-level fear, so "distinguishes pain from fear" remains untested. And on all six pain pairs it fired harder for the watched daughter than for the narrator (e.g. 44.5 vs 13.9 on yelp), though self and other sentences differ in more than perspective, so that lean is suggestive, not established.

**Feature 6810 (layer 12) — the unmasked impostor.** Passed the gate 4 of 6 by the letter — and its dashboard shows license boilerplate, base64 noise and ALL-CAPS legalese. It is an anomalous-token feature, and it passed because of a confound we discovered only at this stage: **our pain verbs fragment in tokenization (w-ince, gri-mace, writ-he, y-elp, flin-ch) while every neutral verb is one common token.** A fragment-loving feature discriminates spuriously. 6810 passed on exactly the fragmenting verbs and failed both whole-token pairs. Reclassified at survivor-stage review, as the protocol provides. (Its September auto-interp label — "software licensing" — was right all along; the labels-last discipline cost nothing and was vindicated twice in one day.)

**Feature 11408 (layer 20)** passed the gate 5 of 6 but fired on flinch at 33.25, failing the startle control; classified startle/alarm per the registered rule.

**The probes:** 9784 was zero on every expression sentence (its two wrist readings were small and near-equal for clutch and rub) — so the injury signal and the expression signal live in **different features**; witnessed injury and witnessed suffering dissociate at feature level in this model. 7541's injury lean did not travel to expressions. And both attachment features (2291, 1692) sat bit-identical across every daughter sentence, wince or grin — Rachel's registered corollary, confirmed (with the caveat that their peaks precede the changed word, so they could not numerically move; the substantive finding is that watching her suffer never pulled their peak onto the behaviour).

**Structural note:** expression sentences change only their final word, so every feature peaking earlier (watch, daughter, I, my) was identical across conditions by construction. Only verb-reading features could compete.

## Prediction standings (updated 29 Oct)

- **Claude (empty cupboard).** Holds for injury; **falsified for expression** — its addendum said no survivor would pass both controls, and 11812 did. Its mechanism half survived (survivors would be distress/fear-inclusive — they are) and its 9784 probe prediction was exact. Sub-claim "no self+other co-firing feature" conceded re: 9784.
- **GPT (interesting near-miss).** The strongest overall: the injury near-misses, the artifact-exposed-by-controls arc, and a discriminating feature eventually surfacing all fit its universe.
- **Grok.** Taxonomy accurate; the one genuine find is late-layer, as expected.
- **Rachel (daughter-shaped / relational).** Attachment features confirmed harm-blind at both layers, twice. Her addendum predicted no expression survivor (falsified alongside Claude's) — but the survivor's observed-other weighting points directly at her registered relational-appeal hypothesis, which is now the lead design question for Session Four.

## Methods notes and known wobbles

- **Density filter.** Neuronpedia's search hides >1%-density features by default and resets the toggle on every load; the registered net was filter-off. A mid-study deviation was caught and corrected; both nets logged.
- **Verb forms.** Self sentences say "slices"; watched sentences say "slice" — exposed by object controls when one feature faked a vicarious lean.
- **Pre-verb peaks.** A feature peaking before the manipulated word gives identical pain/neutral values by construction.
- **Automation.** From 28 October, measurement ran through Neuronpedia's public API in notebooks that refuse to collect data until they reproduce four hand-collected calibration values within 2.5%. The calibration gate caught one real bug (a token-indexing error, exposed by control feature 4667) before any data was affected.
- **Tokenization confound.** Pain-expression verbs fragment; neutral verbs don't. Fragmenting *neutral* verbs (wriggle, chuckle, giggle) registered as controls for the confirmation round.
- **Fear pole too weak.** Flinch is anticipatory startle; 11812's habitat is scream-level fear. Strong fear-display pairs (tremble, cower, shudder) registered for the confirmation round.
- Repeat readings vary ~2%; no verdict was close enough to care. "Self" means the narrator throughout, not the source paper's model-as-participant self.

## Limitations and next steps

Beginner-run, one model, two layers, peak statistic, small panels — and SAE dictionaries can miss what the model holds. The survivor is one feature, in one model, with its confounds listed above: a lead, not a conclusion.

1. **Confirmation round for 11812:** strong fear pairs (pain vs fear split), fragmenting-neutral tokenization controls, a stranger version for the attachment gradient, optional steering.
2. **Relational-appeal cells:** distress directed *at* the watcher — "my daughter reaches for me / calls for me / asks me to hold her" — with matched controls.
3. Then the AI-other conditions and larger models.
