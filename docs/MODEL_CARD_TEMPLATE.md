# Model Card — `<model name>`

> Fill out one card per trained model artifact your team ships or relies on.
> A model card documents what a model is, how it was built, and where it is
> safe to use. Reviewers, auditors, and downstream consumers should be able
> to read one to decide whether the model is fit for purpose.

## Summary

| Field | Value |
|---|---|
| Model name | _e.g._ `byte-lm-transformer-v2` |
| Version / hash | _commit SHA, weight hash, or release tag_ |
| Author / owner | _team or individual responsible_ |
| Last updated | _YYYY-MM-DD_ |
| License | _e.g._ BSL 1.1, Apache 2.0, proprietary |
| Intended use | _one sentence: what the model is for_ |

## Architecture

- Family: _e.g._ Transformer / CNN / Diffusion / RL agent
- Layers / parameters: _e.g._ 12 encoder blocks, dModel=512, 25 M params
- Input shape: _e.g._ `[batch, ctxLen=512]` byte tokens (vocab=256)
- Output shape: _e.g._ `[batch, ctxLen, vocab]` logits
- Notable architectural choices: _e.g._ rotary position embeddings, ALiBi,
  pre-norm, weight tying, etc.

## Training

- Dataset(s): _name, version, license, size, source_
- Splits: _train / val / test ratios + any held-out evaluation sets_
- Preprocessing: _tokenization, normalization, augmentation, sampling_
- Optimizer + schedule: _e.g._ AdamW(lr=3e-4, β=(0.9, 0.98)) + NoamSchedule(warmup=4000)
- Hardware + duration: _GPUs/CPUs used, wall-clock_
- Seed(s): _RNG seeds that produced the published artifact_

## Evaluation

- Metrics tracked: _e.g._ top-1, top-5, perplexity, F1, AUC, BLEU, ROUGE
- Benchmark results: _table comparing this version to prior versions_
- Slice analysis: _performance per relevant input slice (language, domain, length, etc.)_
- Fairness / bias evaluation: _what was tested, what the result was_

## Limitations

- Out-of-distribution behavior: _known failure modes when inputs drift_
- Hallucination / fabrication: _for generative models, document rate + examples_
- Adversarial robustness: _what attacks were tested, what they did_
- Compute / memory ceilings: _practical limits at inference time_

## Intended use & out-of-scope use

- ✅ Use it for: _bounded, validated tasks_
- ❌ Do NOT use it for: _high-risk, regulated, or untested deployments_

## Data governance

- PII handling: _what PII the training data contained and how it was treated_
- Consent / provenance: _how the training data was sourced_
- Retention / deletion: _can the model "forget" specific training examples?_
- Telemetry: _what runtime data the model emits and where it goes_

## Versioning + supply chain

- Artifact location: _e.g._ `models/byte-lm-transformer-v2.bin` (S3 / HuggingFace / etc.)
- Artifact hash: _SHA256 of the weights file_
- Build provenance: _link to the CI run that produced the artifact_
- Dependencies: _AiDotNet version, native accelerator versions, etc._

## Change log

| Version | Date | Author | Notes |
|---|---|---|---|
| v0.1 | _YYYY-MM-DD_ | _name_ | _summary of changes vs. prior version_ |

---

> This template is informational. Maintaining it is your responsibility;
> the AiDotNet runtime does not enforce a card per artifact.
