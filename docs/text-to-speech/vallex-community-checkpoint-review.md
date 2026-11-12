# VALL-E X: paper and community checkpoint evidence

[Microsoft's paper](https://arxiv.org/abs/2303.03926) describes English/Mandarin codec language modeling with language IDs added to AR acoustic embeddings. Microsoft did not publish the original learned weights or BigCiDian lexicon. The paper configuration remains distinct from the community reproduction below.

Community source: `Plachtaa/VALL-E-X` at `3faaf8ccadb154d63b38070caf518ce9309ea0f4`. Checkpoint: `Plachta/VALL-E-X` at `d7240d71278df7b8ab9960f5b91107de15a2d016`, `vallex-checkpoint.pt`, 1,482,302,113 bytes, SHA-256 `c5fcd05ee0c9c84a16a7b44495c46262177e66d5d454c20ca5f1da9832dbd5ac`.

The checkpoint uses 12-layer, 16-head, width-1024 AR and NAR Transformers, 2048 text embedding rows, three language IDs, language embeddings on phonemes, a separate audio BOS, and a fixed 70-token character phoneme vocabulary. Its NAR decoder reads the full combined prompt/target text. These differ from the paper defaults and the earlier native checkpoint assumptions. `VALLEXOptions.CommunityCheckpoint()` and `VALLEXCloneOptions.CommunityCheckpoint()` now select these choices explicitly. Paper defaults remain bilingual with language IDs on AR acoustic tokens.

`tools/reference-data/vallex_pretrained_reference.py` loads the complete original community checkpoint into its original inference implementation. Hooks capture projection outputs; only sampling is controlled to expose two generated frames, their three AR steps, and all seven NAR stages. No neural forward method is replaced. It records original float32 execution, original double execution retaining the source's float32 positional tables, and a separately labeled double execution with regenerated nonlearned positional tables. Native double arithmetic is compared strictly with the last variant.

The complete core comparison and production public cloning-facade checkpoint-loader run pass: maximum logit difference 1.65e-13, and all 16 generated codec tokens identical. The importer also validates the serialized NAR heads against their tied embedding aliases instead of silently ignoring conflicting aliases. The production verification uses Build38; subsequent source regression runs include these tests.

`tools/reference-data/vallex_phone_reference.py` executes the original Hugging Face tokenizer JSON. Nineteen integrated source regressions pass, including all 70 vocabulary IDs, special tokens, whitespace, unknown characters, supplementary Unicode, copied options, existing paper behavior and cloning-facade checks. Eleven UniAudio codec-stage regressions are recorded separately.

This establishes community component compatibility, not equivalence to unreleased Microsoft learned parameters, a recreation of its training corpus, or a complete cross-language listening evaluation. Japanese is a checkpoint language-table entry; callers using Japanese phonemes can supply them explicitly. Automatic Japanese text phonemization is not established by these checks.
