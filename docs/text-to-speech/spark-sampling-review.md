# SparkTTS generation-policy review

The primary source is [the author inference loop](https://github.com/SparkAudio/Spark-TTS/blob/2f1ea9082400547242641f5271b6f941c9f439d1/cli/SparkTTS.py), whose requirements pin Transformers 4.46.2. The released defaults are temperature 0.8, top-k 50 and top-p 0.95. The existing full trained Qwen and BiCodec checkpoint comparisons verify neural computations separately.

`tools/reference-data/spark_sampling_reference.py` executes the original temperature, top-k and top-p processor class bodies from the uninstalled Transformers 4.46.2 wheel. It also verifies their order in the original generation pipeline and records author, policy, pipeline and wheel hashes. No installed Transformers package is replaced; Chatterbox continues to use its independently pinned 5.2.0 reference.

Nine controlled distributions cover the released defaults, tied top-k thresholds, temperature before truncation, nucleus boundaries, top-p zero and unrestricted sampling. Six fixed inverse-CDF quantiles per case and both sides of resolved cumulative boundaries at 1e-9 compare probabilities without assuming identical random generators. A tenth test runs the actual reduced Qwen generation loop with zero learned tensors: top-k one retains all eight tied logits, allowing configured EOS 7 to terminate immediately at the reference quantile. Old native code instead keeps only token zero and emits three tokens.

Seven of ten tests fail against the previous production implementation. The repair keeps every logit at the kth threshold and uses HF ascending-tail nucleus filtering, retaining at least one candidate. That nucleus operation is shared with Chatterbox, whose own ten independent reference tests also pass against the repaired source. Tied output ordering across arbitrary devices is not certified by these controlled CPU fixtures.

The integration case uses deliberately controlled parameters. It complements the released-checkpoint Qwen logits and full codec comparisons; it does not establish speech quality, identical GPU randomness, or a complete distributional end-to-end benchmark.

Build51 compiles the production library on net8.0 and net471 without errors. The normal repository test project passes all 431 reviewed cases on both targets, with zero failures or skips, including the Spark and Chatterbox policy suites. The checkpoint-audit tool also builds without errors.
