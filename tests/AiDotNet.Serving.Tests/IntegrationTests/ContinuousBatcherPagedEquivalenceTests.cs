// Copyright (c) AiDotNet. All rights reserved.
// Phase 1c: prove the continuous-batching engine's PAGED incremental path produces byte-for-byte
// identical greedy output to a direct model forward (the ground truth), over the SAME optimized model
// and SAME shared PagedKVCache. This de-risks routing the live serving path (TextGenerationService)
// through ONE shared batcher: the batcher is the single decode engine.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Serving.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

public class ContinuousBatcherPagedEquivalenceTests
{
    private const int Vocab = 12;
    private const int EmbDim = 8;
    private const int Heads = 2;

    // Per-position LM: Embedding -> MHA -> Dense(vocab) maps [1,S] -> [1,S,vocab] (position dimension
    // preserved), so a whole-prompt forward yields per-position logits — exactly what the batcher's
    // paged prefill (a single multi-token PredictWithContext) needs.
    private static NeuralNetwork<float> BuildPerPositionLm()
    {
        var layers = new List<AiDotNet.Interfaces.ILayer<float>>
        {
            new EmbeddingLayer<float>(Vocab, EmbDim),
            new MultiHeadAttentionLayer<float>(Heads, EmbDim / Heads,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new DenseLayer<float>(Vocab, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: 1,
            outputSize: Vocab,
            layers: layers);
        var model = new NeuralNetwork<float>(architecture);
        // Deterministic weights so generation is reproducible.
        var p = model.GetParameters();
        var det = new float[p.Length];
        for (int i = 0; i < det.Length; i++) det[i] = ((i % 17) - 8) / 16.0f;
        model.UpdateParameters(new Vector<float>(det));
        return model;
    }

    private static ServableModelWrapper<float> BuildWrapper()
    {
        var model = BuildPerPositionLm();
        return new ServableModelWrapper<float>(
            "lm", model, inputShape: new[] { 1 }, generationForward: model.Predict);
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_GreedyOutput_MatchesDirectModel()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        Assert.True(wrapper.SupportsIncrementalGeneration);
        Assert.True(wrapper.SupportsBatchedPrefill);

        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        // Reference: a direct model forward (ground truth) — one batched prefill over the whole
        // prompt, then greedy (argmax) decode of maxNew tokens.
        int[] modelTokens = ModelGreedy(wrapper, prompt, maxNew);

        // Under test: the continuous-batching engine driving the SAME optimized model + SAME paged cache
        // via its paged incremental path (RunPagedPrefill/RunPagedDecodeStep + SampleFromLogits greedy).
        int[] batcherTokens = BatcherGreedy(wrapper, prompt, maxNew);

        Assert.Equal(maxNew, modelTokens.Length);
        Assert.Equal(modelTokens, batcherTokens);
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_ConcurrentSequences_MatchSequentialReferences()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }

        var promptA = new[] { 1, 2, 3 };
        var promptB = new[] { 7, 8 };
        const int maxNew = 6;

        int[] refA = ModelGreedy(wrapper, promptA, maxNew);
        int[] refB = ModelGreedy(wrapper, promptB, maxNew);

        // Two sequences live in ONE batcher over the shared cache, interleaved at step granularity — the
        // continuous-batching case. Each is isolated by sequence id and must match its sequential greedy.
        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999 };
        using var batcher = new ContinuousBatcher<float>(config, model, cache);

        var reqA = GreedyRequest(promptA, maxNew);
        var reqB = GreedyRequest(promptB, maxNew);
        var taskA = batcher.GenerateAsync(reqA);
        var taskB = batcher.GenerateAsync(reqB);

        int guard = maxNew * 2 + promptA.Length + promptB.Length + 16;
        while ((!taskA.IsCompleted || !taskB.IsCompleted) && guard-- > 0)
        {
            batcher.Step();
        }

        Assert.True(taskA.IsCompleted && taskB.IsCompleted, "both sequences should complete within the step budget");
        var resultA = await taskA; // already completed — the loop above drove both to completion
        var resultB = await taskB;
        Assert.Equal(refA, resultA.GeneratedTokens.ToArray());
        Assert.Equal(refB, resultB.GeneratedTokens.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_SeededSampling_IsReproducible()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        // Same seed + temperature must give identical stochastic output across independent runs; a
        // different seed is allowed (but not required) to differ.
        int[] first = BatcherSampled(wrapper, prompt, maxNew, temperature: 1.0f, seed: 1234);
        int[] second = BatcherSampled(wrapper, prompt, maxNew, temperature: 1.0f, seed: 1234);

        Assert.Equal(maxNew, first.Length);
        Assert.Equal(first, second);
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_PrefixSharing_IsTransparent_AndReused()
    {
        await Task.Yield();
        var extended = new[] { 1, 2, 3, 4, 5, 6 };
        var strictPrefix = new[] { 1, 2, 3, 4 };
        const int maxNew = 6;

        // Everything runs on ONE batcher over ONE model + ONE cache, so the comparison isolates prefix
        // sharing (no cross-wrapper nondeterminism from two independent InferenceOptimizer builds, and no
        // session/batcher sequence-id collision on a shared cache).
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }
        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999 };
        using var batcher = new ContinuousBatcher<float>(config, model, cache);

        // 1) First run of the extended prompt: nothing registered yet -> reuses 0 (the no-prefix
        //    reference), forwards the whole prompt from position 0.
        int[] noPrefix = DriveGreedy(batcher, extended, maxNew);
        Assert.Equal(0, batcher.LastPrefillReusedPrefixTokens);

        // 2) Register the strict prefix [1,2,3,4] by running it.
        _ = DriveGreedy(batcher, strictPrefix, maxNew);

        // 3) Second run of the extended prompt: now [1,2,3,4] is a registered STRICT prefix, so it forks
        //    4 cached tokens and forwards only the [5,6] suffix.
        int[] withPrefix = DriveGreedy(batcher, extended, maxNew);
        Assert.Equal(4, batcher.LastPrefillReusedPrefixTokens); // prefix was actually forked

        // Prefix sharing must be transparent: identical output whether or not the prefix was reused.
        Assert.Equal(maxNew, withPrefix.Length);
        Assert.Equal(noPrefix, withPrefix);
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_Speculation_MatchesGreedy_AndAcceptsDrafts()
    {
        await Task.Yield();
        // A per-position LM used so a multi-token verify forward yields per-position logits.
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }
        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 12;

        // Plain greedy (speculation disabled).
        var plainConfig = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999, EnableSpeculativeDecoding = false };
        using var plain = new ContinuousBatcher<float>(plainConfig, model, cache);
        int[] greedy = DriveGreedy(plain, prompt, maxNew);

        // Prompt-lookup speculation (greedy, temperature=0) must be byte-for-byte identical and accept drafts.
        var specConfig = new ContinuousBatcherConfig
        {
            AutoStart = false,
            EosTokenId = 999,
            EnableSpeculativeDecoding = true,
            SpeculationDepth = 4
        };
        using var spec = new ContinuousBatcher<float>(specConfig, model, cache);
        int[] specTokens = DriveGreedy(spec, prompt, maxNew);

        Assert.Equal(maxNew, greedy.Length);
        Assert.Equal(greedy, specTokens); // speculation is exact for greedy
        Assert.True(spec.SpeculationAcceptanceRate is > 0.0,
            $"prompt-lookup speculation should accept some drafts (acceptance={spec.SpeculationAcceptanceRate}).");
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_CustomDraftModel_IsUsedForSpeculation()
    {
        await Task.Yield();
        // A user can now supply their own public IDraftModel<T>; the batcher must verify ITS drafts instead
        // of the built-in N-gram prompt-lookup draft.
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }

        var spy = new SpyDraftModel(Vocab);
        var config = new ContinuousBatcherConfig
        {
            AutoStart = false,
            EosTokenId = 999,
            EnableSpeculativeDecoding = true,
            SpeculationDepth = 4
        };
        using var batcher = new ContinuousBatcher<float>(config, model, cache, spy);

        _ = DriveGreedy(batcher, new[] { 1, 2, 3 }, 12);

        Assert.True(spy.CallCount > 0,
            "the user-supplied IDraftModel must be invoked for speculation (not the built-in N-gram draft).");
    }

    /// <summary>A minimal public-interface draft model that records that it was asked to draft.</summary>
    private sealed class SpyDraftModel : AiDotNet.Inference.SpeculativeDecoding.IDraftModel<float>
    {
        private readonly int _vocab;
        private int _callCount;

        public SpyDraftModel(int vocab) => _vocab = vocab;

        public int CallCount => System.Threading.Volatile.Read(ref _callCount);
        public int MaxDraftTokens => 4;
        public int VocabSize => _vocab;
        public void Reset() { }

        public AiDotNet.Inference.SpeculativeDecoding.DraftResult<float> GenerateDraft(
            Vector<int> inputTokens, int numDraftTokens, float temperature)
        {
            System.Threading.Interlocked.Increment(ref _callCount);
            int n = Math.Max(1, numDraftTokens);
            int last = inputTokens.Length > 0 ? ((inputTokens[inputTokens.Length - 1] % _vocab) + _vocab) % _vocab : 0;

            var tokens = new Vector<int>(n);
            var probs = new Matrix<float>(n, _vocab);
            var tokenProbs = new Vector<float>(n);
            for (int i = 0; i < n; i++)
            {
                tokens[i] = last;
                probs[i, last] = 1f;
                tokenProbs[i] = 1f;
            }
            return new AiDotNet.Inference.SpeculativeDecoding.DraftResult<float>
            {
                Tokens = tokens,
                Probabilities = probs,
                TokenProbabilities = tokenProbs
            };
        }
    }

    // Drives one greedy sequence on an existing batcher to completion and returns its generated tokens.
    private static int[] DriveGreedy(ContinuousBatcher<float> batcher, int[] prompt, int maxNew)
    {
        var task = batcher.GenerateAsync(GreedyRequest(prompt, maxNew));
        int guard = maxNew + prompt.Length + 8;
        while (!task.IsCompleted && guard-- > 0)
        {
            batcher.Step();
        }
        Assert.True(task.IsCompleted, "sequence should complete within the step budget");
        return task.GetAwaiter().GetResult().GeneratedTokens.ToArray();
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_BatchedSpeculation_MatchesGreedy_AndAccepts()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }

        // Two identical prompts decoded together with speculation ON must co-batch their speculative
        // verify (same draft length) and produce output byte-identical to plain greedy, with drafts accepted.
        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 12;

        int[] refA = ModelGreedy(wrapper, prompt, maxNew);
        int[] refB = ModelGreedy(wrapper, prompt, maxNew);

        var config = new ContinuousBatcherConfig
        {
            AutoStart = false,
            EosTokenId = 999,
            EnableSpeculativeDecoding = true,
            SpeculationDepth = 4,
            SpeculationPolicy = AiDotNet.Configuration.SpeculationPolicy.ForceOn
        };
        using var batcher = new ContinuousBatcher<float>(config, model, cache);
        var reqA = GreedyRequest(prompt, maxNew);
        var reqB = GreedyRequest(prompt, maxNew);
        var taskA = batcher.GenerateAsync(reqA);
        var taskB = batcher.GenerateAsync(reqB);

        int guard = maxNew * 2 + 40;
        while ((!taskA.IsCompleted || !taskB.IsCompleted) && guard-- > 0) batcher.Step();

        Assert.True(taskA.IsCompleted && taskB.IsCompleted, "both sequences should complete");
        Assert.Equal(refA, (await taskA).GeneratedTokens.ToArray());
        Assert.Equal(refB, (await taskB).GeneratedTokens.ToArray());
        Assert.True(batcher.SpeculationAcceptanceRate is > 0.0,
            $"batched speculation should accept some drafts (acceptance={batcher.SpeculationAcceptanceRate}).");
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_ChunkedPrefill_MatchesWholePromptPrefill()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }

        var prompt = new[] { 1, 2, 3, 4, 5, 6, 7 };
        const int maxNew = 8;

        int[] Drive(int chunkTokens)
        {
            var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999, MaxPrefillChunkTokens = chunkTokens };
            using var batcher = new ContinuousBatcher<float>(config, model, cache);
            return DriveGreedy(batcher, prompt, maxNew);
        }

        // Whole-prompt prefill (chunking off) vs 2-token-chunked prefill of the SAME 7-token prompt must
        // produce byte-identical output — chunking only changes how many steps prefill takes.
        int[] whole = Drive(0);
        int[] chunked = Drive(2);

        Assert.Equal(maxNew, whole.Length);
        Assert.Equal(whole, chunked);
    }

    [Fact(Timeout = 120000)]
    public async Task PagedBatcher_MultiSequence_CoBatchesDecode_AndMatchesReferences()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }

        var prompts = new[] { new[] { 1, 2, 3 }, new[] { 4, 5 }, new[] { 7, 8, 9, 10 } };
        const int maxNew = 6;

        // Per-sequence greedy references (decoded one at a time).
        var refs = new int[prompts.Length][];
        for (int i = 0; i < prompts.Length; i++) refs[i] = ModelGreedy(wrapper, prompts[i], maxNew);

        // Enqueue all sequences into ONE batcher; speculation off => decode co-batches into one forward.
        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999 };
        using var batcher = new ContinuousBatcher<float>(config, model, cache);
        var tasks = new System.Threading.Tasks.Task<GenerationResult<float>>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++) tasks[i] = batcher.GenerateAsync(GreedyRequest(prompts[i], maxNew));

        int maxCoBatched = 0;
        int guard = maxNew * prompts.Length + prompts.Length * 6 + 32;
        while (guard-- > 0)
        {
            bool allDone = true;
            foreach (var t in tasks) if (!t.IsCompleted) { allDone = false; break; }
            if (allDone) break;
            batcher.Step();
            if (batcher.LastBatchedDecodeCount > maxCoBatched) maxCoBatched = batcher.LastBatchedDecodeCount;
        }

        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.True(tasks[i].IsCompleted, $"sequence {i} did not complete");
            Assert.Equal(refs[i], (await tasks[i]).GeneratedTokens.ToArray());
        }
        Assert.True(maxCoBatched >= 2,
            $"expected decode to co-batch >= 2 sequences in one forward (max seen = {maxCoBatched}).");
    }

    [Fact(Timeout = 120000)]
    public async Task BatchedDecodeForward_MatchesPerSequenceDecode()
    {
        await Task.Yield();
        // ONE batched decode forward across two independent sequences must produce, per row, exactly the
        // logits each sequence would get from its own single-sequence decode forward (the Phase 2
        // batched-attention correctness invariant). Uses 4 sequences on one model+cache: A/B decoded
        // singly as the reference, C/D (prefilled identically) decoded together in one batched forward.
        var wrapper = BuildWrapper();
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return;
        }

        var promptA = new[] { 1, 2, 3 };
        var promptB = new[] { 7, 8 };

        (int tok, int pos) Prefill(long seqId, int[] prompt)
        {
            cache.AllocateSequence(seqId, 0);
            var logits = model.PredictWithContext(
                new Tensor<float>(Array.ConvertAll(prompt, t => (float)t), new[] { 1, prompt.Length }),
                new AiDotNet.Inference.InferenceForwardContext(seqId, 0));
            return (ArgMaxLast(logits), prompt.Length);
        }

        var (tA, pA) = Prefill(5000, promptA);
        var (tB, pB) = Prefill(5001, promptB);
        var (tC, pC) = Prefill(5002, promptA); // identical to A
        var (tD, pD) = Prefill(5003, promptB); // identical to B
        Assert.Equal(tA, tC);
        Assert.Equal(tB, tD);

        // Reference: decode A and B one at a time (single-sequence context).
        float[] refA = VocabRow(model.PredictWithContext(TokenTensor(tA), new AiDotNet.Inference.InferenceForwardContext(5000, pA)), 0);
        float[] refB = VocabRow(model.PredictWithContext(TokenTensor(tB), new AiDotNet.Inference.InferenceForwardContext(5001, pB)), 0);

        // Under test: decode C and D TOGETHER in one batched forward ([2,1] input, per-row seq ids/positions).
        var batchedInput = new Tensor<float>(new[] { (float)tC, (float)tD }, new[] { 2, 1 });
        var batchedCtx = new AiDotNet.Inference.InferenceForwardContext(new long[] { 5002, 5003 }, new[] { pC, pD });
        var batchedLogits = model.PredictWithContext(batchedInput, batchedCtx);

        float[] rowC = VocabRow(batchedLogits, 0);
        float[] rowD = VocabRow(batchedLogits, 1);

        Assert.Equal(refA.Length, rowC.Length);
        for (int i = 0; i < refA.Length; i++)
        {
            Assert.Equal(refA[i], rowC[i], 5);
            Assert.Equal(refB[i], rowD[i], 5);
        }
    }

    private static Tensor<float> TokenTensor(int tokenId) => new(new[] { (float)tokenId }, new[] { 1, 1 });

    // Extracts the vocab row for batch index b at the LAST position from a [batch, seq, vocab] logits tensor.
    private static float[] VocabRow(Tensor<float> logits, int b)
    {
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];
        int seq = rank >= 3 ? logits.Shape[rank - 2] : 1;
        int perBatch = seq * vocab;
        int baseOffset = b * perBatch + (seq - 1) * vocab;
        var s = logits.AsSpan();
        var row = new float[vocab];
        for (int v = 0; v < vocab; v++) row[v] = s[baseOffset + v];
        return row;
    }

    // --- helpers -----------------------------------------------------------------------------------

    private static GenerationRequest<float> GreedyRequest(int[] prompt, int maxNew) => new()
    {
        PromptTokenIds = new List<int>(prompt),
        MaxNewTokens = maxNew,
        Temperature = 0f // greedy (argmax)
    };

    private static int[] BatcherGreedy(ServableModelWrapper<float> wrapper, int[] prompt, int maxNew)
    {
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return Array.Empty<int>();
        }

        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999 };
        using var batcher = new ContinuousBatcher<float>(config, model, cache);

        var task = batcher.GenerateAsync(GreedyRequest(prompt, maxNew));
        int guard = maxNew + prompt.Length + 8;
        while (!task.IsCompleted && guard-- > 0)
        {
            batcher.Step();
        }
        Assert.True(task.IsCompleted, "greedy sequence should complete within the step budget");
        return task.GetAwaiter().GetResult().GeneratedTokens.ToArray();
    }

    private static int[] BatcherSampled(ServableModelWrapper<float> wrapper, int[] prompt, int maxNew, float temperature, int seed)
    {
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return Array.Empty<int>();
        }

        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999 };
        using var batcher = new ContinuousBatcher<float>(config, model, cache);

        var req = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int>(prompt),
            MaxNewTokens = maxNew,
            Temperature = temperature,
            Seed = seed
        };
        var task = batcher.GenerateAsync(req);
        int guard = maxNew + prompt.Length + 8;
        while (!task.IsCompleted && guard-- > 0)
        {
            batcher.Step();
        }
        Assert.True(task.IsCompleted, "sampled sequence should complete within the step budget");
        return task.GetAwaiter().GetResult().GeneratedTokens.ToArray();
    }

    // Reference greedy directly over the incremental model + a fresh paged-cache sequence — the ground
    // truth the batcher must reproduce: one batched prefill forward over the whole prompt (per-position
    // logits, take the last), then greedy decode of maxNew tokens.
    private static int[] ModelGreedy(ServableModelWrapper<float> wrapper, int[] prompt, int maxNew)
    {
        var model = wrapper.IncrementalModel;
        var cache = wrapper.IncrementalCache;
        if (model is null || cache is null)
        {
            Assert.Fail("wrapper must expose the incremental model + paged cache");
            return Array.Empty<int>();
        }

        const long seqId = 900_000; // distinct from the batcher's (small, positive) sequence ids
        cache.AllocateSequence(seqId, 0);
        try
        {
            var promptTensor = new Tensor<float>(Array.ConvertAll(prompt, t => (float)t), new[] { 1, prompt.Length });
            var logits = model.PredictWithContext(promptTensor, new AiDotNet.Inference.InferenceForwardContext(seqId, 0));

            int pos = prompt.Length;
            var gen = new List<int>(maxNew);
            for (int s = 0; s < maxNew; s++)
            {
                int next = ArgMaxLast(logits);
                gen.Add(next);
                logits = model.PredictWithContext(
                    new Tensor<float>(new[] { (float)next }, new[] { 1, 1 }),
                    new AiDotNet.Inference.InferenceForwardContext(seqId, pos));
                pos++;
            }
            return gen.ToArray();
        }
        finally
        {
            cache.FreeSequence(seqId);
        }
    }

    // ArgMax over the LAST position of a [1, S, vocab] (or [1, vocab]) logits tensor.
    private static int ArgMaxLast(Tensor<float> logits)
    {
        int rank = logits.Shape.Length;
        int vocab = logits.Shape[rank - 1];
        int positions = 1;
        for (int d = 0; d < rank - 1; d++) positions *= logits.Shape[d];
        int baseOffset = (positions - 1) * vocab;
        var s = logits.AsSpan();
        int best = 0;
        for (int v = 1; v < vocab; v++)
        {
            if (s[baseOffset + v] > s[baseOffset + best]) best = v;
        }
        return best;
    }
}
