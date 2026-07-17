// Copyright (c) AiDotNet. All rights reserved.
// Phase 1c: prove the continuous-batching engine's PAGED incremental path produces byte-for-byte
// identical greedy output to the proven GenerationSession path, driven over the SAME optimized model
// and SAME shared PagedKVCache. This de-risks routing the live serving path (TextGenerationService)
// through ONE shared batcher: the batcher must match the session before it replaces it.

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
    public async Task PagedBatcher_GreedyOutput_MatchesSessionPath()
    {
        await Task.Yield();
        var wrapper = BuildWrapper();
        Assert.True(wrapper.SupportsIncrementalGeneration);
        Assert.True(wrapper.SupportsBatchedPrefill);

        var prompt = new[] { 1, 2, 3 };
        const int maxNew = 8;

        // Reference: the proven GenerationSession path — one batched prefill forward over the whole
        // prompt, then greedy (argmax) decode of maxNew tokens.
        int[] sessionTokens = SessionGreedy(wrapper, prompt, maxNew);

        // Under test: the continuous-batching engine driving the SAME optimized model + SAME paged cache
        // via its paged incremental path (RunPagedPrefill/RunPagedDecodeStep + SampleFromLogits greedy).
        int[] batcherTokens = BatcherGreedy(wrapper, prompt, maxNew);

        Assert.Equal(maxNew, sessionTokens.Length);
        Assert.Equal(sessionTokens, batcherTokens);
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

        int[] refA = SessionGreedy(wrapper, promptA, maxNew);
        int[] refB = SessionGreedy(wrapper, promptB, maxNew);

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

    // Reference greedy over the GenerationSession path: one batched prefill forward over the whole
    // prompt (per-position logits, take the last), then greedy decode of maxNew tokens.
    private static int[] SessionGreedy(ServableModelWrapper<float> wrapper, int[] prompt, int maxNew)
    {
        using var session = wrapper.BeginGeneration();
        var promptTensor = new Tensor<float>(Array.ConvertAll(prompt, t => (float)t), new[] { 1, prompt.Length });
        var logits = session.Forward(promptTensor);

        var gen = new List<int>(maxNew);
        for (int s = 0; s < maxNew; s++)
        {
            int next = ArgMaxLast(logits);
            gen.Add(next);
            logits = session.Forward(new Tensor<float>(new[] { (float)next }, new[] { 1, 1 }));
        }
        return gen.ToArray();
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
