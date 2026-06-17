// Copyright (c) AiDotNet. All rights reserved.
// #99 Stage 1b end-to-end: a real (embedding-based) transformer LM served through
// ServableModelWrapper drives KV-cached incremental decode via per-request sessions. Proves the
// wrapper builds the incremental path, generation produces in-range tokens, and concurrent requests
// to the SAME model are isolated (the user's core requirement) — not just layer-level.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

public class IncrementalGenerationEndToEndTests
{
    private const int Vocab = 12;
    private const int EmbDim = 8;
    private const int Heads = 2;

    private static NeuralNetwork<float> BuildLm()
    {
        var layers = new List<AiDotNet.Interfaces.ILayer<float>>
        {
            new EmbeddingLayer<float>(Vocab, EmbDim),
            new MultiHeadAttentionLayer<float>(Heads, EmbDim / Heads,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new FlattenLayer<float>(),
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

    private static (TextGenerationService Service, ServableModelWrapper<float> Wrapper) BuildService(bool quantize = false)
    {
        var model = BuildLm();
        var wrapper = new ServableModelWrapper<float>(
            "lm", model, inputShape: new[] { 1 }, generationForward: model.Predict,
            quantizeIncrementalWeights: quantize);
        var repo = new OneModelRepo("lm", wrapper);
        return (new TextGenerationService(repo, NullLogger<TextGenerationService>.Instance), wrapper);
    }

    private static SpeculativeDecodingRequest Req(int[] tokens) => new()
    {
        InputTokens = tokens,
        MaxNewTokens = 5,
        EosTokenId = 999, // out of range -> run to the token limit
        NumDraftTokens = 2
    };

    [Fact(Timeout = 120000)]
    public async Task ServedLm_BuildsIncrementalPath_AndGenerates()
    {
        await Task.Yield();
        var (service, wrapper) = BuildService();

        Assert.True(wrapper.SupportsIncrementalGeneration,
            "wrapper should build the KV-cached incremental path for an optimizable transformer LM");

        var resp = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3 }));

        Assert.Null(resp.Error);
        Assert.Equal(5, resp.NumGenerated);
        Assert.Equal(resp.NumGenerated, resp.GeneratedTokens.Length);
        Assert.All(resp.GeneratedTokens, t => Assert.InRange(t, 0, Vocab - 1));
        Assert.Equal(new[] { 1, 2, 3 }, resp.AllTokens[..3]);
    }

    [Fact(Timeout = 120000)]
    public async Task QuantizedIncrementalServing_BuildsAndGenerates()
    {
        await Task.Yield();
        var (service, wrapper) = BuildService(quantize: true);

        Assert.True(wrapper.SupportsIncrementalGeneration,
            "wrapper should build a quantized KV-cached incremental path");

        var resp = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3 }));

        Assert.Null(resp.Error);
        Assert.Equal(5, resp.NumGenerated);
        Assert.All(resp.GeneratedTokens, t => Assert.InRange(t, 0, Vocab - 1));
    }

    [Fact(Timeout = 120000)]
    public async Task ConcurrentRequests_SameModel_AreIsolated()
    {
        await Task.Yield();
        var (service, _) = BuildService();

        var reqA = Req(new[] { 1, 2, 3 });
        var reqB = Req(new[] { 7, 8 });

        // Sequential references.
        var refA = service.Generate("lm", NumericType.Float, reqA).GeneratedTokens;
        var refB = service.Generate("lm", NumericType.Float, reqB).GeneratedTokens;

        // Concurrent — each request opens its own session (distinct cache sequence id) over the
        // shared optimized model + paged cache.
        int[]? outA = null, outB = null;
        await Task.WhenAll(
            Task.Run(() => outA = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3 })).GeneratedTokens),
            Task.Run(() => outB = service.Generate("lm", NumericType.Float, Req(new[] { 7, 8 })).GeneratedTokens));

        Assert.Equal(refA, outA);
        Assert.Equal(refB, outB);
    }

    [Fact(Timeout = 120000)]
    public async Task PrefixSharing_ReusesRegisteredPrefix()
    {
        await Task.Yield();
        var (service, wrapper) = BuildService();

        // First request registers its prompt [1,2,3,4] as a reusable prefix.
        _ = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3, 4 }));

        // A later prompt that extends it forks the cached prefix (copy-on-write): 4 tokens already
        // cached, only [5,6] need forwarding.
        using var session = wrapper.BeginGeneration(new[] { 1, 2, 3, 4, 5, 6 });
        Assert.Equal(4, session.CachedPromptTokens);
    }

    [Fact(Timeout = 120000)]
    public async Task PrefixSharing_IsTransparent_SameResultAsNoSharing()
    {
        await Task.Yield();
        var extended = new[] { 1, 2, 3, 4, 5, 6 };

        // Use ONE wrapper/model so the comparison isolates prefix sharing (not weight differences
        // between two separately-built model instances).
        var (service, wrapper) = BuildService();

        // Register prefix [1,2,3,4], then decode `extended` two ways on the SAME model:
        //  - forked: BeginGeneration(extended) reuses the cached [1,2,3,4] prefix (CachedPromptTokens=4)
        //  - fresh:  BeginGeneration() with no prefix
        _ = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3, 4 }));

        var forked = GreedyDecode(wrapper.BeginGeneration(extended), extended, 5);
        var fresh = GreedyDecode(wrapper.BeginGeneration(), extended, 5);

        Assert.True(forked.Length > 0);
        Assert.Equal(fresh, forked); // prefix sharing must not change the output
    }

    // Per-position LM (no Flatten): Embedding -> MHA -> Dense(vocab) maps [1,S] -> [1,S,vocab], so it
    // accepts a multi-token forward (batched prefill / speculative verification).
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
        var p = model.GetParameters();
        var det = new float[p.Length];
        for (int i = 0; i < det.Length; i++) det[i] = ((i % 17) - 8) / 16.0f;
        model.UpdateParameters(new Vector<float>(det));
        return model;
    }

    [Fact(Timeout = 120000)]
    public async Task SequenceCollapsingLm_DoesNotAdvertiseBatchedPrefill()
    {
        await Task.Yield();
        // BuildLm is Embedding -> MHA -> Flatten -> Dense: the Flatten collapses the sequence into one
        // fixed-width row, so a multi-token forward does NOT produce per-position logits (and would make
        // a shape-dependent Dense re-fit its weights to the wider flattened input). The batched-prefill
        // probe must reject it (require the position dimension to be preserved) so it uses per-token
        // prefill. A false positive here fed [1,n] then [1,1] forwards of differing flattened widths and
        // caused nondeterministic decode (the Dense layer re-initialized on each width change).
        var model = BuildLm();
        var wrapper = new ServableModelWrapper<float>("lm", model, inputShape: new[] { 1 }, generationForward: model.Predict);
        Assert.False(wrapper.SupportsBatchedPrefill,
            "a Flatten+Dense (sequence-collapsing) LM must not advertise batched prefill");
    }

    [Fact(Timeout = 120000)]
    public async Task RepeatedRequests_SamePrompt_AreDeterministic()
    {
        await Task.Yield();
        // Generations of the same prompt over the SAME served model (shared paged cache + prefix
        // registry) must be byte-for-byte identical across repeated requests — no state leaks between
        // requests. This is the sequential analogue of ConcurrentRequests_SameModel_AreIsolated and
        // directly guards the per-token vs batched-prefill width-mismatch regression.
        var (service, _) = BuildService();

        var first = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3 })).GeneratedTokens;
        _ = service.Generate("lm", NumericType.Float, Req(new[] { 7, 8 })).GeneratedTokens;
        var third = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3 })).GeneratedTokens;
        var fourth = service.Generate("lm", NumericType.Float, Req(new[] { 1, 2, 3 })).GeneratedTokens;

        Assert.Equal(first, third);
        Assert.Equal(first, fourth);
    }

    [Fact(Timeout = 120000)]
    public async Task BatchedPrefill_IsTransparent_SameResultAsPerToken()
    {
        await Task.Yield();
        var model = BuildPerPositionLm();
        var wrapper = new ServableModelWrapper<float>("lm", model, inputShape: new[] { 1 }, generationForward: model.Predict);
        Assert.True(wrapper.SupportsBatchedPrefill, "per-position LM should accept a multi-token forward");

        var prompt = new[] { 1, 2, 3, 4 };

        // Batched prefill: one multi-token forward over the whole prompt.
        var batched = GreedyDecodeBatchedPrefill(wrapper.BeginGeneration(), prompt, 5);
        // Per-token prefill: forward each prompt token.
        var perToken = GreedyDecode(wrapper.BeginGeneration(), prompt, 5);

        Assert.True(batched.Length > 0);
        Assert.Equal(perToken, batched);
    }

    [Fact(Timeout = 120000)]
    public async Task FromModel_WithGenerationEnabled_YieldsGenerativeWrapper()
    {
        await Task.Yield();
        var model = BuildPerPositionLm();

        // Auto-detect path (the production loader path: ModelLoader.Load -> FromModel). Opting in to
        // text generation must wire the token-to-logits forward + the KV-cached incremental path.
        var generative = ServableModelWrapper<float>.FromModel(
            "lm", model, enableBatching: true, enableSpeculativeDecoding: false,
            enableTextGeneration: true);

        Assert.True(generative.SupportsGeneration,
            "enableTextGeneration:true should advertise token-level generation");
        Assert.True(generative.SupportsIncrementalGeneration,
            "enableTextGeneration:true should build the KV-cached incremental path");
    }

    [Fact(Timeout = 120000)]
    public async Task FromModel_GenerationDisabledByDefault()
    {
        await Task.Yield();
        var model = BuildPerPositionLm();

        // Default: a tensor model is NOT assumed generative (e.g. diffusion models have no
        // next-token-logits semantics), so generation stays off unless explicitly enabled.
        var nonGenerative = ServableModelWrapper<float>.FromModel("lm", model);

        Assert.False(nonGenerative.SupportsGeneration,
            "tensor models must not advertise generation unless enableTextGeneration is set");
        Assert.False(nonGenerative.SupportsIncrementalGeneration);
    }

    [Fact(Timeout = 120000)]
    public async Task FromModel_QuantizedGeneration_BuildsIncrementalPath()
    {
        await Task.Yield();
        var model = BuildPerPositionLm();

        var generative = ServableModelWrapper<float>.FromModel(
            "lm", model, enableBatching: true, enableSpeculativeDecoding: false,
            enableTextGeneration: true, quantizeKvCacheWeights: true);

        Assert.True(generative.SupportsIncrementalGeneration,
            "quantized generation config should still build the incremental path");
    }

    private static int[] GreedyDecodeBatchedPrefill(IGenerationSession<float> session, int[] prompt, int maxNew)
    {
        using (session)
        {
            // Single multi-token prefill forward over the whole prompt (per-position logits; last one).
            var promptTensor = new Tensor<float>(System.Array.ConvertAll(prompt, t => (float)t), new[] { 1, prompt.Length });
            var logits = session.Forward(promptTensor);

            var gen = new List<int>(maxNew);
            for (int s = 0; s < maxNew; s++)
            {
                int next = ArgMaxLast(logits);
                gen.Add(next);
                logits = session.Forward(TokenTensor(next));
            }
            return gen.ToArray();
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

    /// <summary>Drives a session: prefill the remaining prompt then greedily decode maxNew tokens.</summary>
    private static int[] GreedyDecode(IGenerationSession<float> session, int[] prompt, int maxNew)
    {
        using (session)
        {
            int start = session.CachedPromptTokens;
            var logits = session.Forward(TokenTensor(prompt[start]));
            for (int i = start + 1; i < prompt.Length; i++)
            {
                logits = session.Forward(TokenTensor(prompt[i]));
            }

            var gen = new List<int>(maxNew);
            for (int s = 0; s < maxNew; s++)
            {
                int next = ArgMax(logits);
                gen.Add(next);
                logits = session.Forward(TokenTensor(next));
            }
            return gen.ToArray();
        }
    }

    private static Tensor<float> TokenTensor(int tokenId)
        => new(new[] { (float)tokenId }, new[] { 1, 1 });

    private static int ArgMax(Tensor<float> logits)
    {
        var s = logits.AsSpan();
        int best = 0;
        for (int i = 1; i < s.Length; i++)
        {
            if (s[i] > s[best]) best = i;
        }
        return best;
    }

    private sealed class OneModelRepo : IModelRepository
    {
        private readonly string _name;
        private readonly object _model;
        public OneModelRepo(string name, object model) { _name = name; _model = model; }

        public IServableModel<T>? GetModel<T>(string name)
            => name == _name && _model is IServableModel<T> m ? m : null;

        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public bool UnloadModel(string name) => throw new NotSupportedException();
        public List<ModelInfo> GetAllModelInfo() => [];
        public ModelInfo? GetModelInfo(string name) => null;
        public bool ModelExists(string name) => name == _name;
        public bool LoadModelFromRegistry<T>(string name, IServableModel<T> model, int registryVersion, string registryStage, string? sourcePath = null) => throw new NotSupportedException();
        public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name) => throw new NotSupportedException();
    }
}
