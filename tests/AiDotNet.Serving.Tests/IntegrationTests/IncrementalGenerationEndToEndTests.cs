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

    private static (TextGenerationService Service, ServableModelWrapper<float> Wrapper) BuildService()
    {
        var model = BuildLm();
        var wrapper = new ServableModelWrapper<float>(
            "lm", model, inputShape: new[] { 1 }, generationForward: model.Predict);
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
