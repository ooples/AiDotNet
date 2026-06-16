// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// Proves the serving <see cref="ITextGenerationService"/> actually drives the (previously orphaned)
/// continuous-batching engine end to end: a generative model wired through <see cref="ServableModelWrapper{T}"/>
/// produces real autoregressive tokens, EOS stops generation, and non-generative models are rejected
/// with a clear error instead of silently failing.
/// </summary>
public class TextGenerationServiceTests
{
    private const int Vocab = 10;

    /// <summary>A deterministic token-level forward that always peaks at <paramref name="peakToken"/>.</summary>
    private static Func<Tensor<float>, Tensor<float>> PeakedForward(int peakToken)
        => input =>
        {
            int seqLen = input.Shape[input.Shape.Length - 1];
            var logits = new Tensor<float>(new[] { 1, seqLen, Vocab });
            for (int p = 0; p < seqLen; p++)
            {
                // A large peak makes softmax numerically 1.0 at peakToken, so sampling is deterministic.
                logits[new[] { 0, p, peakToken }] = 100f;
            }
            return logits;
        };

    private static ServableModelWrapper<float> GenerativeModel(string name, int peakToken)
        => new ServableModelWrapper<float>(
            name,
            inputDimension: 1,
            outputDimension: Vocab,
            predictFunc: v => v,
            generationForward: PeakedForward(peakToken));

    private static ServableModelWrapper<float> NonGenerativeModel(string name)
        => new ServableModelWrapper<float>(
            name,
            inputDimension: 1,
            outputDimension: 1,
            predictFunc: v => v);

    private static TextGenerationService BuildService(string name, IServableModel<float> model)
        => new TextGenerationService(
            new SingleModelRepository(name, model),
            NullLogger<TextGenerationService>.Instance);

    [Fact(Timeout = 120000)]
    public async Task Generate_ProducesAutoregressiveTokens_UpToMaxNewTokens()
    {
        await Task.Yield();
        var service = BuildService("lm", GenerativeModel("lm", peakToken: 7));

        var request = new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1, 5, 9 },
            MaxNewTokens = 4,
            EosTokenId = 2, // model never emits 2, so generation runs to the token limit
            NumDraftTokens = 3,
            Temperature = 1.0
        };

        var response = service.Generate("lm", NumericType.Float, request);

        Assert.Null(response.Error);
        Assert.Equal(4, response.NumGenerated);
        Assert.Equal(response.NumGenerated, response.GeneratedTokens.Length);
        // Every generated token is the model's peaked prediction.
        Assert.All(response.GeneratedTokens, t => Assert.Equal(7, t));
        // AllTokens is the prompt followed by the generated tokens.
        Assert.Equal(new[] { 1, 5, 9, 7, 7, 7, 7 }, response.AllTokens);
    }

    [Fact(Timeout = 120000)]
    public async Task Generate_StopsAtEosToken()
    {
        await Task.Yield();
        // Model always predicts the EOS token, so generation must stop after the first token.
        var service = BuildService("lm", GenerativeModel("lm", peakToken: 2));

        var request = new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1 },
            MaxNewTokens = 20,
            EosTokenId = 2,
            NumDraftTokens = 3,
            Temperature = 1.0
        };

        var response = service.Generate("lm", NumericType.Float, request);

        Assert.Null(response.Error);
        Assert.Equal(1, response.NumGenerated);
        Assert.Equal(new[] { 2 }, response.GeneratedTokens);
        Assert.Equal(new[] { 1, 2 }, response.AllTokens);
    }

    [Fact(Timeout = 120000)]
    public async Task Generate_NonGenerativeModel_ReturnsError()
    {
        await Task.Yield();
        var service = BuildService("reg", NonGenerativeModel("reg"));

        var request = new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1, 2, 3 },
            MaxNewTokens = 5
        };

        var response = service.Generate("reg", NumericType.Float, request);

        Assert.NotNull(response.Error);
        Assert.Contains("does not support text generation", response.Error);
        Assert.Empty(response.GeneratedTokens);
    }

    [Fact(Timeout = 120000)]
    public async Task Generate_MissingModel_ReturnsError()
    {
        await Task.Yield();
        var service = BuildService("lm", GenerativeModel("lm", peakToken: 7));

        var request = new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1 },
            MaxNewTokens = 3
        };

        var response = service.Generate("does-not-exist", NumericType.Float, request);

        Assert.NotNull(response.Error);
        Assert.Empty(response.GeneratedTokens);
    }

    /// <summary>Repository that serves exactly one model by name (only GetModel is needed by the service).</summary>
    private sealed class SingleModelRepository : IModelRepository
    {
        private readonly string _name;
        private readonly object _model;

        public SingleModelRepository(string name, object model)
        {
            _name = name;
            _model = model;
        }

        public IServableModel<T>? GetModel<T>(string name)
            => name == _name && _model is IServableModel<T> typed ? typed : null;

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
