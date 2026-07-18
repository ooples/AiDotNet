using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.StructuredOutput;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Algorithms;
using Microsoft.Extensions.Logging.Abstractions;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// Tests the serving-layer wiring of structured output: the OpenAI response_format -> ITokenConstraint
/// factory, and the end-to-end flow of a constraint through TextGenerationService into the batcher
/// (including stop-when-complete).
/// </summary>
public class StructuredOutputServingTests
{
    // ---- StructuredOutputFactory (response_format parsing) ----

    [Fact(Timeout = 60000)]
    public async Task Factory_ParsesResponseFormats()
    {
        await Task.Yield();
        var tok = CharacterTokenizer.CreateAscii();

        Assert.Null(StructuredOutputFactory.Build(null, tok, -1));
        Assert.Null(StructuredOutputFactory.Build(JToken.Parse(@"{""type"":""text""}"), tok, -1));

        Assert.IsType<RegexTokenConstraint>(
            StructuredOutputFactory.Build(JToken.Parse(@"{""type"":""json_object""}"), tok, -1));

        Assert.IsType<RegexTokenConstraint>(
            StructuredOutputFactory.Build(JToken.Parse(
                @"{""type"":""json_schema"",""json_schema"":{""schema"":{""type"":""object"",""properties"":{""x"":{""type"":""integer""}}}}}"), tok, -1));

        Assert.IsType<RegexTokenConstraint>(
            StructuredOutputFactory.Build(JToken.Parse(@"{""type"":""regex"",""regex"":""\\d+""}"), tok, -1));
    }

    [Fact(Timeout = 60000)]
    public async Task Factory_RejectsMalformedResponseFormat()
    {
        await Task.Yield();
        var tok = CharacterTokenizer.CreateAscii();

        Assert.Throws<ArgumentException>(() => StructuredOutputFactory.Build(JToken.Parse("42"), tok, -1));
        Assert.Throws<ArgumentException>(() =>
            StructuredOutputFactory.Build(JToken.Parse(@"{""type"":""bogus""}"), tok, -1));
        Assert.Throws<ArgumentException>(() =>
            StructuredOutputFactory.Build(JToken.Parse(@"{""type"":""regex""}"), tok, -1)); // missing regex
        Assert.Throws<ArgumentException>(() =>
            StructuredOutputFactory.Build(JToken.Parse(@"{""type"":""json_schema""}"), tok, -1)); // missing schema
    }

    // ---- end-to-end: constraint flows through the service into the batcher ----

    private sealed class OneModelRepo : IModelRepository
    {
        private readonly string _name;
        private readonly object _model;
        public OneModelRepo(string name, object model) { _name = name; _model = model; }
        public IServableModel<T>? GetModel<T>(string name)
            => name == _name && _model is IServableModel<T> m ? m : null;
        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public bool UnloadModel(string name) => throw new NotSupportedException();
        public List<ModelInfo> GetAllModelInfo() => new();
        public ModelInfo? GetModelInfo(string name) => null;
        public bool ModelExists(string name) => name == _name;
        public bool LoadModelFromRegistry<T>(string name, IServableModel<T> model, int registryVersion, string registryStage, string? sourcePath = null) => throw new NotSupportedException();
        public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null) => throw new NotSupportedException();
        public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name) => throw new NotSupportedException();
    }

    [Fact(Timeout = 120000)]
    public async Task Constraint_FlowsThroughService_AndStopsWhenComplete()
    {
        await Task.Yield();
        const int vocab = 32;
        const int eos = vocab - 1;

        // A synthetic token-level forward that ALWAYS prefers token 5 (a forbidden token).
        Tensor<float> gen(Tensor<float> input)
        {
            int seq = input.Shape[^1];
            var logits = new Tensor<float>(new[] { 1, seq, vocab });
            for (int p = 0; p < seq; p++)
                for (int i = 0; i < vocab; i++)
                    logits[new[] { 0, p, i }] = i == 5 ? 10f : 0f;
            return logits;
        }

        var wrapper = new ServableModelWrapper<float>(
            "sm", inputDimension: 1, outputDimension: vocab, predictFunc: v => v, generationForward: gen);
        var service = new TextGenerationService(new OneModelRepo("sm", wrapper), NullLogger<TextGenerationService>.Instance);

        var request = new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1, 4, 6 },
            MaxNewTokens = 10,       // deliberately larger than the constrained output
            Temperature = 0,         // greedy: the mask alone decides
            NumDraftTokens = 3,      // speculation requested; a constraint must disable it
            EosTokenId = eos,
            Constraint = TokenFsmConstraint.FromSequence(new[] { 3, 7 }, eos)
        };

        var resp = service.Generate("sm", NumericType.Float, request);

        Assert.Null(resp.Error);
        // Stops the instant the constraint completes ([3,7] then done) — not at MaxNewTokens.
        Assert.Equal(2, resp.NumGenerated);
        Assert.Equal(new[] { 3, 7 }, resp.GeneratedTokens);
    }

    [Fact(Timeout = 120000)]
    public async Task LogProbs_FlowThroughService()
    {
        await Task.Yield();
        const int vocab = 16;

        Tensor<float> gen(Tensor<float> input)
        {
            int seq = input.Shape[^1];
            var logits = new Tensor<float>(new[] { 1, seq, vocab });
            for (int p = 0; p < seq; p++)
            {
                logits[new[] { 0, p, 5 }] = 10f;
                logits[new[] { 0, p, 6 }] = 9f;
            }
            return logits;
        }

        var wrapper = new ServableModelWrapper<float>(
            "sm", inputDimension: 1, outputDimension: vocab, predictFunc: v => v, generationForward: gen);
        var service = new TextGenerationService(new OneModelRepo("sm", wrapper), NullLogger<TextGenerationService>.Instance);

        var resp = service.Generate("sm", NumericType.Float, new SpeculativeDecodingRequest
        {
            InputTokens = new[] { 1, 2 },
            MaxNewTokens = 3,
            Temperature = 0,
            EosTokenId = 999,
            Logprobs = true,
            TopLogprobs = 2
        });

        Assert.Null(resp.Error);
        Assert.NotNull(resp.LogProbs);
        Assert.Equal(resp.NumGenerated, resp.LogProbs!.Count);
        Assert.All(resp.LogProbs, p =>
        {
            Assert.Equal(5, p.TokenId);            // greedy picks the top-logit token
            Assert.True(p.LogProb < 0f);
            Assert.Equal(2, p.TopLogProbs.Count);  // top_logprobs=2
            Assert.Equal(5, p.TopLogProbs[0].TokenId);
            Assert.Equal(6, p.TopLogProbs[1].TokenId);
        });
    }
}
