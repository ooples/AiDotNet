using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.QueryExpansion;
using AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration;

/// <summary>
/// Verifies the previously-stubbed LLM-powered RAG components now actually call an injected
/// <see cref="ITextGenerator"/> (and fall back to their offline heuristic when none is supplied),
/// rather than running lexical templates with unused credentials.
/// </summary>
public class DeStubbedLlmComponentsTests
{
    /// <summary>Records prompts and returns a scripted response — a stand-in for a real LLM.</summary>
    private sealed class FakeTextGen : ITextGenerator
    {
        private readonly Func<string, string> _responder;
        public List<string> Prompts { get; } = new();
        public FakeTextGen(Func<string, string> responder) => _responder = responder;
        public string Generate(string prompt)
        {
            Prompts.Add(prompt);
            return _responder(prompt);
        }
    }

    private sealed class RecordingRetriever : IRetriever<double>
    {
        public List<string> Queries { get; } = new();
        public int DefaultTopK => 5;
        public IEnumerable<Document<double>> Retrieve(string query) => Retrieve(query, 5, new Dictionary<string, object>());
        public IEnumerable<Document<double>> Retrieve(string query, int topK) => Retrieve(query, topK, new Dictionary<string, object>());
        public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            Queries.Add(query);
            return new[] { new Document<double>("d-" + Queries.Count, query) { RelevanceScore = 1.0, HasRelevanceScore = true } };
        }
    }

    [Fact]
    public void HyDE_WithGenerator_UsesLlmPassages()
    {
        var gen = new FakeTextGen(_ => "A hypothetical answer passage.");
        var hyde = new HyDEQueryExpansion(gen, numHypotheticals: 2);

        var expansions = hyde.ExpandQuery("what is X?");

        Assert.Equal("what is X?", expansions[0]);
        Assert.Contains("A hypothetical answer passage.", expansions);
        Assert.Equal(2, gen.Prompts.Count);
        Assert.Contains("Passage:", gen.Prompts[0]); // real HyDE prompt, not a template
    }

    [Fact]
    public void HyDE_WithoutGenerator_FallsBackToTemplate()
    {
        var hyde = new HyDEQueryExpansion();
        var expansions = hyde.ExpandQuery("what is X?");
        Assert.True(expansions.Count >= 2);
        Assert.Equal("what is X?", expansions[0]);
    }

    [Fact]
    public void LLMQueryExpansion_WithGenerator_ParsesLlmLines()
    {
        var gen = new FakeTextGen(_ => "1. how does X work\n- explain X\nX overview");
        var exp = new LLMQueryExpansion(numExpansions: 3, generator: gen);

        var result = exp.ExpandQuery("X");

        Assert.Equal("X", result[0]);
        Assert.Contains("how does X work", result); // numbering stripped
        Assert.Contains("explain X", result);       // bullet stripped
        Assert.Contains("X overview", result);
        Assert.Single(gen.Prompts);
    }

    [Fact]
    public void LLMBasedReranker_WithGenerator_OrdersByLlmScore()
    {
        // Score the doc mentioning "dogs" high, the other low. The query deliberately omits that keyword
        // so the fake keys only on the document content, not the query echoed into the prompt.
        var gen = new FakeTextGen(p => p.Contains("dogs") ? "9" : "1");
        var reranker = new LLMBasedReranker<double>(generator: gen);
        var docs = new[]
        {
            new Document<double>("low", "cats sleep a lot"),
            new Document<double>("high", "dogs bark loudly"),
        };

        var ranked = reranker.Rerank("which animal is loud", docs).ToList();

        Assert.Equal("high", ranked[0].Id);
        Assert.Equal("low", ranked[1].Id);
        Assert.True(gen.Prompts.Count >= 2);
    }

    [Fact]
    public void LLMContextCompressor_WithGenerator_ReturnsLlmExcerpt()
    {
        var gen = new FakeTextGen(_ => "the relevant excerpt");
        var comp = new LLMContextCompressor<double>(generator: gen);

        var compressed = comp.CompressText("q", "a long document with lots of text and one relevant bit");

        Assert.Equal("the relevant excerpt", compressed);
        Assert.Single(gen.Prompts);
    }

    [Fact]
    public void MultiQueryRetriever_WithGenerator_RetrievesWithDistinctVariations()
    {
        var gen = new FakeTextGen(_ => "rewording one\nrewording two");
        var baseRetriever = new RecordingRetriever();
        var mq = new MultiQueryRetriever<double>(baseRetriever, numQueries: 3, defaultTopK: 5, generator: gen);

        _ = mq.Retrieve("original query", 5, new Dictionary<string, object>()).ToList();

        Assert.Contains("original query", baseRetriever.Queries);
        Assert.Contains("rewording one", baseRetriever.Queries);
        Assert.True(baseRetriever.Queries.Distinct().Count() >= 2, "expected multiple distinct query variations");
    }
}
