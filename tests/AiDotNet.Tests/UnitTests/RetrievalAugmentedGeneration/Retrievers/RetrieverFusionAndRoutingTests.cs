using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Tests for the RRF-based fusion retrievers (Hybrid/Ensemble) and the generator-driven routing and
/// self-query retrievers added in the RAG productionization work.
/// </summary>
public class RetrieverFusionAndRoutingTests
{
    private sealed class FakeRetriever : IRetriever<double>
    {

        public System.Threading.Tasks.Task<IEnumerable<Document<double>>> RetrieveAsync(string query, int topK, Dictionary<string, object>? metadataFilters = null, System.Threading.CancellationToken cancellationToken = default) { cancellationToken.ThrowIfCancellationRequested(); return System.Threading.Tasks.Task.FromResult(metadataFilters == null ? Retrieve(query, topK) : Retrieve(query, topK, metadataFilters)); }
        private readonly List<Document<double>> _docs;
        public string? LastQuery { get; private set; }
        public Dictionary<string, object>? LastFilters { get; private set; }
        public int DefaultTopK => 5;

        public FakeRetriever(params string[] docIds)
            => _docs = docIds.Select((id, i) => new Document<double>(id, id + " content")
            { RelevanceScore = 1.0 - i * 0.1, HasRelevanceScore = true }).ToList();

        public IEnumerable<Document<double>> Retrieve(string query) => Retrieve(query, 5, new Dictionary<string, object>());
        public IEnumerable<Document<double>> Retrieve(string query, int topK) => Retrieve(query, topK, new Dictionary<string, object>());
        public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            LastQuery = query;
            LastFilters = metadataFilters;
            return _docs.Take(topK).ToList();
        }
    }

    private sealed class FakeTextGen : ITextGenerator
    {
        private readonly Func<string, string> _responder;
        public FakeTextGen(Func<string, string> responder) => _responder = responder;
        public string Generate(string prompt) => _responder(prompt);
    }

    [Fact]
    public void HybridRetriever_RRF_RanksSharedDocHighest()
    {
        // "B" appears in both lists (rank 1 dense, rank 0 sparse) → RRF should float it to the top.
        var dense = new FakeRetriever("A", "B");
        var sparse = new FakeRetriever("B", "C");
        var hybrid = new HybridRetriever<double>(dense, sparse, defaultTopK: 3);

        var results = hybrid.Retrieve("q", 3, new Dictionary<string, object>()).ToList();

        Assert.Equal("B", results[0].Id);
        Assert.Equal(3, results.Count); // A, B, C all present
    }

    [Fact]
    public void EnsembleRetriever_RRF_FavorsDocInMultipleLists()
    {
        var r1 = new FakeRetriever("X", "Y");
        var r2 = new FakeRetriever("Y", "Z");
        var r3 = new FakeRetriever("Y", "W");
        var ensemble = new EnsembleRetriever<double>(new IRetriever<double>[] { r1, r2, r3 }, defaultTopK: 4);

        var results = ensemble.Retrieve("q", 4, new Dictionary<string, object>()).ToList();

        Assert.Equal("Y", results[0].Id); // in all three lists
    }

    [Fact]
    public void QueryRoutingRetriever_UsesGeneratorChoice()
    {
        var news = new FakeRetriever("news-doc");
        var code = new FakeRetriever("code-doc");
        var routes = new List<QueryRoutingRetriever<double>.Route>
        {
            new("news", "current events and articles", news),
            new("code", "source code and APIs", code),
        };
        var gen = new FakeTextGen(_ => "code");
        var router = new QueryRoutingRetriever<double>(routes, gen, defaultTopK: 3);

        var results = router.Retrieve("how do I call the API?", 3, new Dictionary<string, object>()).ToList();

        Assert.Equal("code-doc", results[0].Id);
        Assert.Equal("code", router.ChooseRoute("anything").Name);
    }

    [Fact]
    public void QueryRoutingRetriever_FallsBackToOverlapWithoutGenerator()
    {
        var news = new FakeRetriever("news-doc");
        var code = new FakeRetriever("code-doc");
        var routes = new List<QueryRoutingRetriever<double>.Route>
        {
            new("news", "current events and articles", news),
            new("code", "source code and APIs", code),
        };
        var router = new QueryRoutingRetriever<double>(routes, generator: null);

        Assert.Equal("code", router.ChooseRoute("show me the source code").Name);
    }

    [Fact]
    public void SelfQueryRetriever_ExtractsFilterAndCleansQuery()
    {
        var baseRetriever = new FakeRetriever("d1");
        var gen = new FakeTextGen(_ => "{\"query\": \"laptops\", \"filters\": {\"year\": 2020}}");
        var selfQuery = new SelfQueryRetriever<double>(baseRetriever, new[] { "year" }, gen, defaultTopK: 3);

        _ = selfQuery.Retrieve("laptops released in 2020", 3, new Dictionary<string, object>()).ToList();

        Assert.Equal("laptops", baseRetriever.LastQuery);
        Assert.True(baseRetriever.LastFilters!.ContainsKey("year"));
        Assert.Equal(2020L, baseRetriever.LastFilters!["year"]);
    }

    [Fact]
    public void SelfQueryRetriever_CallerFiltersTakePrecedence()
    {
        var baseRetriever = new FakeRetriever("d1");
        var gen = new FakeTextGen(_ => "{\"query\": \"laptops\", \"filters\": {\"year\": 2020}}");
        var selfQuery = new SelfQueryRetriever<double>(baseRetriever, new[] { "year" }, gen);

        _ = selfQuery.Retrieve("laptops in 2020", 3, new Dictionary<string, object> { ["year"] = 1999L }).ToList();

        Assert.Equal(1999L, baseRetriever.LastFilters!["year"]); // caller wins
    }
}
