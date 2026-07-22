using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration;

/// <summary>
/// Focused tests for the async + cancellation surface added across the RAG pipeline (task #21).
/// Each test asserts that the *Async methods produce the same result as their synchronous
/// counterparts and that a pre-cancelled <see cref="CancellationToken"/> causes an
/// <see cref="OperationCanceledException"/> to be thrown.
/// </summary>
public class RagAsyncCancellationTests
{
    private static Document<double> Doc(string id, string content)
        => new Document<double>(id, content, new Dictionary<string, object>());

    // ---- Fakes built on the RAG base classes so we exercise the real async plumbing ----

    private sealed class FakeRetriever : RetrieverBase<double>
    {
        public FakeRetriever() : base(5) { }
        protected override IEnumerable<Document<double>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
            => new List<Document<double>> { Doc("r1", "retrieved: " + query) }.Take(topK);
    }

    private sealed class FakeReranker : RerankerBase<double>
    {
        public override bool ModifiesScores => true;
        protected override IEnumerable<Document<double>> RerankCore(string query, IList<Document<double>> documents)
            => documents.Reverse();
    }

    private sealed class FakeCompressor : ContextCompressorBase<double>
    {
        protected override List<Document<double>> CompressCore(List<Document<double>> documents, string query, Dictionary<string, object>? options = null)
            => documents.Select(d => Doc(d.Id, d.Content.Substring(0, Math.Min(3, d.Content.Length)))).ToList();
    }

    private sealed class FakeGenerator : GeneratorBase<double>
    {
        public FakeGenerator() : base(2048, 512) { }
        protected override string GenerateCore(string prompt) => "answer[1]";
    }

    private sealed class FakeStore : DocumentStoreBase<double>
    {
        private readonly Dictionary<string, VectorDocument<double>> _docs = new();
        public override int DocumentCount => _docs.Count;
        public override int VectorDimension => 2;
        protected override void AddCore(VectorDocument<double> vectorDocument) => _docs[vectorDocument.Document.Id] = vectorDocument;
        protected override IEnumerable<Document<double>> GetSimilarCore(Vector<double> queryVector, int topK, Dictionary<string, object> metadataFilters)
            => _docs.Values.Select(v => v.Document).Take(topK);
        protected override Document<double>? GetByIdCore(string documentId) => _docs.TryGetValue(documentId, out var v) ? v.Document : null;
        protected override bool RemoveCore(string documentId) => _docs.Remove(documentId);
        public override void Clear() => _docs.Clear();
        protected override IEnumerable<Document<double>> GetAllCore() => _docs.Values.Select(v => v.Document).ToList();
    }

    private static VectorDocument<double> VDoc(string id)
        => new VectorDocument<double>(Doc(id, "c-" + id), new Vector<double>(new[] { 1.0, 0.0 }));

    private static CancellationToken Cancelled()
    {
        var cts = new CancellationTokenSource();
        cts.Cancel();
        return cts.Token;
    }

    // ---------------------------- Retriever ----------------------------

    [Fact]
    public async Task RetrieveAsync_ReturnsExpectedResults()
    {
        var retriever = new FakeRetriever();
        var results = (await retriever.RetrieveAsync("hello", 5)).ToList();
        Assert.Single(results);
        Assert.Equal("retrieved: hello", results[0].Content);
    }

    [Fact]
    public async Task RetrieveAsync_PreCancelledToken_Throws()
    {
        var retriever = new FakeRetriever();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => retriever.RetrieveAsync("hello", 5, null, Cancelled()));
    }

    // ---------------------------- Reranker ----------------------------

    [Fact]
    public async Task RerankAsync_ReturnsReorderedResults()
    {
        var reranker = new FakeReranker();
        var input = new List<Document<double>> { Doc("a", "a"), Doc("b", "b") };
        var results = (await reranker.RerankAsync("q", input)).ToList();
        Assert.Equal("b", results[0].Id);
        Assert.Equal("a", results[1].Id);
    }

    [Fact]
    public async Task RerankAsync_PreCancelledToken_Throws()
    {
        var reranker = new FakeReranker();
        var input = new List<Document<double>> { Doc("a", "a"), Doc("b", "b") };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => reranker.RerankAsync("q", input, Cancelled()));
    }

    // ------------------------- Context compressor -------------------------

    [Fact]
    public async Task CompressAsync_ReturnsCompressedResults()
    {
        var compressor = new FakeCompressor();
        var input = new List<Document<double>> { Doc("a", "abcdef") };
        var results = await compressor.CompressAsync(input, "q");
        Assert.Equal("abc", results[0].Content);
    }

    [Fact]
    public async Task CompressAsync_PreCancelledToken_Throws()
    {
        var compressor = new FakeCompressor();
        var input = new List<Document<double>> { Doc("a", "abcdef") };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => compressor.CompressAsync(input, "q", null, Cancelled()));
    }

    // ---------------------------- Generator ----------------------------

    [Fact]
    public async Task GenerateAsync_ReturnsText()
    {
        var generator = new FakeGenerator();
        Assert.Equal("answer[1]", await generator.GenerateAsync("prompt"));
    }

    [Fact]
    public async Task GenerateGroundedAsync_ReturnsGroundedAnswer()
    {
        var generator = new FakeGenerator();
        var answer = await generator.GenerateGroundedAsync("q", new[] { Doc("s1", "src") });
        Assert.Equal("answer[1]", answer.Answer);
        Assert.Contains("s1", answer.Citations);
    }

    [Fact]
    public async Task GenerateAsync_PreCancelledToken_Throws()
    {
        var generator = new FakeGenerator();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => generator.GenerateAsync("prompt", Cancelled()));
    }

    [Fact]
    public async Task GenerateGroundedAsync_PreCancelledToken_Throws()
    {
        var generator = new FakeGenerator();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => generator.GenerateGroundedAsync("q", new[] { Doc("s1", "src") }, Cancelled()));
    }

    // ------------------------- Document store -------------------------

    [Fact]
    public async Task DocumentStore_AsyncRoundTrip_Works()
    {
        var store = new FakeStore();
        await store.AddAsync(VDoc("d1"));
        await store.AddBatchAsync(new[] { VDoc("d2"), VDoc("d3") });

        Assert.Equal(3, store.DocumentCount);

        var byId = await store.GetByIdAsync("d2");
        Assert.NotNull(byId);
        Assert.Equal("d2", byId!.Id);

        var all = (await store.GetAllAsync()).ToList();
        Assert.Equal(3, all.Count);

        var similar = (await store.GetSimilarAsync(new Vector<double>(new[] { 1.0, 0.0 }), 2)).ToList();
        Assert.Equal(2, similar.Count);

        Assert.True(await store.RemoveAsync("d1"));
        await store.ClearAsync();
        Assert.Equal(0, store.DocumentCount);
    }

    [Fact]
    public async Task DocumentStore_PreCancelledToken_Throws()
    {
        var store = new FakeStore();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => store.AddAsync(VDoc("d1"), Cancelled()));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => store.GetSimilarAsync(new Vector<double>(new[] { 1.0, 0.0 }), 2, Cancelled()));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => store.GetAllAsync(Cancelled()));
    }
}
