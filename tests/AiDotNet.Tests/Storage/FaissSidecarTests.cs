#if NET10_0_OR_GREATER
using System.Collections.Generic;
using System.Linq;

using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.Storage;

/// <summary>
/// Pure, native-free unit tests for the FAISS sidecar (id mapping + persistence) and the
/// over-fetch/filter retrieval planner from the opt-in <c>AiDotNet.Storage.Faiss</c> package.
/// These exercise the managed logic that maps FAISS's int64 ids back to real documents and the
/// metadata-filter over-fetch selection — no native FAISS library is loaded, so they run on any
/// host where the net10 build is available.
/// </summary>
public class FaissSidecarTests
{
    [Fact]
    public void Upsert_AssignsMonotonicIdsAndTracksCount()
    {
        var sidecar = new FaissSidecar();

        var id0 = sidecar.Upsert("a", "content-a", null, new float[] { 1, 0 }, out var replaced0);
        var id1 = sidecar.Upsert("b", "content-b", null, new float[] { 0, 1 }, out var replaced1);

        Assert.Equal(0, id0);
        Assert.Equal(1, id1);
        Assert.Null(replaced0);
        Assert.Null(replaced1);
        Assert.Equal(2, sidecar.Count);
        Assert.Equal(2, sidecar.NextId);
    }

    [Fact]
    public void Upsert_SameDocumentId_ReplacesAndReportsOldId()
    {
        var sidecar = new FaissSidecar();
        var first = sidecar.Upsert("dup", "v1", null, new float[] { 1 }, out _);

        var second = sidecar.Upsert("dup", "v2", null, new float[] { 2 }, out var replaced);

        Assert.Equal(first, replaced);            // old FAISS id reported for eviction
        Assert.NotEqual(first, second);           // a fresh id is always allocated
        Assert.Equal(1, sidecar.Count);           // still a single live document
        Assert.True(sidecar.TryGetByDocumentId("dup", out var entry));
        Assert.Equal("v2", entry.Content);
        Assert.Equal(second, entry.FaissId);
        Assert.False(sidecar.TryGetByFaissId(first, out _)); // stale id gone
    }

    [Fact]
    public void RemoveByDocumentId_FreesEntry_AndNeverReusesId()
    {
        var sidecar = new FaissSidecar();
        var id = sidecar.Upsert("x", "c", null, new float[] { 1 }, out _);

        Assert.True(sidecar.RemoveByDocumentId("x", out var removedId));
        Assert.Equal(id, removedId);
        Assert.Equal(0, sidecar.Count);
        Assert.False(sidecar.TryGetByDocumentId("x", out _));
        Assert.False(sidecar.RemoveByDocumentId("x", out _)); // idempotent

        // Re-adding the same document id must not reuse the freed int64 id.
        var newId = sidecar.Upsert("x", "c2", null, new float[] { 1 }, out _);
        Assert.NotEqual(id, newId);
    }

    [Fact]
    public void JsonRoundTrip_PreservesEntriesAndIdCounter()
    {
        var sidecar = new FaissSidecar();
        sidecar.Upsert("a", "content-a", new Dictionary<string, object> { ["year"] = 2024, ["cat"] = "sci" }, new float[] { 1, 2, 3 }, out _);
        sidecar.Upsert("b", "content-b", new Dictionary<string, object> { ["cat"] = "hist" }, new float[] { 4, 5, 6 }, out _);
        sidecar.RemoveByDocumentId("a", out _);
        var expectedNextId = sidecar.NextId;

        var restored = FaissSidecar.FromJson(sidecar.ToJson());

        Assert.Equal(1, restored.Count);
        Assert.True(expectedNextId <= restored.NextId); // never hands back a used id
        Assert.True(restored.TryGetByDocumentId("b", out var b));
        Assert.Equal("content-b", b.Content);
        Assert.Equal("hist", b.Metadata["cat"]);
        Assert.Equal(new float[] { 4, 5, 6 }, b.Embedding);
        Assert.False(restored.TryGetByDocumentId("a", out _));
    }

    [Fact]
    public void FromJson_EmptyOrNull_ReturnsEmptySidecar()
    {
        Assert.Equal(0, FaissSidecar.FromJson("").Count);
        Assert.Equal(0, FaissSidecar.FromJson("   ").Count);
    }

    [Theory]
    [InlineData(5, 4, 100, false, 5)]    // no filter: fetch exactly topK
    [InlineData(5, 4, 100, true, 20)]    // filter: over-fetch topK * oversample
    [InlineData(5, 4, 12, true, 12)]     // over-fetch clamped to total docs
    [InlineData(5, 4, 3, false, 3)]      // topK clamped to total docs
    [InlineData(0, 4, 100, true, 0)]     // topK <= 0
    [InlineData(5, 4, 0, true, 0)]       // empty index
    public void ComputeFetchCount_AppliesOversampleAndClamp(int topK, int oversample, int total, bool hasFilters, int expected)
    {
        Assert.Equal(expected, FaissRetrievalPlanner.ComputeFetchCount(topK, oversample, total, hasFilters));
    }

    [Fact]
    public void SelectTopK_OverFetchesThenFiltersAndTakesTopK_PreservingOrder()
    {
        // Candidates already in FAISS best-first order; only some pass the metadata filter.
        var ranked = new List<(FaissSidecarEntry Entry, double RawScore)>
        {
            (Entry("d1", 2024), 0.99),
            (Entry("d2", 2019), 0.95), // filtered out (year != 2024)
            (Entry("d3", 2024), 0.90),
            (Entry("d4", 2024), 0.80),
        };
        var filters = new Dictionary<string, object> { ["year"] = 2024 };

        var result = FaissRetrievalPlanner.SelectTopK<double>(
            ranked,
            filters,
            topK: 2,
            scoreConverter: d => d,
            matches: EqualityMatch);

        Assert.Equal(2, result.Count);
        Assert.Equal("d1", result[0].Id);   // order preserved, d2 skipped
        Assert.Equal("d3", result[1].Id);
        Assert.Equal(0.99, result[0].RelevanceScore, 5);
        Assert.True(result[0].HasRelevanceScore);
    }

    [Fact]
    public void SelectTopK_NoFilters_ReturnsAllUpToTopK()
    {
        var ranked = new List<(FaissSidecarEntry Entry, double RawScore)>
        {
            (Entry("d1", 2024), 0.9),
            (Entry("d2", 2019), 0.8),
        };

        var result = FaissRetrievalPlanner.SelectTopK<double>(
            ranked,
            new Dictionary<string, object>(),
            topK: 10,
            scoreConverter: d => d,
            matches: EqualityMatch);

        Assert.Equal(new[] { "d1", "d2" }, result.Select(r => r.Id).ToArray());
    }

    private static FaissSidecarEntry Entry(string id, int year) => new()
    {
        DocumentId = id,
        Content = "content-" + id,
        Metadata = new Dictionary<string, object> { ["year"] = year }
    };

    private static bool EqualityMatch(Dictionary<string, object> metadata, Dictionary<string, object> filters)
        => filters.All(kv => metadata.TryGetValue(kv.Key, out var v) && Equals(v, kv.Value));
}
#endif
