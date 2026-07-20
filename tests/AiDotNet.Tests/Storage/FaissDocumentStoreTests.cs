#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.Storage;

/// <summary>
/// Integration tests that build a real native FAISS index and search it through
/// <see cref="FaissDocumentStore{T}"/>. They self-skip when the native FAISS runtime cannot be
/// loaded (e.g. non-win-x64 CI), so the suite stays green without the native library.
/// </summary>
/// <remarks>
/// IVF / IVFPQ tests additionally require FAISS's k-means training, which links Intel MKL. Some
/// packaged FAISS builds ship an incomplete MKL redistributable (missing <c>mkl_def.*.dll</c>),
/// and a missing MKL kernel raises a PROCESS-FATAL error that a try/catch cannot recover — it would
/// abort the whole test host. Those tests are therefore opt-in via the
/// <c>AIDOTNET_FAISS_IVF=1</c> environment variable and are only meaningful where a complete MKL
/// is present. Flat and HNSW require no training and run whenever the native lib loads.
/// </remarks>
[Trait("Category", "Integration")]
public class FaissDocumentStoreTests
{
    private const int Dim = 8;

    /// <summary>Probes whether the native FAISS library loads by exercising a tiny Flat index.</summary>
    private static bool NativeAvailable()
    {
        try
        {
            using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.Flat, FaissDistanceMetric.L2);
            store.Add(Doc("probe", Unit(0)));
            _ = store.GetSimilar(Vec(Unit(0)), 1).ToList();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool IvfEnabled() =>
        Environment.GetEnvironmentVariable("AIDOTNET_FAISS_IVF") == "1";

    [SkippableFact]
    public void Flat_Cosine_FindsNearestNeighbor()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");

        using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.Flat, FaissDistanceMetric.Cosine);
        store.AddBatch(new[]
        {
            Doc("a", Unit(0)),
            Doc("b", Unit(1)),
            Doc("c", Unit(2)),
        });

        Assert.Equal(3, store.DocumentCount);

        // A query aligned with axis 1 (scaled) should return "b" first regardless of magnitude (cosine).
        var results = store.GetSimilar(Vec(Scaled(1, 5f)), 2).ToList();
        Assert.Equal("b", results[0].Id);
        Assert.True(results[0].HasRelevanceScore);
        Assert.True(Convert.ToDouble(results[0].RelevanceScore) >= Convert.ToDouble(results[1].RelevanceScore));
    }

    [SkippableFact]
    public void Flat_L2_OrdersByEuclideanDistance()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");

        using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.Flat, FaissDistanceMetric.L2);
        store.AddBatch(new[]
        {
            Doc("near", Custom(0.0f)),
            Doc("far", Custom(9.0f)),
        });

        var results = store.GetSimilar(Vec(Custom(0.1f)), 2).ToList();
        Assert.Equal("near", results[0].Id);
        Assert.Equal("far", results[1].Id);
    }

    [SkippableFact]
    public void GetSimilarWithFilters_OverFetchesThenFiltersMetadata()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");

        using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.Flat, FaissDistanceMetric.Cosine);
        // Closest vector is "wrong-cat"; the filter must push retrieval to the next matching doc.
        store.AddBatch(new[]
        {
            DocMeta("wrong-cat", Unit(1), new Dictionary<string, object> { ["cat"] = "history" }),
            DocMeta("right-cat", Scaled(1, 0.9f), new Dictionary<string, object> { ["cat"] = "science" }),
            DocMeta("other", Unit(2), new Dictionary<string, object> { ["cat"] = "science" }),
        });

        var filters = new Dictionary<string, object> { ["cat"] = "science" };
        var results = store.GetSimilarWithFilters(Vec(Unit(1)), 1, filters).ToList();

        Assert.Single(results);
        Assert.Equal("right-cat", results[0].Id);
    }

    [SkippableFact]
    public void Remove_Flat_DeletesFromIndexAndSidecar()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");

        using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.Flat, FaissDistanceMetric.L2);
        store.AddBatch(new[] { Doc("a", Unit(0)), Doc("b", Unit(1)) });

        Assert.True(store.Remove("a"));
        Assert.Equal(1, store.DocumentCount);
        Assert.Null(store.GetById("a"));
        Assert.NotNull(store.GetById("b"));

        var results = store.GetSimilar(Vec(Unit(0)), 5).ToList();
        Assert.DoesNotContain(results, r => r.Id == "a");
    }

    [SkippableFact]
    public void Hnsw_AddSearch_AndRemoveViaRebuild()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");

        using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.HNSW, FaissDistanceMetric.L2);
        store.AddBatch(Enumerable.Range(0, 10).Select(i => Doc("d" + i, Custom(i))).ToArray());

        var results = store.GetSimilar(Vec(Custom(3)), 3).ToList();
        Assert.Equal("d3", results[0].Id);

        // HNSW has no in-place delete; the store must rebuild from the sidecar and stay consistent.
        Assert.True(store.Remove("d3"));
        Assert.Equal(9, store.DocumentCount);
        var after = store.GetSimilar(Vec(Custom(3)), 3).ToList();
        Assert.DoesNotContain(after, r => r.Id == "d3");
    }

    [SkippableFact]
    public void SaveAndLoad_RoundTripsIndexAndSidecar()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");

        var basePath = Path.Combine(Path.GetTempPath(), "faiss_test_" + Guid.NewGuid().ToString("N"));
        try
        {
            using (var store = new FaissDocumentStore<float>(Dim, FaissIndexType.Flat, FaissDistanceMetric.Cosine))
            {
                store.AddBatch(new[]
                {
                    DocMeta("a", Unit(0), new Dictionary<string, object> { ["k"] = "v" }),
                    DocMeta("b", Unit(1), new Dictionary<string, object>()),
                });
                store.Save(basePath);
            }

            using var loaded = FaissDocumentStore<float>.Load(basePath);
            Assert.Equal(2, loaded.DocumentCount);
            Assert.Equal(FaissIndexType.Flat, loaded.IndexType);
            Assert.Equal(FaissDistanceMetric.Cosine, loaded.Metric);

            var doc = loaded.GetById("a");
            Assert.NotNull(doc);
            Assert.Equal("v", doc!.Metadata["k"]);

            var results = loaded.GetSimilar(Vec(Unit(0)), 1).ToList();
            Assert.Equal("a", results[0].Id);
        }
        finally
        {
            TryDelete(basePath + ".faissindex");
            TryDelete(basePath + ".faissmeta.json");
        }
    }

    [SkippableFact]
    public void IvfFlat_TrainAddSearch_OptIn()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");
        Skip.IfNot(IvfEnabled(), "IVF training requires a complete Intel MKL runtime; set AIDOTNET_FAISS_IVF=1 to enable");

        using var store = new FaissDocumentStore<float>(Dim, FaissIndexType.IVFFlat, FaissDistanceMetric.L2, nlist: 4);
        var rnd = new Random(0);
        var docs = Enumerable.Range(0, 256)
            .Select(i => Doc("d" + i, Enumerable.Range(0, Dim).Select(_ => (float)rnd.NextDouble()).ToArray()))
            .ToArray();
        store.AddBatch(docs);

        Assert.Equal(256, store.DocumentCount);
        var results = store.GetSimilar(Vec(docs[0].Embedding.ToArray()), 3).ToList();
        Assert.NotEmpty(results);
    }

    [SkippableFact]
    public void IvfPq_TrainAddSearch_OptIn()
    {
        Skip.IfNot(NativeAvailable(), "Native FAISS library not available on this host");
        Skip.IfNot(IvfEnabled(), "IVFPQ training requires a complete Intel MKL runtime; set AIDOTNET_FAISS_IVF=1 to enable");

        const int dim = 16;
        using var store = new FaissDocumentStore<float>(dim, FaissIndexType.IVFPQ, FaissDistanceMetric.L2, nlist: 8, pqM: 4);
        var rnd = new Random(1);
        var docs = Enumerable.Range(0, 512)
            .Select(i => Doc("d" + i, Enumerable.Range(0, dim).Select(_ => (float)rnd.NextDouble()).ToArray()))
            .ToArray();
        store.AddBatch(docs);

        Assert.Equal(512, store.DocumentCount);
        var results = store.GetSimilar(Vec(docs[0].Embedding.ToArray()), 3).ToList();
        Assert.NotEmpty(results);
    }

    // ---- helpers -------------------------------------------------------------

    private static float[] Unit(int axis)
    {
        var v = new float[Dim];
        v[axis] = 1f;
        return v;
    }

    private static float[] Scaled(int axis, float scale)
    {
        var v = new float[Dim];
        v[axis] = scale;
        return v;
    }

    private static float[] Custom(float baseValue)
        => Enumerable.Range(0, Dim).Select(j => baseValue + j * 0.01f).ToArray();

    private static Vector<float> Vec(float[] data) => new(data);

    private static VectorDocument<float> Doc(string id, float[] embedding)
        => new(new Document<float>(id, "content-" + id), new Vector<float>(embedding));

    private static VectorDocument<float> DocMeta(string id, float[] embedding, Dictionary<string, object> metadata)
        => new(new Document<float>(id, "content-" + id, metadata), new Vector<float>(embedding));

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best effort */ }
    }
}
#endif
