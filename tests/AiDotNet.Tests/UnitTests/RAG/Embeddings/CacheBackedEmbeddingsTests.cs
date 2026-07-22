#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Xunit;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    /// <summary>
    /// A fake <see cref="IEmbeddingModel{T}"/> that produces deterministic embeddings and records how many
    /// times, and with which texts, it was invoked. Used to assert caching behavior without any network access.
    /// </summary>
    internal sealed class CountingEmbeddingModel<T> : IEmbeddingModel<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        public CountingEmbeddingModel(int dimension = 8, int maxTokens = 512)
        {
            EmbeddingDimension = dimension;
            MaxTokens = maxTokens;
        }

        public int EmbeddingDimension { get; }
        public int MaxTokens { get; }

        public int EmbedCallCount { get; private set; }
        public int BatchCallCount { get; private set; }
        public int TotalTextsEmbedded { get; private set; }
        public List<string> EmbeddedTexts { get; } = new();

        /// <summary>Deterministic, dimension-aware signature for a text (independent of call state).</summary>
        public static Vector<T> ExpectedVector(string text, int dimension)
        {
            double baseValue = 0;
            for (int i = 0; i < text.Length; i++)
            {
                baseValue += text[i] * (i + 1);
            }

            var values = new T[dimension];
            for (int j = 0; j < dimension; j++)
            {
                values[j] = NumOps.FromDouble(baseValue + j);
            }

            return new Vector<T>(values);
        }

        public Vector<T> Embed(string text)
        {
            EmbedCallCount++;
            TotalTextsEmbedded++;
            EmbeddedTexts.Add(text);
            return ExpectedVector(text, EmbeddingDimension);
        }

        public Task<Vector<T>> EmbedAsync(string text) => Task.FromResult(Embed(text));

        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var list = texts.ToList();
            BatchCallCount++;
            TotalTextsEmbedded += list.Count;
            EmbeddedTexts.AddRange(list);

            var matrix = new Matrix<T>(list.Count, EmbeddingDimension);
            for (int i = 0; i < list.Count; i++)
            {
                var v = ExpectedVector(list[i], EmbeddingDimension);
                for (int j = 0; j < EmbeddingDimension; j++)
                {
                    matrix[i, j] = v[j];
                }
            }

            return matrix;
        }

        public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts) => Task.FromResult(EmbedBatch(texts));
    }

    public class CacheBackedEmbeddingsTests
    {
        private static bool VectorsEqual(Vector<double> a, Vector<double> b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (Math.Abs(a[i] - b[i]) > 1e-9) return false;
            }
            return true;
        }

        // ---------- ContentHash ----------

        [Fact]
        public void ContentHash_IsStableAndKnownForKnownInput()
        {
            // SHA-256("abc") well-known digest.
            Assert.Equal(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                ContentHash.ComputeHash("abc"));
        }

        [Fact]
        public void ContentHash_SameInput_SameHash_DifferentInput_DifferentHash()
        {
            Assert.Equal(ContentHash.ComputeHash("hello world"), ContentHash.ComputeHash("hello world"));
            Assert.NotEqual(ContentHash.ComputeHash("hello world"), ContentHash.ComputeHash("hello world!"));
            Assert.Equal(64, ContentHash.ComputeHash("anything").Length);
        }

        // ---------- Basic delegation ----------

        [Fact]
        public void DimensionAndMaxTokens_DelegateToInner()
        {
            var inner = new CountingEmbeddingModel<double>(dimension: 32, maxTokens: 777);
            var cached = new CacheBackedEmbeddings<double>(inner);

            Assert.Equal(32, cached.EmbeddingDimension);
            Assert.Equal(777, cached.MaxTokens);
        }

        [Fact]
        public void Constructor_NullInner_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new CacheBackedEmbeddings<double>(null));
        }

        // ---------- (a) Single embed caching ----------

        [Fact]
        public void Embed_SameTextTwice_InnerCalledOnce_ReturnsCachedValue()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            var first = cached.Embed("repeated text");
            var second = cached.Embed("repeated text");

            Assert.Equal(1, inner.EmbedCallCount);
            Assert.True(VectorsEqual(first, second));
            Assert.True(VectorsEqual(first, CountingEmbeddingModel<double>.ExpectedVector("repeated text", inner.EmbeddingDimension)));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedAsync_SameTextTwice_InnerCalledOnce()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            var first = await cached.EmbedAsync("async text");
            var second = await cached.EmbedAsync("async text");

            Assert.Equal(1, inner.EmbedCallCount);
            Assert.True(VectorsEqual(first, second));
        }

        [Fact]
        public void Embed_DifferentTexts_InnerCalledPerDistinctText()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            cached.Embed("one");
            cached.Embed("two");
            cached.Embed("one");

            Assert.Equal(2, inner.EmbedCallCount);
            Assert.Equal(2, cached.Cache.Count);
        }

        // ---------- (b) Batch dedup + ordering ----------

        [Fact]
        public void EmbedBatch_DedupsWithinCall_PreservesOrder()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            var texts = new List<string> { "a", "b", "a", "c", "b" };
            var matrix = cached.EmbedBatch(texts);

            // Distinct misses = a, b, c => inner sees 3 texts in a single batch call.
            Assert.Equal(1, inner.BatchCallCount);
            Assert.Equal(3, inner.TotalTextsEmbedded);
            Assert.Equal(new List<string> { "a", "b", "c" }, inner.EmbeddedTexts);

            // Ordering + correctness: row i must equal the expected embedding for texts[i].
            Assert.Equal(5, matrix.Rows);
            for (int i = 0; i < texts.Count; i++)
            {
                var expected = CountingEmbeddingModel<double>.ExpectedVector(texts[i], inner.EmbeddingDimension);
                for (int j = 0; j < inner.EmbeddingDimension; j++)
                {
                    Assert.Equal(expected[j], matrix[i, j], 9);
                }
            }
        }

        [Fact]
        public void EmbedBatch_DedupsAcrossCalls()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            cached.EmbedBatch(new List<string> { "a", "b", "c" });
            Assert.Equal(3, inner.TotalTextsEmbedded);

            // Second call: only "d" is new; a/b/c are served from cache.
            var matrix = cached.EmbedBatch(new List<string> { "a", "b", "d" });

            Assert.Equal(4, inner.TotalTextsEmbedded);
            Assert.Equal(new List<string> { "a", "b", "c", "d" }, inner.EmbeddedTexts);

            var texts = new[] { "a", "b", "d" };
            for (int i = 0; i < texts.Length; i++)
            {
                var expected = CountingEmbeddingModel<double>.ExpectedVector(texts[i], inner.EmbeddingDimension);
                for (int j = 0; j < inner.EmbeddingDimension; j++)
                {
                    Assert.Equal(expected[j], matrix[i, j], 9);
                }
            }
        }

        [Fact]
        public void EmbedBatch_AllCached_DoesNotCallInner()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            cached.EmbedBatch(new List<string> { "x", "y" });
            var batchCallsAfterWarmup = inner.BatchCallCount;

            cached.EmbedBatch(new List<string> { "x", "y" });

            // No new inner batch invocation because everything was cached.
            Assert.Equal(batchCallsAfterWarmup, inner.BatchCallCount);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatchAsync_DedupsAndPreservesOrder()
        {
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner);

            var texts = new List<string> { "p", "q", "p" };
            var matrix = await cached.EmbedBatchAsync(texts);

            Assert.Equal(2, inner.TotalTextsEmbedded);
            for (int i = 0; i < texts.Count; i++)
            {
                var expected = CountingEmbeddingModel<double>.ExpectedVector(texts[i], inner.EmbeddingDimension);
                for (int j = 0; j < inner.EmbeddingDimension; j++)
                {
                    Assert.Equal(expected[j], matrix[i, j], 9);
                }
            }
        }

        [Fact]
        public void EmbedBatch_NullOrEmpty_Throws()
        {
            var cached = new CacheBackedEmbeddings<double>(new CountingEmbeddingModel<double>());

            Assert.Throws<ArgumentNullException>(() => cached.EmbedBatch(null));
            Assert.Throws<ArgumentException>(() => cached.EmbedBatch(new List<string>()));
        }

        // ---------- (c) Model identity / dimension isolation ----------

        [Fact]
        public void DifferentModelDimension_SharedCache_NoCrossHits()
        {
            var cache = new InMemoryEmbeddingCache<double>();
            var innerA = new CountingEmbeddingModel<double>(dimension: 8);
            var innerB = new CountingEmbeddingModel<double>(dimension: 16);

            var cachedA = new CacheBackedEmbeddings<double>(innerA, cache);
            var cachedB = new CacheBackedEmbeddings<double>(innerB, cache);

            cachedA.Embed("same text");
            cachedB.Embed("same text");

            // Each inner model was invoked; the entries did not collide.
            Assert.Equal(1, innerA.EmbedCallCount);
            Assert.Equal(1, innerB.EmbedCallCount);
            Assert.Equal(2, cache.Count);
            Assert.NotEqual(cachedA.Namespace, cachedB.Namespace);
        }

        [Fact]
        public void ExplicitNamespace_IsolatesEntries()
        {
            var cache = new InMemoryEmbeddingCache<double>();
            var innerA = new CountingEmbeddingModel<double>();
            var innerB = new CountingEmbeddingModel<double>();

            var cachedA = new CacheBackedEmbeddings<double>(innerA, cache, modelNamespace: "model-a");
            var cachedB = new CacheBackedEmbeddings<double>(innerB, cache, modelNamespace: "model-b");

            cachedA.Embed("shared");
            cachedB.Embed("shared");

            Assert.Equal(1, innerA.EmbedCallCount);
            Assert.Equal(1, innerB.EmbedCallCount);
            Assert.Equal(2, cache.Count);
        }

        // ---------- (d) LRU eviction ----------

        [Fact]
        public void InMemoryCache_Unbounded_KeepsEverything()
        {
            var cache = new InMemoryEmbeddingCache<double>();
            for (int i = 0; i < 100; i++)
            {
                cache.Set("k" + i, CountingEmbeddingModel<double>.ExpectedVector("t" + i, 4));
            }

            Assert.Equal(100, cache.Count);
        }

        [Fact]
        public void InMemoryCache_LruEviction_EvictsLeastRecentlyUsed()
        {
            var cache = new InMemoryEmbeddingCache<double>(maxSize: 2);
            var v = CountingEmbeddingModel<double>.ExpectedVector("v", 4);

            cache.Set("k1", v);
            cache.Set("k2", v);

            // Touch k1 so k2 becomes least recently used.
            Assert.True(cache.TryGet("k1", out _));

            cache.Set("k3", v);

            Assert.Equal(2, cache.Count);
            Assert.True(cache.TryGet("k1", out _));
            Assert.False(cache.TryGet("k2", out _));
            Assert.True(cache.TryGet("k3", out _));
        }

        [Fact]
        public void Decorator_WithLruCache_ReembedsAfterEviction()
        {
            var cache = new InMemoryEmbeddingCache<double>(maxSize: 1);
            var inner = new CountingEmbeddingModel<double>();
            var cached = new CacheBackedEmbeddings<double>(inner, cache);

            cached.Embed("a");            // miss -> count 1
            cached.Embed("b");            // miss -> count 2, evicts "a"
            cached.Embed("a");            // miss again (evicted) -> count 3

            Assert.Equal(3, inner.EmbedCallCount);
            Assert.Equal(1, cache.Count);
        }

        [Fact]
        public void Cache_Clear_RemovesAllEntries()
        {
            var cache = new InMemoryEmbeddingCache<double>();
            cache.Set("k1", CountingEmbeddingModel<double>.ExpectedVector("a", 4));
            cache.Set("k2", CountingEmbeddingModel<double>.ExpectedVector("b", 4));
            Assert.Equal(2, cache.Count);

            cache.Clear();
            Assert.Equal(0, cache.Count);
        }
    }
}
