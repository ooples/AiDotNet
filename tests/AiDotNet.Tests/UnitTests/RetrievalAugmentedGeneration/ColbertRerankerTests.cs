using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for <see cref="ColbertReranker{T}"/> covering the ColBERT MaxSim scoring rule and the
    /// resulting relevance ordering.
    /// </summary>
    public class ColbertRerankerTests
    {
        #region Test Helpers

        /// <summary>
        /// Deterministic token embedding model that returns caller-supplied hand-computed vectors for
        /// specific texts. Enables asserting the exact MaxSim value produced through the public API.
        /// </summary>
        private sealed class DictionaryTokenEmbeddingModel : ITokenEmbeddingModel<double>
        {
            private readonly Dictionary<string, Vector<double>[]> _map;

            public DictionaryTokenEmbeddingModel(Dictionary<string, Vector<double>[]> map, int embeddingDimension)
            {
                _map = map;
                EmbeddingDimension = embeddingDimension;
            }

            public int EmbeddingDimension { get; }

            public Vector<double>[] EmbedTokens(string text)
            {
                return _map.TryGetValue(text, out var vectors)
                    ? vectors
                    : Array.Empty<Vector<double>>();
            }
        }

        private static Vector<double> V(params double[] values) => new Vector<double>(values);

        private static Document<double> Doc(string id, string content) => new Document<double>(id, content);

        #endregion

        #region MaxSim Math

        [Fact(Timeout = 60000)]
        public async Task MaxSim_HandComputedVectors_ReturnsSumOfPerQueryTokenMaxDotProducts()
        {
            // Arrange: query tokens q1=[1,0], q2=[0,1]; doc tokens d1=[1,0], d2=[0.6,0.8].
            var queryTokens = new[] { V(1.0, 0.0), V(0.0, 1.0) };
            var docTokens = new[] { V(1.0, 0.0), V(0.6, 0.8) };

            // Act
            var score = ColbertReranker<double>.MaxSim(queryTokens, docTokens);

            // Assert: max(1.0, 0.6) + max(0.0, 0.8) = 1.0 + 0.8 = 1.8
            Assert.Equal(1.8, score, 6);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task MaxSim_EmptyDocumentTokens_ReturnsZero()
        {
            var queryTokens = new[] { V(1.0, 0.0) };
            var docTokens = Array.Empty<Vector<double>>();

            var score = ColbertReranker<double>.MaxSim(queryTokens, docTokens);

            Assert.Equal(0.0, score, 6);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task MaxSim_IdenticalUnitTokens_ScoresOnePerQueryToken()
        {
            // Two query tokens each perfectly matched by a doc token (cosine 1.0) => score == 2.
            var queryTokens = new[] { V(1.0, 0.0), V(0.0, 1.0) };
            var docTokens = new[] { V(1.0, 0.0), V(0.0, 1.0) };

            var score = ColbertReranker<double>.MaxSim(queryTokens, docTokens);

            Assert.Equal(2.0, score, 6);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Rerank_UsesMaxSim_ScoreMatchesHandComputedValue()
        {
            // Arrange: supply exact per-token vectors via a dictionary embedder so the reranker's
            // MaxSim result is fully determined and hand-checkable.
            var map = new Dictionary<string, Vector<double>[]>
            {
                ["query"] = new[] { V(1.0, 0.0), V(0.0, 1.0) },
                ["doc content"] = new[] { V(1.0, 0.0), V(0.6, 0.8) }
            };
            var reranker = new ColbertReranker<double>(new DictionaryTokenEmbeddingModel(map, 2));
            var docs = new[] { Doc("d1", "doc content") };

            // Act
            var result = reranker.Rerank("query", docs).ToList();

            // Assert: same 1.8 computed by hand (vectors are already unit length).
            Assert.Single(result);
            Assert.True(result[0].HasRelevanceScore);
            Assert.Equal(1.8, result[0].RelevanceScore, 6);
            await Task.CompletedTask;
        }

        #endregion

        #region Reranking Order

        [Fact(Timeout = 60000)]
        public async Task Rerank_OrdersClearlyMoreRelevantDocumentFirst()
        {
            // Arrange: default offline hashing token embedder gives exact-match tokens a cosine of 1.
            var reranker = new ColbertReranker<double>();
            var docs = new[]
            {
                Doc("irrelevant", "xylophone zebra umbrella"),
                Doc("relevant", "quantum computing qubits and entanglement"),
                Doc("partial", "classical computing algorithms")
            };

            // Act
            var result = reranker.Rerank("quantum computing", docs).ToList();

            // Assert: the doc containing both query tokens must rank first.
            Assert.Equal("relevant", result[0].Id);
            // Scores must be in non-increasing order.
            for (int i = 1; i < result.Count; i++)
            {
                Assert.True(result[i - 1].RelevanceScore >= result[i].RelevanceScore);
            }
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Rerank_TopK_ReturnsOnlyTopResults()
        {
            var reranker = new ColbertReranker<double>();
            var docs = new[]
            {
                Doc("d1", "alpha beta gamma"),
                Doc("d2", "beta gamma delta"),
                Doc("d3", "totally unrelated words here")
            };

            var result = reranker.Rerank("alpha beta", docs, topK: 2).ToList();

            Assert.Equal(2, result.Count);
            Assert.Equal("d1", result[0].Id);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task ModifiesScores_IsTrue()
        {
            Assert.True(new ColbertReranker<double>().ModifiesScores);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Rerank_EmptyDocuments_ReturnsEmpty()
        {
            var reranker = new ColbertReranker<double>();

            var result = reranker.Rerank("anything", new List<Document<double>>()).ToList();

            Assert.Empty(result);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task RerankAsync_ProducesSameOrderingAsSync()
        {
            var reranker = new ColbertReranker<double>();
            var docs = new[]
            {
                Doc("irrelevant", "xylophone zebra umbrella"),
                Doc("relevant", "quantum computing qubits")
            };

            var result = (await reranker.RerankAsync("quantum computing", docs, CancellationToken.None)).ToList();

            Assert.Equal("relevant", result[0].Id);
        }

        [Fact(Timeout = 60000)]
        public async Task RerankAsync_CanceledToken_Throws()
        {
            var reranker = new ColbertReranker<double>();
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            await Assert.ThrowsAsync<OperationCanceledException>(
                () => reranker.RerankAsync("q", new[] { Doc("d1", "content") }, cts.Token));
        }

        #endregion

        #region Token Embedding Model

        [Fact(Timeout = 60000)]
        public async Task HashingTokenEmbeddingModel_IsDeterministicAndUnitLength()
        {
            var model = new HashingTokenEmbeddingModel<double>(embeddingDimension: 64);

            var a = model.EmbedTokens("hello world");
            var b = model.EmbedTokens("hello world");

            Assert.Equal(2, a.Length);
            Assert.Equal(a.Length, b.Length);
            for (int t = 0; t < a.Length; t++)
            {
                // Deterministic: same text -> identical vectors.
                for (int i = 0; i < a[t].Length; i++)
                {
                    Assert.Equal(a[t][i], b[t][i], 10);
                }

                // Unit length.
                double norm = Math.Sqrt(Enumerable.Range(0, a[t].Length).Sum(i => a[t][i] * a[t][i]));
                Assert.Equal(1.0, norm, 6);
            }
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task HashingTokenEmbeddingModel_IdenticalTokensHaveCosineOne()
        {
            var model = new HashingTokenEmbeddingModel<double>(embeddingDimension: 64);

            var one = model.EmbedTokens("apple")[0];
            var two = model.EmbedTokens("apple")[0];

            double dot = Enumerable.Range(0, one.Length).Sum(i => one[i] * two[i]);
            Assert.Equal(1.0, dot, 6);
            await Task.CompletedTask;
        }

        #endregion
    }
}
