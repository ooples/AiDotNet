using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for <see cref="MultiHopRetriever{T}"/> covering multi-hop iteration, de-duplication by
    /// document id, and early stopping when a hop contributes nothing new.
    /// </summary>
    public class MultiHopRetrieverTests
    {
        #region Stubs

        /// <summary>
        /// Retriever that returns a pre-configured sequence of result batches, one per call, and
        /// records the queries it received.
        /// </summary>
        private sealed class QueueRetriever : IRetriever<double>
        {
            private readonly Queue<List<Document<double>>> _batches;
            public List<string> QueriesReceived { get; } = new();
            public int CallCount { get; private set; }

            public QueueRetriever(IEnumerable<List<Document<double>>> batches)
            {
                _batches = new Queue<List<Document<double>>>(batches);
            }

            public int DefaultTopK => 5;

            public IEnumerable<Document<double>> Retrieve(string query) => Retrieve(query, DefaultTopK);

            public IEnumerable<Document<double>> Retrieve(string query, int topK) =>
                Retrieve(query, topK, new Dictionary<string, object>());

            public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
            {
                CallCount++;
                QueriesReceived.Add(query);
                return _batches.Count > 0 ? _batches.Dequeue() : new List<Document<double>>();
            }

            public Task<IEnumerable<Document<double>>> RetrieveAsync(
                string query, int topK, Dictionary<string, object>? metadataFilters = null,
                CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested();
                return Task.FromResult(Retrieve(query, topK, metadataFilters ?? new Dictionary<string, object>()));
            }
        }

        /// <summary>
        /// Generator that always proposes the same fixed follow-up query.
        /// </summary>
        private sealed class FixedFollowUpGenerator : IGenerator<double>
        {
            private readonly string _followUp;
            public int GenerateCallCount { get; private set; }

            public FixedFollowUpGenerator(string followUp)
            {
                _followUp = followUp;
            }

            public int MaxContextTokens => 4096;
            public int MaxGenerationTokens => 512;

            public string Generate(string prompt)
            {
                GenerateCallCount++;
                return _followUp;
            }

            public GroundedAnswer<double> GenerateGrounded(string query, IEnumerable<Document<double>> context) =>
                new GroundedAnswer<double> { Query = query, Answer = string.Empty };

            public Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested();
                return Task.FromResult(Generate(prompt));
            }

            public Task<GroundedAnswer<double>> GenerateGroundedAsync(
                string query, IEnumerable<Document<double>> context, CancellationToken cancellationToken = default)
            {
                cancellationToken.ThrowIfCancellationRequested();
                return Task.FromResult(GenerateGrounded(query, context));
            }
        }

        private static Document<double> Doc(string id, string content, double score)
        {
            return new Document<double>(id, content)
            {
                RelevanceScore = score,
                HasRelevanceScore = true
            };
        }

        #endregion

        #region Constructor

        [Fact(Timeout = 60000)]
        public async Task Constructor_NullBaseRetriever_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new MultiHopRetriever<double>(null!, new FixedFollowUpGenerator("f")));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_NullGenerator_Throws()
        {
            var retriever = new QueueRetriever(new List<List<Document<double>>>());
            Assert.Throws<ArgumentNullException>(() =>
                new MultiHopRetriever<double>(retriever, null!));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ZeroMaxHops_Throws()
        {
            var retriever = new QueueRetriever(new List<List<Document<double>>>());
            Assert.Throws<ArgumentException>(() =>
                new MultiHopRetriever<double>(retriever, new FixedFollowUpGenerator("f"), maxHops: 0));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_DefaultMaxHops_IsThree()
        {
            var retriever = new QueueRetriever(new List<List<Document<double>>>());
            var multiHop = new MultiHopRetriever<double>(retriever, new FixedFollowUpGenerator("f"));
            Assert.Equal(3, multiHop.MaxHops);
            await Task.CompletedTask;
        }

        #endregion

        #region Hop Iteration and Dedup

        [Fact(Timeout = 60000)]
        public async Task Retrieve_PerformsUpToMaxHops_AndDeduplicatesById()
        {
            // Arrange: 3 batches, overlapping documents across hops.
            var batches = new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9), Doc("d2", "second", 0.5) },
                new() { Doc("d2", "second again", 0.5), Doc("d3", "third", 0.7) },
                new() { Doc("d3", "third again", 0.7), Doc("d4", "fourth", 0.8) }
            };
            var retriever = new QueueRetriever(batches);
            var generator = new FixedFollowUpGenerator("follow up query");
            var multiHop = new MultiHopRetriever<double>(retriever, generator, maxHops: 3);

            // Act
            var results = multiHop.Retrieve("initial question").ToList();

            // Assert: 3 hops executed, 4 unique docs (d2, d3 deduped), ordered by score desc.
            Assert.Equal(3, retriever.CallCount);
            Assert.Equal(4, results.Count);
            Assert.Equal(new[] { "d1", "d2", "d3", "d4" }, results.Select(r => r.Id).OrderBy(x => x).ToArray());
            // No duplicate ids.
            Assert.Equal(results.Count, results.Select(r => r.Id).Distinct().Count());
            // Relevance-ordered (descending).
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore);
            }
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Retrieve_UsesGeneratorFollowUpAsNextQuery()
        {
            var batches = new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9) },
                new() { Doc("d2", "second", 0.8) }
            };
            var retriever = new QueueRetriever(batches);
            var generator = new FixedFollowUpGenerator("the follow up");
            var multiHop = new MultiHopRetriever<double>(retriever, generator, maxHops: 2);

            _ = multiHop.Retrieve("original").ToList();

            // First hop uses the original query; second hop uses the generator's follow-up.
            Assert.Equal("original", retriever.QueriesReceived[0]);
            Assert.Equal("the follow up", retriever.QueriesReceived[1]);
            Assert.True(generator.GenerateCallCount >= 1);
            await Task.CompletedTask;
        }

        #endregion

        #region Early Stopping

        [Fact(Timeout = 60000)]
        public async Task Retrieve_StopsEarly_WhenHopAddsNothingNew_EmptyBatch()
        {
            // Second hop returns no documents -> stop after 2 calls (not the full 3 hops).
            var batches = new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9), Doc("d2", "second", 0.5) },
                new List<Document<double>>()
            };
            var retriever = new QueueRetriever(batches);
            var multiHop = new MultiHopRetriever<double>(
                retriever, new FixedFollowUpGenerator("f"), maxHops: 3);

            var results = multiHop.Retrieve("q").ToList();

            Assert.Equal(2, retriever.CallCount);
            Assert.Equal(2, results.Count);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Retrieve_StopsEarly_WhenHopAddsOnlyDuplicates()
        {
            // Second hop returns the same doc already seen -> nothing new -> stop.
            var batches = new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9) },
                new() { Doc("d1", "first duplicate", 0.9) },
                new() { Doc("d2", "should never be reached", 0.5) }
            };
            var retriever = new QueueRetriever(batches);
            var multiHop = new MultiHopRetriever<double>(
                retriever, new FixedFollowUpGenerator("f"), maxHops: 3);

            var results = multiHop.Retrieve("q").ToList();

            Assert.Equal(2, retriever.CallCount);
            Assert.Single(results);
            Assert.Equal("d1", results[0].Id);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Retrieve_StopsEarly_WhenGeneratorReturnsEmptyFollowUp()
        {
            var batches = new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9) },
                new() { Doc("d2", "second", 0.8) }
            };
            var retriever = new QueueRetriever(batches);
            var multiHop = new MultiHopRetriever<double>(
                retriever, new FixedFollowUpGenerator("   "), maxHops: 3);

            var results = multiHop.Retrieve("q").ToList();

            // Only the first hop runs; empty follow-up prevents a second hop.
            Assert.Equal(1, retriever.CallCount);
            Assert.Single(results);
            await Task.CompletedTask;
        }

        #endregion

        #region Async

        [Fact(Timeout = 60000)]
        public async Task RetrieveAsync_ProducesSameResultAsSync()
        {
            var batches = new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9) },
                new() { Doc("d2", "second", 0.8) },
                new() { Doc("d3", "third", 0.7) }
            };
            var retriever = new QueueRetriever(batches);
            var multiHop = new MultiHopRetriever<double>(
                retriever, new FixedFollowUpGenerator("f"), maxHops: 3);

            var results = (await multiHop.RetrieveAsync("q", CancellationToken.None)).ToList();

            Assert.Equal(3, results.Count);
            Assert.Equal(3, retriever.CallCount);
        }

        [Fact(Timeout = 60000)]
        public async Task RetrieveAsync_CanceledToken_Throws()
        {
            var retriever = new QueueRetriever(new List<List<Document<double>>>
            {
                new() { Doc("d1", "first", 0.9) }
            });
            var multiHop = new MultiHopRetriever<double>(retriever, new FixedFollowUpGenerator("f"));
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            await Assert.ThrowsAsync<OperationCanceledException>(
                () => multiHop.RetrieveAsync("q", cts.Token));
        }

        #endregion
    }
}
