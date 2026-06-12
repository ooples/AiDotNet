using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Memory;
using AiDotNet.Agentic.Models;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    public class AgentMemoryStoreTests
    {
        // ---- InMemoryAgentMemoryStore (lexical) ----

        [Fact(Timeout = 60000)]
        public async Task InMemoryStore_RanksByLexicalOverlap()
        {
            var store = new InMemoryAgentMemoryStore();
            await store.AddAsync("The project deadline is in June.");
            await store.AddAsync("The user prefers dark mode in the editor.");
            await store.AddAsync("Coffee is served in the break room.");

            var results = await store.SearchAsync("when is the project deadline", topK: 2);

            Assert.NotEmpty(results);
            Assert.Contains("deadline", results[0].Memory.Content);
            // Scores are sorted descending.
            for (var i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].Score >= results[i].Score);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task InMemoryStore_NoOverlap_ReturnsEmpty()
        {
            var store = new InMemoryAgentMemoryStore();
            await store.AddAsync("Coffee is served in the break room.");

            var results = await store.SearchAsync("quantum chromodynamics", topK: 5);

            Assert.Empty(results);
        }

        [Fact(Timeout = 60000)]
        public async Task InMemoryStore_RemoveAndGetAll()
        {
            var store = new InMemoryAgentMemoryStore();
            var id = await store.AddAsync("Fact one.");
            await store.AddAsync("Fact two.");

            Assert.Equal(2, (await store.GetAllAsync()).Count);

            await store.RemoveAsync(id);
            var all = await store.GetAllAsync();
            Assert.Single(all);
            Assert.Equal("Fact two.", all[0].Content);
        }

        // ---- EmbeddingAgentMemoryStore (semantic, reuses the RAG embedding + cosine stack) ----

        [Fact(Timeout = 60000)]
        public async Task EmbeddingStore_RanksBySemanticSimilarity_NotWords()
        {
            // The fake model maps synonyms to the same vector, so "due date" matches "deadline" by meaning.
            var model = new FakeEmbeddingModel(new Dictionary<string, double[]>
            {
                ["The project deadline is in June."] = new[] { 1.0, 0.0, 0.0 },
                ["The user prefers dark mode."] = new[] { 0.0, 1.0, 0.0 },
                ["Coffee is in the break room."] = new[] { 0.0, 0.0, 1.0 },
                ["when is the due date"] = new[] { 1.0, 0.0, 0.0 }, // same vector as the deadline memory
            });
            var store = new EmbeddingAgentMemoryStore<double>(model);
            await store.AddAsync("The project deadline is in June.");
            await store.AddAsync("The user prefers dark mode.");
            await store.AddAsync("Coffee is in the break room.");

            var results = await store.SearchAsync("when is the due date", topK: 1);

            Assert.Single(results);
            // No lexical overlap with "deadline", yet it ranks first by vector similarity.
            Assert.Equal("The project deadline is in June.", results[0].Memory.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbeddingStore_RespectsTopK()
        {
            var model = new FakeEmbeddingModel(new Dictionary<string, double[]>
            {
                ["a"] = new[] { 1.0, 0.0 },
                ["b"] = new[] { 0.9, 0.1 },
                ["c"] = new[] { 0.8, 0.2 },
                ["q"] = new[] { 1.0, 0.0 },
            });
            var store = new EmbeddingAgentMemoryStore<double>(model);
            await store.AddAsync("a");
            await store.AddAsync("b");
            await store.AddAsync("c");

            var results = await store.SearchAsync("q", topK: 2);
            Assert.Equal(2, results.Count);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbeddingStore_EmptyStore_ReturnsEmpty()
        {
            var model = new FakeEmbeddingModel(new Dictionary<string, double[]> { ["q"] = new[] { 1.0 } });
            var store = new EmbeddingAgentMemoryStore<double>(model);
            Assert.Empty(await store.SearchAsync("q"));
        }

        // ---- MemoryAugmentedAgent ----

        [Fact(Timeout = 60000)]
        public async Task MemoryAugmentedAgent_InjectsRelevantMemoryAsContext()
        {
            var store = new InMemoryAgentMemoryStore();
            await store.AddAsync("The user prefers dark mode in the editor.");

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("Dark mode it is."));
            var inner = new AgentExecutor<double>(client);
            var augmented = new MemoryAugmentedAgent<double>(inner, store);

            await augmented.RunAsync(new[] { ChatMessage.User("Which editor mode do I prefer?") });

            // The recalled memory was injected as a system-context message before the user's question.
            var request = client.Requests[0];
            Assert.Contains(request, m => m.Role == ChatRole.System && m.Text.Contains("dark mode"));
        }

        [Fact(Timeout = 60000)]
        public async Task MemoryAugmentedAgent_NoRelevantMemory_DoesNotInject()
        {
            var store = new InMemoryAgentMemoryStore();
            await store.AddAsync("Coffee is served in the break room.");

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var augmented = new MemoryAugmentedAgent<double>(new AgentExecutor<double>(client), store);

            await augmented.RunAsync(new[] { ChatMessage.User("Tell me about quantum chromodynamics.") });

            var request = client.Requests[0];
            Assert.DoesNotContain(request, m => m.Role == ChatRole.System);
        }

        [Fact(Timeout = 60000)]
        public async Task MemoryAugmentedAgent_ComposesWithThreadedAgent()
        {
            // Long-term recall (across conversations) + short-term thread memory together.
            var store = new InMemoryAgentMemoryStore();
            await store.AddAsync("The user's name is Alice.");

            var client = ScriptedChatClient<double>.Sequence(
                ChatResponses.Text("Hello Alice."),
                ChatResponses.Text("Yes, Alice."));
            var recall = new MemoryAugmentedAgent<double>(new AgentExecutor<double>(client), store);
            var threaded = new ThreadedAgent<double>(recall, new InMemoryConversationStore());

            await threaded.RunAsync("conv", "What is my name?");
            await threaded.RunAsync("conv", "Do you remember my name?");

            // Second turn sees both the long-term memory (name) and the prior turn (thread history).
            var secondRequest = client.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.System && m.Text.Contains("Alice"));
            Assert.Contains(secondRequest, m => m.Role == ChatRole.User && m.Text == "What is my name?");
        }

        // Deterministic embedding model for tests: looks up a fixed vector per exact text.
        private sealed class FakeEmbeddingModel : IEmbeddingModel<double>
        {
            private readonly Dictionary<string, double[]> _vectors;
            private readonly int _dimension;

            public FakeEmbeddingModel(Dictionary<string, double[]> vectors)
            {
                _vectors = vectors;
                _dimension = vectors.Values.Select(v => v.Length).DefaultIfEmpty(1).Max();
            }

            public int EmbeddingDimension => _dimension;

            public int MaxTokens => 512;

            public Vector<double> Embed(string text)
            {
                var data = _vectors.TryGetValue(text, out var vector)
                    ? Pad(vector)
                    : new double[_dimension];
                return new Vector<double>(data);
            }

            public Task<Vector<double>> EmbedAsync(string text) => Task.FromResult(Embed(text));

            public Matrix<double> EmbedBatch(IEnumerable<string> texts)
            {
                var list = texts.ToList();
                var matrix = new Matrix<double>(list.Count, _dimension);
                for (var i = 0; i < list.Count; i++)
                {
                    var row = Embed(list[i]);
                    for (var j = 0; j < _dimension; j++)
                    {
                        matrix[i, j] = row[j];
                    }
                }

                return matrix;
            }

            public Task<Matrix<double>> EmbedBatchAsync(IEnumerable<string> texts) =>
                Task.FromResult(EmbedBatch(texts));

            private double[] Pad(double[] vector)
            {
                if (vector.Length == _dimension)
                {
                    return vector;
                }

                var padded = new double[_dimension];
                Array.Copy(vector, padded, Math.Min(vector.Length, _dimension));
                return padded;
            }
        }
    }
}
