using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    public class ConstrainedDecodingTests
    {
        // ---- TokenSampler with an allowed set ----

        [Fact(Timeout = 60000)]
        public async Task Sampler_Greedy_RespectsAllowedSet()
        {
            await Task.Yield();

            var sampler = new TokenSampler<double>();
            // Index 3 has the highest logit, but it is excluded by the allowed set.
            var logits = new Vector<double>(new[] { 0.1, 0.5, 0.2, 0.9 });

            var id = sampler.Sample(logits, new LocalSamplingOptions { Temperature = 0.0 }, new[] { 0, 1, 2 });

            Assert.Equal(1, id); // best among {0,1,2}
        }

        [Fact(Timeout = 60000)]
        public async Task Sampler_Sampling_NeverEmitsDisallowedToken()
        {
            await Task.Yield();

            var sampler = new TokenSampler<double>(seed: 5);
            var logits = new Vector<double>(new[] { 5.0, 0.1, 5.0, 0.1 });
            var allowed = new[] { 1, 3 };

            for (var i = 0; i < 40; i++)
            {
                var id = sampler.Sample(logits, new LocalSamplingOptions { Temperature = 1.0 }, allowed);
                Assert.Contains(id, allowed);
            }
        }

        // ---- FiniteStateTokenConstraint forces structure ----

        [Fact(Timeout = 60000)]
        public async Task FiniteStateConstraint_ForcesExactSequence_RegardlessOfModel()
        {
            await Task.Yield();

            var tokenizer = new MapTokenizer(
                eos: 0,
                ("{", 1), ("\"ok\"", 2), ("}", 3), ("garbage", 4));

            // A model that, left unconstrained, always wants token 4 ("garbage").
            var model = new BiasedModel(tokenizer.VocabularySize, preferredToken: 4);

            // Grammar: start -> 1 ; 1 -> 2 ; 2 -> 3 ; 3 -> terminal (stop).
            var constraint = new FiniteStateTokenConstraint(
                start: new[] { 1 },
                transitions: new Dictionary<int, IReadOnlyCollection<int>>
                {
                    [1] = new[] { 2 },
                    [2] = new[] { 3 },
                    [3] = Array.Empty<int>(),
                });

            var engine = new LocalEngineChatClient<double>(model, tokenizer, options: new LocalEngineOptions
            {
                Sampling = new LocalSamplingOptions { Temperature = 0.0 },
                Constraint = constraint,
            });

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("emit json") });

            // Despite the model preferring "garbage", the constraint forced the exact grammar output.
            Assert.Equal("{\"ok\"}", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        [Fact(Timeout = 60000)]
        public async Task AllowedTokenSetConstraint_RestrictsWholeOutput()
        {
            await Task.Yield();

            var tokenizer = new MapTokenizer(eos: 0, ("a", 1), ("b", 2), ("c", 3));
            var model = new BiasedModel(tokenizer.VocabularySize, preferredToken: 3); // wants "c"
            var constraint = new AllowedTokenSetConstraint(new[] { 1, 2 }); // only a or b allowed

            var engine = new LocalEngineChatClient<double>(model, tokenizer, options: new LocalEngineOptions
            {
                Sampling = new LocalSamplingOptions { Temperature = 0.0 },
                Constraint = constraint,
                MaxOutputTokens = 4,
            });

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("x") });

            Assert.DoesNotContain("c", response.Text);
            Assert.True(response.Text.Replace(" ", "").Length > 0);
        }

        // ---- Stop sequences ----

        [Fact(Timeout = 60000)]
        public async Task StopSequence_HaltsGeneration_AndTrimsOutput()
        {
            await Task.Yield();

            var tokenizer = new MapTokenizer(eos: 0, ("hello", 1), ("STOP", 2), ("world", 3));
            // Model emits: hello, STOP, world, ... (never EOS on its own within the budget).
            var model = new ScriptedModel(tokenizer.VocabularySize, new[] { 1, 2, 3, 3, 3 });

            var engine = new LocalEngineChatClient<double>(model, tokenizer, options: new LocalEngineOptions
            {
                Sampling = new LocalSamplingOptions { Temperature = 0.0 },
            });

            var response = await engine.GetResponseAsync(
                new[] { ChatMessage.User("go") },
                new ChatOptions { StopSequences = new[] { "STOP" }, MaxOutputTokens = 10 });

            // Output is trimmed at the stop sequence and generation halts.
            Assert.Equal("hello", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        // ---- Test doubles ----

        // Always biases toward one preferred token (one-hot logits), ignoring context.
        private sealed class BiasedModel : ICausalLanguageModel<double>
        {
            private readonly int _preferred;

            public BiasedModel(int vocabularySize, int preferredToken)
            {
                VocabularySize = vocabularySize;
                _preferred = preferredToken;
            }

            public int VocabularySize { get; }

            public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds)
            {
                var logits = new double[VocabularySize];
                for (var i = 0; i < VocabularySize; i++)
                {
                    logits[i] = i == _preferred ? 10.0 : 0.0;
                }

                return new Vector<double>(logits);
            }
        }

        // Emits a fixed sequence (one-hot), then EOS.
        private sealed class ScriptedModel : ICausalLanguageModel<double>
        {
            private readonly int[] _sequence;
            private int _index;

            public ScriptedModel(int vocabularySize, int[] sequence)
            {
                VocabularySize = vocabularySize;
                _sequence = sequence;
            }

            public int VocabularySize { get; }

            public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds)
            {
                var id = _index < _sequence.Length ? _sequence[_index] : 0;
                _index++;
                var logits = new double[VocabularySize];
                logits[id] = 1.0;
                return new Vector<double>(logits);
            }
        }

        // Tokenizer over an explicit (text, id) map; id 0 reserved for EOS.
        private sealed class MapTokenizer : IGenerationTokenizer
        {
            private readonly Dictionary<string, int> _wordToId = new(StringComparer.Ordinal);
            private readonly Dictionary<int, string> _idToWord = new();

            public MapTokenizer(int eos, params (string Word, int Id)[] entries)
            {
                EosTokenId = eos;
                _idToWord[eos] = string.Empty;
                var max = eos;
                foreach (var (word, id) in entries)
                {
                    _wordToId[word] = id;
                    _idToWord[id] = word;
                    max = Math.Max(max, id);
                }

                VocabularySize = max + 1;
            }

            public int VocabularySize { get; }

            public int EosTokenId { get; }

            public IReadOnlyList<int> Encode(string text)
            {
                var ids = text
                    .Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(w => _wordToId.TryGetValue(w, out var id) ? id : EosTokenId)
                    .ToList();
                if (ids.Count == 0)
                {
                    ids.Add(EosTokenId == 0 && _wordToId.Count > 0 ? _wordToId.Values.First() : 0);
                }

                return ids;
            }

            // Concatenate token texts directly (no separators) so grammar output like {"ok"} is exact.
            public string Decode(IReadOnlyList<int> tokenIds) =>
                string.Concat(tokenIds.Select(id => _idToWord.TryGetValue(id, out var w) ? w : string.Empty));
        }
    }
}
