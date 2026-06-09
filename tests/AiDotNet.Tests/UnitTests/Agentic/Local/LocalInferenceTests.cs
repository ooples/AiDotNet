using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    public class LocalInferenceTests
    {
        // ---- TokenSampler ----

        [Fact(Timeout = 60000)]
        public async Task Sampler_Greedy_PicksArgMax()
        {
            var sampler = new TokenSampler<double>();
            var logits = new Vector<double>(new[] { 0.1, 0.9, 0.3, 0.2 });

            var id = sampler.Sample(logits, new LocalSamplingOptions { Temperature = 0.0 });

            Assert.Equal(1, id);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Sampler_SameSeed_IsReproducible()
        {
            var logits = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
            var options = new LocalSamplingOptions { Temperature = 1.0 };

            var a = new TokenSampler<double>(seed: 123);
            var b = new TokenSampler<double>(seed: 123);

            var seqA = Enumerable.Range(0, 20).Select(_ => a.Sample(logits, options)).ToList();
            var seqB = Enumerable.Range(0, 20).Select(_ => b.Sample(logits, options)).ToList();

            Assert.Equal(seqA, seqB);
        }

        [Fact(Timeout = 60000)]
        public async Task Sampler_TopK1_AlwaysSelectsTopToken()
        {
            var sampler = new TokenSampler<double>(seed: 7);
            var logits = new Vector<double>(new[] { 0.2, 5.0, 0.1 }); // index 1 dominates
            var options = new LocalSamplingOptions { Temperature = 1.0, TopK = 1 };

            for (var i = 0; i < 25; i++)
            {
                Assert.Equal(1, sampler.Sample(logits, options));
            }

            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Sampler_TinyTopP_RestrictsToMostLikely()
        {
            var sampler = new TokenSampler<double>(seed: 7);
            var logits = new Vector<double>(new[] { 0.1, 5.0, 0.2 });
            // A tiny nucleus keeps only the single most-likely token.
            var options = new LocalSamplingOptions { Temperature = 1.0, TopP = 0.01 };

            for (var i = 0; i < 25; i++)
            {
                Assert.Equal(1, sampler.Sample(logits, options));
            }

            await Task.CompletedTask;
        }

        // ---- ChatMlPromptTemplate ----

        [Fact(Timeout = 60000)]
        public async Task Template_RendersRoles_AndOpensAssistantTurn()
        {
            var template = new ChatMlPromptTemplate();
            var prompt = template.Render(new[]
            {
                ChatMessage.System("be brief"),
                ChatMessage.User("hi"),
            });

            Assert.Contains("<|system|>\nbe brief\n", prompt);
            Assert.Contains("<|user|>\nhi\n", prompt);
            Assert.EndsWith("<|assistant|>\n", prompt);
            await Task.CompletedTask;
        }

        // ---- LocalEngineChatClient ----

        [Fact(Timeout = 60000)]
        public async Task Engine_GreedyGeneration_DecodesExpectedText_AndStopsOnEos()
        {
            var tokenizer = new WordTokenizer("hello", "world");
            // Emit "hello" (1), "world" (2), then EOS (0).
            var model = new ScriptedCausalModel(tokenizer.VocabularySize, new[] { 1, 2, 0 });
            var engine = new LocalEngineChatClient<double>(model, tokenizer);

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("greet") },
                new ChatOptions { Temperature = 0.0 });

            Assert.Equal("hello world", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
            Assert.NotNull(response.Usage);
            Assert.Equal(2, response.Usage.OutputTokens);
            Assert.Equal("local", response.ModelId);
        }

        [Fact(Timeout = 60000)]
        public async Task Engine_HitsTokenLimit_ReportsLength()
        {
            var tokenizer = new WordTokenizer("again");
            // Always emits the non-EOS token "again" (1), never stopping on its own.
            var model = new ScriptedCausalModel(tokenizer.VocabularySize, new[] { 1 }, repeatLast: true);
            var engine = new LocalEngineChatClient<double>(model, tokenizer);

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("go") },
                new ChatOptions { Temperature = 0.0, MaxOutputTokens = 3 });

            Assert.Equal(ChatFinishReason.Length, response.FinishReason);
            Assert.Equal(3, response.Usage.OutputTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Engine_Streaming_DeltasReconstructFullText()
        {
            var tokenizer = new WordTokenizer("alpha", "beta", "gamma");
            var model = new ScriptedCausalModel(tokenizer.VocabularySize, new[] { 1, 2, 3, 0 });
            var engine = new LocalEngineChatClient<double>(model, tokenizer);

            var text = string.Empty;
            ChatFinishReason? finish = null;
            await foreach (var update in engine.GetStreamingResponseAsync(new[] { ChatMessage.User("x") },
                new ChatOptions { Temperature = 0.0 }))
            {
                if (update.TextDelta is { } delta)
                {
                    text += delta;
                }

                if (update.FinishReason is { } reason)
                {
                    finish = reason;
                }
            }

            Assert.Equal("alpha beta gamma", text);
            Assert.Equal(ChatFinishReason.Stop, finish);
        }

        [Fact(Timeout = 60000)]
        public async Task Engine_IsDropInForAgentExecutor()
        {
            var tokenizer = new WordTokenizer("done");
            var model = new ScriptedCausalModel(tokenizer.VocabularySize, new[] { 1, 0 });
            var engine = new LocalEngineChatClient<double>(model, tokenizer);

            // The local engine drives a standard agent with no code changes.
            var agent = new AgentExecutor<double>(engine);
            var result = await agent.RunAsync("anything");

            Assert.True(result.Completed);
            Assert.Equal("done", result.FinalText);
        }

        // ---- Deterministic test doubles ----

        // Emits a predetermined token sequence as one-hot logits, ignoring context. When the sequence is
        // exhausted it emits EOS (id 0), or repeats the last token when repeatLast is set.
        private sealed class ScriptedCausalModel : ICausalLanguageModel<double>
        {
            private readonly int[] _sequence;
            private readonly bool _repeatLast;
            private int _index;

            public ScriptedCausalModel(int vocabularySize, int[] sequence, bool repeatLast = false)
            {
                VocabularySize = vocabularySize;
                _sequence = sequence;
                _repeatLast = repeatLast;
            }

            public int VocabularySize { get; }

            public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds)
            {
                int id;
                if (_index < _sequence.Length)
                {
                    id = _sequence[_index];
                }
                else
                {
                    id = _repeatLast ? _sequence[_sequence.Length - 1] : 0;
                }

                _index++;
                var logits = new double[VocabularySize];
                logits[id] = 1.0;
                return new Vector<double>(logits);
            }
        }

        // Minimal whitespace tokenizer: id 0 is EOS, words get ids 1.. in order. Unknown words map to id 1.
        private sealed class WordTokenizer : IGenerationTokenizer
        {
            private readonly Dictionary<string, int> _wordToId = new(StringComparer.Ordinal);
            private readonly Dictionary<int, string> _idToWord = new();

            public WordTokenizer(params string[] words)
            {
                _idToWord[0] = string.Empty; // EOS decodes to nothing
                var nextId = 1;
                foreach (var word in words)
                {
                    _wordToId[word] = nextId;
                    _idToWord[nextId] = word;
                    nextId++;
                }

                VocabularySize = nextId;
            }

            public int VocabularySize { get; }

            public int EosTokenId => 0;

            public IReadOnlyList<int> Encode(string text)
            {
                var ids = new List<int>();
                foreach (var word in text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    ids.Add(_wordToId.TryGetValue(word, out var id) ? id : 1);
                }

                if (ids.Count == 0)
                {
                    ids.Add(1);
                }

                return ids;
            }

            public string Decode(IReadOnlyList<int> tokenIds)
            {
                var words = tokenIds
                    .Select(id => _idToWord.TryGetValue(id, out var word) ? word : string.Empty)
                    .Where(w => w.Length > 0);
                return string.Join(" ", words);
            }
        }
    }
}
