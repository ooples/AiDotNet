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
    public class IncrementalDecodingTests
    {
        // Deterministic next-token rule keyed on the last token: 0(EOS)->1, 1->2, 2->3, 3->0(EOS).
        private static readonly Dictionary<int, int> Transition = new()
        {
            [0] = 1,
            [1] = 2,
            [2] = 3,
            [3] = 0,
        };

        private const int Vocab = 4;

        private static MapTokenizer BuildTokenizer() => new(eos: 0, ("A", 1), ("B", 2), ("C", 3));

        private static double[] LogitsFor(int lastToken)
        {
            var next = Transition.TryGetValue(lastToken, out var n) ? n : 0;
            var logits = new double[Vocab];
            logits[next] = 10.0;
            return logits;
        }

        [Fact(Timeout = 60000)]
        public async Task Engine_UsesIncrementalPath_FeedingOneTokenAtATime()
        {
            await Task.Yield();

            var model = new IncrementalProbe();
            var engine = new LocalEngineChatClient<double>(model, BuildTokenizer(), options: new LocalEngineOptions
            {
                Sampling = new LocalSamplingOptions { Temperature = 0.0 },
            });

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("go") });

            Assert.Equal("ABC", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);

            // The engine took the incremental fast path: reset once, primed once with the prompt, then fed
            // each generated token singly via AppendToken.
            Assert.Equal(1, model.ResetCalls);
            Assert.Equal(1, model.StartCalls);
            Assert.Equal(new[] { 1, 2, 3 }, model.AppendedTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task IncrementalAndFullRefeed_ProduceIdenticalOutput()
        {
            await Task.Yield();

            var tokenizer = BuildTokenizer();
            var sampling = new LocalSamplingOptions { Temperature = 0.0 };

            var incremental = new LocalEngineChatClient<double>(new IncrementalProbe(), tokenizer,
                options: new LocalEngineOptions { Sampling = sampling });
            var fullRefeed = new LocalEngineChatClient<double>(new FullRefeedModel(), tokenizer,
                options: new LocalEngineOptions { Sampling = sampling });

            var a = await incremental.GetResponseAsync(new[] { ChatMessage.User("go") });
            var b = await fullRefeed.GetResponseAsync(new[] { ChatMessage.User("go") });

            Assert.Equal(b.Text, a.Text);
            Assert.Equal("ABC", a.Text);
        }

        // ---- Test doubles ----

        // A KV-cached model: tracks how the engine drives it and advances state by the last token only.
        private sealed class IncrementalProbe : IIncrementalCausalLanguageModel<double>
        {
            private int _lastToken;

            public int VocabularySize => Vocab;

            public int ResetCalls { get; private set; }

            public int StartCalls { get; private set; }

            public List<int> AppendedTokens { get; } = new();

            public void ResetCache()
            {
                ResetCalls++;
                _lastToken = 0;
            }

            public Vector<double> StartSequence(IReadOnlyList<int> promptTokenIds)
            {
                StartCalls++;
                _lastToken = promptTokenIds[promptTokenIds.Count - 1];
                return new Vector<double>(LogitsFor(_lastToken));
            }

            public Vector<double> AppendToken(int tokenId)
            {
                AppendedTokens.Add(tokenId);
                _lastToken = tokenId;
                return new Vector<double>(LogitsFor(_lastToken));
            }

            // Full-refeed fallback (unused when the engine takes the incremental path).
            public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds) =>
                new(LogitsFor(tokenIds[tokenIds.Count - 1]));
        }

        // Same next-token rule, but no KV-cache: forces the engine's full-context path.
        private sealed class FullRefeedModel : ICausalLanguageModel<double>
        {
            public int VocabularySize => Vocab;

            public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds) =>
                new(LogitsFor(tokenIds[tokenIds.Count - 1]));
        }

        private sealed class MapTokenizer : IGenerationTokenizer
        {
            private readonly Dictionary<string, int> _wordToId = new(StringComparer.Ordinal);
            private readonly Dictionary<int, string> _idToWord = new();

            public MapTokenizer(int eos, params (string Word, int Id)[] entries)
            {
                EosTokenId = eos;
                _idToWord[eos] = string.Empty;
                foreach (var (word, id) in entries)
                {
                    _wordToId[word] = id;
                    _idToWord[id] = word;
                }
            }

            public int EosTokenId { get; }

            public IReadOnlyList<int> Encode(string text)
            {
                var ids = text
                    .Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(w => _wordToId.TryGetValue(w, out var id) ? id : EosTokenId)
                    .ToList();
                if (ids.Count == 0)
                {
                    ids.Add(EosTokenId);
                }

                return ids;
            }

            public string Decode(IReadOnlyList<int> tokenIds) =>
                string.Concat(tokenIds.Select(id => _idToWord.TryGetValue(id, out var w) ? w : string.Empty));
        }
    }
}
