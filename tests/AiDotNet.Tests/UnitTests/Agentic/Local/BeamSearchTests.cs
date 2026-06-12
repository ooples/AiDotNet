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
    public class BeamSearchTests
    {
        // Vocabulary: 0=EOS, 1=A, 2=B, 3=C. The model's next-token logits depend on the last token, set up
        // so that the locally-best first token (A) leads to a worse overall sequence than B -> C.
        private static TransitionModel BuildModel() => new(
            vocabularySize: 4,
            transitions: new Dictionary<int, double[]>
            {
                [0] = new[] { 0.1, 2.0, 1.9, 0.1 }, // start (prompt's last token): A slightly beats B
                [1] = new[] { 2.0, 0.1, 0.1, 0.1 }, // after A: EOS (A is a short, lower-total path)
                [2] = new[] { 0.1, 0.1, 0.1, 5.0 }, // after B: C is almost certain
                [3] = new[] { 5.0, 0.1, 0.1, 0.1 }, // after C: EOS
            });

        private static MapTokenizer BuildTokenizer() =>
            new(eos: 0, ("A", 1), ("B", 2), ("C", 3));

        [Fact(Timeout = 60000)]
        public async Task Greedy_TakesLocallyBestToken()
        {
            var engine = new LocalEngineChatClient<double>(BuildModel(), BuildTokenizer(), options: new LocalEngineOptions
            {
                Sampling = new LocalSamplingOptions { Temperature = 0.0 }, // greedy, beam width 1
            });

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("go") });

            Assert.Equal("A", response.Text);
        }

        [Fact(Timeout = 60000)]
        public async Task BeamSearch_FindsHigherProbabilitySequence()
        {
            var engine = new LocalEngineChatClient<double>(BuildModel(), BuildTokenizer(), options: new LocalEngineOptions
            {
                BeamWidth = 2,
            });

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("go") });

            // Beam search explores B as well as A and discovers the globally better B -> C completion.
            Assert.Equal("BC", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        [Fact(Timeout = 60000)]
        public async Task BeamSearch_RespectsConstraint()
        {
            // Force the grammar A -> B -> (terminal) even though the model would prefer other paths.
            var constraint = new FiniteStateTokenConstraint(
                start: new[] { 1 },
                transitions: new Dictionary<int, IReadOnlyCollection<int>>
                {
                    [1] = new[] { 2 },
                    [2] = Array.Empty<int>(),
                });

            var engine = new LocalEngineChatClient<double>(BuildModel(), BuildTokenizer(), options: new LocalEngineOptions
            {
                BeamWidth = 3,
                Constraint = constraint,
            });

            var response = await engine.GetResponseAsync(new[] { ChatMessage.User("go") });

            Assert.Equal("AB", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        // ---- Test doubles ----

        private sealed class TransitionModel : ICausalLanguageModel<double>
        {
            private readonly Dictionary<int, double[]> _transitions;
            private readonly double[] _default;

            public TransitionModel(int vocabularySize, Dictionary<int, double[]> transitions)
            {
                VocabularySize = vocabularySize;
                _transitions = transitions;
                _default = new double[vocabularySize];
                _default[0] = 5.0; // unknown state -> prefer EOS
            }

            public int VocabularySize { get; }

            public Vector<double> NextTokenLogits(IReadOnlyList<int> tokenIds)
            {
                var last = tokenIds[tokenIds.Count - 1];
                var logits = _transitions.TryGetValue(last, out var row) ? row : _default;
                return new Vector<double>((double[])logits.Clone());
            }
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
