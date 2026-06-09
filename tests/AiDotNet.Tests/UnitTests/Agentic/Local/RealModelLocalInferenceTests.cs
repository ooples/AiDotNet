using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    /// <summary>
    /// Marks the real-network inference tests as a non-parallel collection. Running a real neural-network
    /// forward pass mutates process-wide tensor-engine state (engine selection / LazyTensorScope), which can
    /// disturb tensor ops in tests running concurrently; isolating this collection avoids that.
    /// </summary>
    [CollectionDefinition("RealModelInference", DisableParallelization = true)]
    public sealed class RealModelInferenceCollection
    {
    }

    /// <summary>
    /// Integration tests that run the local generation engine over a real (tiny, untrained) AiDotNet
    /// <see cref="MambaLanguageModel{T}"/>. They validate the full in-process pipeline — one-hot encoding,
    /// forward pass, last-position logit extraction, sampling, autoregressive loop — not output quality.
    /// All use greedy decoding (temperature 0) for deterministic assertions.
    /// </summary>
    [Collection("RealModelInference")]
    public class RealModelLocalInferenceTests
    {
        private const int Vocab = 20;
        private const int MaxSeq = 16;

        // A real network forward dispatches through AiDotNetEngine.Current, which a module initializer may
        // auto-set to a GPU engine; some GPU backends return zeros for small shapes. Pin a CPU engine for the
        // duration of each test (documented mitigation) and restore it afterwards.
        private static async Task PinCpuAsync(Func<Task> body)
        {
            var priorEngine = AiDotNetEngine.Current;
            AiDotNetEngine.Current = new CpuEngine();
            try
            {
                await body();
            }
            finally
            {
                AiDotNetEngine.Current = priorEngine;
            }
        }

        private static MambaLanguageModel<double> BuildTinyModel()
        {
            var architecture = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional,
                NeuralNetworkTaskType.TextGeneration,
                inputSize: Vocab,
                outputSize: Vocab);

            return new MambaLanguageModel<double>(
                architecture,
                vocabSize: Vocab,
                modelDimension: 16,
                numLayers: 1,
                stateDimension: 4,
                maxSeqLength: MaxSeq);
        }

        [Fact(Timeout = 120000)]
        public async Task Adapter_NextTokenLogits_ReturnsVocabWidthLogits()
        {
            await PinCpuAsync(() =>
            {
                var lm = new NeuralNetworkCausalLanguageModel<double>(BuildTinyModel(), Vocab, maxContextTokens: MaxSeq);

                var logits = lm.NextTokenLogits(new[] { 1, 2, 3 });

                Assert.Equal(Vocab, logits.Length);
                for (var i = 0; i < logits.Length; i++)
                {
                    Assert.False(double.IsNaN(logits[i]));
                }

                return Task.CompletedTask;
            });
        }

        [Fact(Timeout = 120000)]
        public async Task Engine_GeneratesEndToEnd_OverRealNetwork()
        {
            await PinCpuAsync(async () =>
            {
                var lm = new NeuralNetworkCausalLanguageModel<double>(BuildTinyModel(), Vocab, maxContextTokens: MaxSeq);
                var tokenizer = new BoundedTokenizer(Vocab);
                var engine = new LocalEngineChatClient<double>(lm, tokenizer);

                var response = await engine.GetResponseAsync(
                    new[] { ChatMessage.User("hello there") },
                    new ChatOptions { Temperature = 0.0, MaxOutputTokens = 3 });

                // The untrained model produces arbitrary tokens, but the full pipeline must run cleanly and
                // honor the token budget.
                Assert.Equal(ChatFinishReason.Length, response.FinishReason);
                Assert.NotNull(response.Usage);
                Assert.Equal(3, response.Usage.OutputTokens);
                Assert.NotNull(response.Text);
            });
        }

        [Fact(Timeout = 120000)]
        public async Task Engine_StreamingOverRealNetwork_TerminatesWithFinish()
        {
            await PinCpuAsync(async () =>
            {
                var lm = new NeuralNetworkCausalLanguageModel<double>(BuildTinyModel(), Vocab, maxContextTokens: MaxSeq);
                var engine = new LocalEngineChatClient<double>(lm, new BoundedTokenizer(Vocab));

                ChatFinishReason? finish = null;
                var updates = 0;
                await foreach (var update in engine.GetStreamingResponseAsync(
                    new[] { ChatMessage.User("hi") },
                    new ChatOptions { Temperature = 0.0, MaxOutputTokens = 2 }))
                {
                    updates++;
                    if (update.FinishReason is { } reason)
                    {
                        finish = reason;
                    }
                }

                Assert.True(updates > 0);
                Assert.Equal(ChatFinishReason.Length, finish);
            });
        }

        // A whitespace tokenizer whose ids stay within the model's vocabulary. EOS is disabled (-1) so
        // greedy generation runs deterministically to the token budget regardless of the untrained model.
        private sealed class BoundedTokenizer : IGenerationTokenizer
        {
            private readonly int _vocab;

            public BoundedTokenizer(int vocab)
            {
                _vocab = vocab;
            }

            public int EosTokenId => -1;

            public IReadOnlyList<int> Encode(string text)
            {
                var ids = new List<int>();
                foreach (var word in text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    // Deterministic, bounded id from a stable char-sum (string.GetHashCode is randomized per process).
                    var sum = 0;
                    foreach (var ch in word)
                    {
                        sum = (sum + ch) & 0x7fffffff;
                    }

                    ids.Add((sum % (_vocab - 1)) + 1);
                }

                if (ids.Count == 0)
                {
                    ids.Add(1);
                }

                return ids;
            }

            public string Decode(IReadOnlyList<int> tokenIds) =>
                string.Join(" ", tokenIds.Select(id => "t" + id));
        }
    }
}
