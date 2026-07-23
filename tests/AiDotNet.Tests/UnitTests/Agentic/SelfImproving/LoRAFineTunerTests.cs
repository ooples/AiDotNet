using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Agentic.SelfImproving;
using AiDotNet.Enums;
using AiDotNet.FineTuning;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
    // The end-to-end tensor-training test runs a real Mamba forward/backward, which dispatches through the
    // process-wide AiDotNetEngine.Current. Other tests mutate that global in parallel, so join the serialized
    // RealModelInference collection to keep the CpuEngine pin stable for the duration of training.
    [Collection("RealModelInference")]
    public class LoRAFineTunerTests
    {
        private static FineTuningDataset Dataset()
        {
            var store = new InMemoryTrajectoryStore();
            // Build directly: two graded prompt/answer trajectories.
            var trajectories = new[]
            {
                new AgentTrajectory("1", "a", new List<ChatMessage> { ChatMessage.User("2+2?"), ChatMessage.Assistant("4") }, "4", 1, reward: 1.0),
                new AgentTrajectory("2", "a", new List<ChatMessage> { ChatMessage.User("3+3?"), ChatMessage.Assistant("6") }, "6", 1, reward: 0.8),
            };
            return new RewardFilteredDatasetBuilder(0.5).Build(trajectories);
        }

        [Fact(Timeout = 60000)]
        public async Task Converter_MapsPromptsToInputs_CompletionsToOutputs_RewardsToWeights()
        {
            await Task.Yield();

            var data = FineTuningDataConverter.ToSupervisedData<double>(Dataset());

            Assert.Equal(2, data.Count);
            Assert.True(data.HasSFTData);
            Assert.Contains("4", data.Outputs);
            Assert.Contains("6", data.Outputs);
            Assert.Contains(data.Inputs, i => i.Contains("2+2?"));
            Assert.Equal(new[] { 1.0, 0.8 }, data.SampleWeights);
        }

        [Fact(Timeout = 60000)]
        public async Task Runner_ConvertsAndInvokesFineTuner()
        {
            await Task.Yield();

            var tuner = new RecordingFineTuner();
            var model = new StubStringModel();

            var result = await LoRAFineTuner.FineTuneFromDatasetAsync<double>(tuner, model, Dataset());

            Assert.Same(model, result);                         // fine-tuner returned the (fine-tuned) model
            Assert.NotNull(tuner.LastData);
            Assert.Equal(2, tuner.LastData?.Count);             // converted data reached the fine-tuner
            Assert.True(tuner.LastData?.HasSFTData == true);
        }

        [Fact(Timeout = 180000)]
        public async Task TensorBridge_EndToEnd_ReducesLoss_OnRealMamba()
        {
            await Task.Yield();

            // End-to-end CPU-verifiable self-improvement: tokenize a reward-filtered dataset into next-token
            // tensor supervision and actually fine-tune a real (tiny) Mamba model. Training must reduce the
            // dataset loss — proving the loop runs, not just that the bridge forwards data.
            const int vocab = 16;
            const int seqLen = 6;

            var priorEngine = AiDotNetEngine.Current;
            AiDotNetEngine.Current = new CpuEngine();
            try
            {
                var arch = new NeuralNetworkArchitecture<double>(
                    InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration, inputSize: vocab, outputSize: vocab);
                var model = new MambaLanguageModel<double>(
                    arch, vocabSize: vocab, modelDimension: 16, numLayers: 1, stateDimension: 4, maxSeqLength: seqLen);

                // A small, repetitive dataset so the tiny model has a clear pattern to fit.
                var dataset = new FineTuningDataset(new[]
                {
                    new FineTuningExample("two plus two", "is four", 1.0),
                    new FineTuningExample("three plus three", "is six", 1.0),
                    new FineTuningExample("two plus two", "is four", 0.9),
                    new FineTuningExample("three plus three", "is six", 0.9),
                });

                var tokenizer = new BoundedTokenizer(vocab);

                // Train one epoch and record the early loss, then train many more and record the late loss.
                LoRAFineTuner.TrainTensorModelOnDataset<double>(model, tokenizer, vocab, seqLen, dataset, epochs: 1);
                var earlyLoss = Convert.ToDouble(model.GetLastLoss());

                LoRAFineTuner.TrainTensorModelOnDataset<double>(model, tokenizer, vocab, seqLen, dataset, epochs: 60);
                var lateLoss = Convert.ToDouble(model.GetLastLoss());

                Assert.True(lateLoss < earlyLoss,
                    $"Training did not reduce loss: early={earlyLoss:G6}, late={lateLoss:G6}");
            }
            finally
            {
                AiDotNetEngine.Current = priorEngine;
            }
        }

        // Deterministic whitespace tokenizer whose ids stay within the model's vocabulary.
        private sealed class BoundedTokenizer : IGenerationTokenizer
        {
            private readonly int _vocab;

            public BoundedTokenizer(int vocab) => _vocab = vocab;

            public int EosTokenId => -1;

            public IReadOnlyList<int> Encode(string text)
            {
                var ids = new List<int>();
                foreach (var word in text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries))
                {
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

        // A fine-tuner that records the data it received and returns the model unchanged.
        private sealed class RecordingFineTuner : FineTuningBase<double, string, string>
        {
            public RecordingFineTuner() : base(new FineTuningOptions<double>()) { }

            public FineTuningData<double, string, string>? LastData { get; private set; }

            public override string MethodName => "recording";

            public override FineTuningCategory Category => FineTuningCategory.SupervisedFineTuning;

            public override bool RequiresRewardModel => false;

            public override bool RequiresReferenceModel => false;

            public override Task<IFullModel<double, string, string>> FineTuneAsync(
                IFullModel<double, string, string> baseModel,
                FineTuningData<double, string, string> trainingData,
                CancellationToken cancellationToken = default)
            {
                LastData = trainingData;
                return Task.FromResult(baseModel);
            }

            public override Task<FineTuningMetrics<double>> EvaluateAsync(
                IFullModel<double, string, string> model,
                FineTuningData<double, string, string> evaluationData,
                CancellationToken cancellationToken = default) =>
                Task.FromResult(new FineTuningMetrics<double>());
        }

        // Minimal IFullModel test double; the fine-tuner only stores and returns it, so members are unused.
        private sealed class StubStringModel : IFullModel<double, string, string>
        {
            public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

            public void Train(string input, string expectedOutput) => throw new NotSupportedException();

            public string Predict(string input) => string.Empty;

            public ModelMetadata<double> GetModelMetadata() => new();

            public byte[] Serialize() => Array.Empty<byte>();

            public void Deserialize(byte[] data) => throw new NotSupportedException();

            public void SaveModel(string filePath) => throw new NotSupportedException();

            public void LoadModel(string filePath) => throw new NotSupportedException();

            public void SaveState(Stream stream) => throw new NotSupportedException();

            public void LoadState(Stream stream) => throw new NotSupportedException();

            public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();

            public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => throw new NotSupportedException();

            public bool IsFeatureUsed(int featureIndex) => false;

            public Dictionary<string, double> GetFeatureImportance() => new();

            public IFullModel<double, string, string> DeepCopy() => this;

            public IFullModel<double, string, string> Clone() => this;

            public void Dispose()
            {
            }
        }
    }
}
