using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.SelfImproving;
using AiDotNet.Enums;
using AiDotNet.FineTuning;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.SelfImproving
{
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
            var data = FineTuningDataConverter.ToSupervisedData<double>(Dataset());

            Assert.Equal(2, data.Count);
            Assert.True(data.HasSFTData);
            Assert.Contains("4", data.Outputs);
            Assert.Contains("6", data.Outputs);
            Assert.Contains(data.Inputs, i => i.Contains("2+2?"));
            Assert.Equal(new[] { 1.0, 0.8 }, data.SampleWeights);
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Runner_ConvertsAndInvokesFineTuner()
        {
            var tuner = new RecordingFineTuner();
            var model = new StubStringModel();

            var result = await LoRAFineTuner.FineTuneFromDatasetAsync<double>(tuner, model, Dataset());

            Assert.Same(model, result);                         // fine-tuner returned the (fine-tuned) model
            Assert.NotNull(tuner.LastData);
            Assert.Equal(2, tuner.LastData?.Count);             // converted data reached the fine-tuner
            Assert.True(tuner.LastData?.HasSFTData == true);
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
