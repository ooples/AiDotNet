using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NER.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NER.SpanBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NER;

/// <summary>
/// <c>INERModel.TrainAsync</c> must actually train (#2155).
/// </summary>
/// <remarks>
/// The async loop in TransformerNERBase, SpanBasedNERBase and LSTMCRF computed a detached loss and then
/// called the optimizer's UpdateParameters with no backward pass - the backward had been commented out
/// when manual backprop was removed - so every epoch stepped on stale, initially zero, gradients. It
/// reported a loss each epoch and changed no weight. No test called TrainAsync on an NER model, so
/// nothing noticed.
/// </remarks>
public class NERTrainAsyncTests
{
    [Fact(Timeout = 120000)]
    public async Task TrainAsync_ChangesTheWeightsAndReportsEveryEpoch()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32,
            outputSize: 9);
        var options = new BiaffineNEROptions
        {
            HiddenDimension = 32,
            NumAttentionHeads = 4,
            NumTransformerLayers = 2,
            IntermediateDimension = 64,
            NumLabels = 9,
            MaxSequenceLength = 12,
            MaxSpanLength = 3,
            SpanEmbeddingDimension = 32,
            DropoutRate = 0.0,
            LearningRate = 1e-3,
            BiLstmHiddenSize = 7,
            BiLstmLayers = 1,
            BiLstmDropout = 0.0,
            EmbeddingsDropout = 0.0,
        };
        using var model = new BiaffineNER<double>(architecture, options);

        var rng = RandomHelper.CreateSeededRandom(42);
        var tokens = new Tensor<double>(new[] { 8, 32 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = rng.NextDouble() * 2.0 - 1.0;
        var labels = new Tensor<double>(new[] { 8 });
        for (int i = 0; i < labels.Length; i++) labels[i] = rng.Next(9);

        // Some weights are sized from the first real input, so one inference forward runs before the
        // snapshot; Predict does not train. Without it the parameter count itself grows during
        // TrainAsync and the comparison is between vectors of different lengths.
        model.Predict(tokens);
        var before = model.GetParameters().Clone();

        int reports = 0;
        var progress = new SynchronousProgress(_ => Interlocked.Increment(ref reports));
        await ((INERModel<double>)model).TrainAsync(tokens, labels, epochs: 2, progress, CancellationToken.None);

        var after = model.GetParameters();
        Assert.Equal(before.Length, after.Length);
        double change = 0;
        for (int i = 0; i < after.Length; i++)
        {
            Assert.False(double.IsNaN(after[i]) || double.IsInfinity(after[i]), $"Parameter {i} is {after[i]}.");
            change += Math.Abs(after[i] - before[i]);
        }

        Assert.True(change > 0, "Two epochs of TrainAsync left every weight unchanged.");
        Assert.Equal(2, reports);
    }

    /// <summary>Reports on the calling thread, so the count is final when TrainAsync returns.</summary>
    private sealed class SynchronousProgress : IProgress<NERTrainingProgress>
    {
        private readonly Action<NERTrainingProgress> _report;

        public SynchronousProgress(Action<NERTrainingProgress> report) => _report = report;

        public void Report(NERTrainingProgress value) => _report(value);
    }
}
