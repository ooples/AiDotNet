using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.MusicGen;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A text-to-speech model its paper trains on <see cref="TtsTrainingSample{T}"/> supervision trains through
/// <c>AiModelBuilder.Build(samples)</c>: the facade runs the epochs on the model's own objective, holds samples out,
/// and reports the per-epoch losses on the result.
/// </summary>
/// <remarks>
/// FastSpeech 2 is the subject because it refuses token/mel training (it needs forced-alignment durations), so the
/// ordinary <c>Build(features, labels)</c> cannot train it; before <c>Build(samples)</c> the only route was calling
/// <c>Train(TtsTrainingSample)</c> on the model, past the facade.
/// </remarks>
public class SampleObjectiveFacadeTrainingTests
{
    private static FastSpeech2<double> CreateModel() =>
        new FastSpeech2<double>(Architecture(), SmallOptions(), Adam());

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 16, outputSize: 80);

    // Plain Adam at 1e-3: the paper's Noam schedule warms up over 4000 steps and would barely move in a test.
    private static AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>> Adam() => new(null);

    private static FastSpeech2Options SmallOptions()
    {
        return new FastSpeech2Options
        {
            EncoderDim = 32,
            HiddenDim = 32,
            NumHeads = 2,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            FftFilterSize = 64,
            VariancePredictorFilterSize = 32,
            VariancePredictorDropout = 0.0,
            DropoutRate = 0.0,
            // The energy target is raw frame energy whose MSE needs thousands of steps; ablated as in FastSpeech2PaperTests.
            UseEnergyPredictor = false,
        };
    }

    /// <summary>
    /// FastSpeech 2 whose held-out objective follows a script, one value per epoch, and which records its parameters
    /// at each epoch's evaluation (after that epoch's training steps).
    /// </summary>
    private sealed class ScriptedObjectiveFastSpeech2 : FastSpeech2<double>
    {
        private readonly double[] _script;
        private int _evaluations;

        public ScriptedObjectiveFastSpeech2(double[] script)
            : base(Architecture(), SmallOptions(), Adam()) => _script = script;

        public List<Vector<double>> ParametersAtEvaluation { get; } = new();

        public override double EvaluateTrainingObjective(TtsTrainingSample<double> sample)
        {
            ParametersAtEvaluation.Add(GetParameters().Clone());
            return _script[_evaluations++];
        }
    }

    [Theory(Timeout = 180000)]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Build_EarlyStopping_ReturnsTheBestHeldOutEpochsModel(bool restoreBestModel)
    {
        await Task.Yield();
        // Ten samples hold one out, so each epoch evaluates once. The objective is best at epoch 2 (1.0) and does
        // not improve for the next two, so patience 2 stops after epoch 4.
        var model = new ScriptedObjectiveFastSpeech2(new[] { 3.0, 1.0, 2.0, 2.0, 2.0, 2.0 });
        var samples = Enumerable.Range(0, 10).Select(Sample).ToList();

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureSampleObjectiveTraining(new SampleObjectiveTrainingOptions
            {
                Epochs = 6,
                EarlyStoppingPatience = 2,
                RestoreBestModel = restoreBestModel,
            })
            .BuildAsync(samples);

        Assert.Equal(new[] { 3.0, 1.0, 2.0, 2.0 }, result.GetTrainingMetricsHistory()!["validation_loss"]);
        Assert.Equal(4, model.ParametersAtEvaluation.Count);
        var best = model.ParametersAtEvaluation[1].ToArray();
        var last = model.ParametersAtEvaluation[3].ToArray();
        Assert.NotEqual(best, last);   // training moved the parameters, so the check below can tell them apart
        Assert.Equal(restoreBestModel ? best : last, model.GetParameters().ToArray());
        Assert.Same(model, result.Model);
    }

    private static Tensor<double> Recording(double baseHz)
    {
        const int fs = 22050;
        const double seconds = 0.3;
        int n = (int)(fs * seconds);
        var x = new Tensor<double>(new[] { n });
        double phase = 0;
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / fs;
            phase += 2 * Math.PI * (baseHz + 100.0 * t / seconds) / fs;
            double v = 0;
            for (int h = 1; h < 6; h++) v += 0.3 * Math.Sin(h * phase) / h;
            x[i] = v;
        }
        return x;
    }

    private static TtsTrainingSample<double> Sample(int seed)
    {
        var audio = Recording(110.0 + 10 * seed);
        int frames = new TacotronSpectrogram().FrameCount(audio.Length);
        const int count = 8;
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7 + seed * 3) % 60;
        return new TtsTrainingSample<double>
        {
            Tokens = tokens,
            Audio = audio,
            Durations = Enumerable.Range(0, count).Select(i => frames / count + (i < frames % count ? 1 : 0)).ToArray(),
        };
    }

    [Fact(Timeout = 180000)]
    public async Task Build_TrainsASampleObjectiveModel_WithFallingLoss()
    {
        await Task.Yield();
        var model = CreateModel();
        var samples = new List<TtsTrainingSample<double>> { Sample(0) };

        var result = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureSampleObjectiveTraining(new SampleObjectiveTrainingOptions { Epochs = 30 })
            .Build(samples);

        var history = result.GetTrainingMetricsHistory();
        Assert.NotNull(history);
        var losses = history!["train_loss"];
        Assert.Equal(30, losses.Count);
        Assert.All(losses, loss => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(loss), $"Non-finite loss {loss}."));
        Assert.True(losses[losses.Count - 1] < losses[0] * 0.5,
            $"The FastSpeech 2 objective did not halve through the facade: {losses[0]} -> {losses[losses.Count - 1]}.");
        Assert.False(history.ContainsKey("validation_loss"), "One sample cannot be split; nothing should be held out.");
    }

    [Fact(Timeout = 180000)]
    public async Task Build_WithDefaults_HoldsSamplesOutAndMeasuresThem()
    {
        await Task.Yield();
        var model = CreateModel();
        var samples = Enumerable.Range(0, 10).Select(Sample).ToList();

        // Only the epoch count changes; the held-out fraction, shuffling and early stopping keep their defaults.
        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureSampleObjectiveTraining(new SampleObjectiveTrainingOptions { Epochs = 3 })
            .BuildAsync(samples);

        var history = result.GetTrainingMetricsHistory();
        Assert.NotNull(history);
        Assert.Equal(3, history!["train_loss"].Count);
        Assert.Equal(3, history["validation_loss"].Count);
        Assert.All(history["validation_loss"], loss => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(loss)));
        Assert.True(history["validation_loss"][2] < history["validation_loss"][0],
            $"The held-out objective did not fall: {string.Join(", ", history["validation_loss"])}.");
        Assert.Same(model, result.Model);
    }

    [Fact(Timeout = 60000)]
    public async Task Build_RejectsSamplesTheModelDoesNotTrainOn()
    {
        await Task.Yield();
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>().ConfigureModel(CreateModel());
        var samples = new List<MusicGenTrainingSample<double>> { new() { Audio = new Tensor<double>(new[] { 16 }) } };

        var error = Assert.Throws<InvalidOperationException>(() => builder.Build(samples));
        Assert.Contains("MusicGenTrainingSample", error.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task ConfigureSampleObjectiveTraining_RejectsOutOfRangeSettings()
    {
        await Task.Yield();
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            builder.ConfigureSampleObjectiveTraining(new SampleObjectiveTrainingOptions { Epochs = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            builder.ConfigureSampleObjectiveTraining(new SampleObjectiveTrainingOptions { ValidationFraction = 1.0 }));
    }
}
