using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VoiceFlow trains and samples as its paper specifies (Guo et al. 2024): Grad-TTS's encoder on forced-alignment
/// durations, a U-Net vector field trained with conditional flow matching towards x₁ − x₀, Euler sampling, and flow
/// rectification on generated (noise, mel) pairs.
/// </summary>
/// <remarks>
/// Before this change VoiceFlow had no vector field estimator and no flow matching: synthesis moved a scalar per frame
/// with a hand-written update.
/// </remarks>
public class VoiceFlowPaperTests
{
    private const int MelBins = 8;

    private static VoiceFlow<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new VoiceFlowOptions
        {
            HiddenDim = 16, EncoderDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, PrenetDropout = 0.0,
            DropoutRate = 0.0, DurationPredictorFilterChannels = 16, FlowDim = 8, MelChannels = MelBins, SegmentFrames = 12,
            NumFlowSteps = 3,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static TtsTrainingSample<double> Sample()
    {
        var tokens = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < 4; i++) tokens[i] = 10 + i * 9;
        var mel = new Tensor<double>(new[] { 12, MelBins });
        for (int f = 0; f < 12; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return new TtsTrainingSample<double> { Tokens = tokens, Mel = mel, Durations = new[] { 3, 3, 3, 3 } };
    }

    [Fact(Timeout = 240000)]
    public async Task Training_ReducesTheFlowMatchingAndDurationObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 40; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 240000)]
    public async Task Rectification_TrainsOnTheModelsOwnNoiseMelPairs()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        for (int i = 0; i < 5; i++) model.Train(sample);
        var (noise, generated) = model.GenerateRectificationPair(sample);
        Assert.Equal(new[] { 12, MelBins }, noise.Shape.ToArray());
        Assert.Equal(new[] { 12, MelBins }, generated.Shape.ToArray());
        var rectified = new TtsTrainingSample<double> { Tokens = sample.Tokens, Mel = generated, Durations = sample.Durations };
        double first = Convert.ToDouble(model.TrainRectified(rectified, noise));
        double last = first;
        for (int i = 0; i < 20; i++) last = Convert.ToDouble(model.TrainRectified(rectified, noise));
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(first) && AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(last));
    }

    [Fact(Timeout = 120000)]
    public async Task Sampling_IsRepeatable_AndMelShaped()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.Synthesize("hello");
        Assert.Equal(mel.ToVector().ToArray(), model.Synthesize("hello").ToVector().ToArray());
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        for (int i = 0; i < mel.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(mel[i]));
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseThePaperTrainsOnForcedAlignments()
    {
        await Task.Yield();
        var sample = Sample();
        Assert.Throws<NotSupportedException>(() => CreateModel().Train(sample.Tokens, sample.Mel!));
    }
}
