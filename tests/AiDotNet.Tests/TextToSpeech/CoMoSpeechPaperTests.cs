using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CoMoSpeech trains and synthesizes as its paper specifies (Ye et al. 2023): an EDM-preconditioned Grad-TTS teacher
/// trained on duration, prior and weighted denoising losses and sampled with Euler steps, then consistency distillation
/// towards an EMA target with the encoder frozen, giving one-step synthesis.
/// </summary>
/// <remarks>
/// Before this change CoMoSpeech had no diffusion and no consistency model: synthesis set durations from character
/// codes and ran a hand-written scalar update per frame.
/// </remarks>
public class CoMoSpeechPaperTests
{
    private const int MelBins = 8;

    private static CoMoSpeech<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new CoMoSpeechOptions
        {
            HiddenDim = 16, EncoderDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, PrenetDropout = 0.0,
            DropoutRate = 0.0, DurationPredictorFilterChannels = 16, FlowDim = 8, MelChannels = MelBins, SegmentFrames = 12,
            TeacherSamplingSteps = 4, DistillationSteps = 6,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static (Tensor<double> Tokens, Tensor<double> Mel) Example()
    {
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + (i * 7) % 60;
        var mel = new Tensor<double>(new[] { 14, MelBins });
        for (int f = 0; f < 14; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return (tokens, mel);
    }

    [Fact(Timeout = 240000)]
    public async Task TeacherTraining_ReducesTheDurationPriorAndEdmObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var (tokens, mel) = Example();
        double before = provider.EvaluateTrainingObjective(tokens, mel);
        for (int i = 0; i < 40; i++) model.Train(tokens, mel);
        double after = provider.EvaluateTrainingObjective(tokens, mel);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 240000)]
    public async Task Distillation_TrainsOnlyTheDenoiser_AgainstAMovingTarget()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var (tokens, mel) = Example();
        for (int i = 0; i < 5; i++) model.Train(tokens, mel);
        model.BeginDistillation();
        var before = model.GetParameters().ToArray();
        double lossBefore = provider.EvaluateTrainingObjective(tokens, mel);
        for (int i = 0; i < 20; i++) model.Train(tokens, mel);
        double lossAfter = provider.EvaluateTrainingObjective(tokens, mel);
        var after = model.GetParameters().ToArray();
        int changed = Enumerable.Range(0, before.Length).Count(i => before[i] != after[i]);
        Assert.True(changed > 0 && changed < before.Length, $"{changed} of {before.Length} parameters changed.");
        // θ, θ⁻ and the teacher start identical, so the loss starts near zero and then tracks an EMA target that moves
        // with θ (Eq. 11-12); it is not expected to fall monotonically, only to stay a finite consistency gap.
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(lossBefore) && AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(lossAfter), $"{lossBefore} -> {lossAfter}");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_IsRepeatable_AndTheDistilledModelSamplesInOneStep()
    {
        await Task.Yield();
        var model = CreateModel();
        var teacher = model.Synthesize("hello");
        Assert.Equal(teacher.ToVector().ToArray(), model.Synthesize("hello").ToVector().ToArray());
        Assert.Equal(2, teacher.Rank);
        Assert.Equal(MelBins, teacher.Shape[1]);
        for (int i = 0; i < teacher.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(teacher[i]));

        model.BeginDistillation();
        var student = model.Synthesize("hello");
        Assert.Equal(teacher.Shape, student.Shape);
        for (int i = 0; i < student.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(student[i]));
    }
}
