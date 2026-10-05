using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// ProDiff trains and samples as its paper specifies (Huang et al. 2022): FastSpeech 2's encoder and variance adaptor,
/// a WaveNet denoiser that predicts the clean spectrogram, a 4-step generator-based teacher, distillation of two teacher
/// DDIM steps into one student step, and posterior sampling.
/// </summary>
/// <remarks>
/// Before this change ProDiff had no denoiser and no diffusion: synthesis derived durations from the encoder output
/// with a fixed formula and moved a scalar per frame toward it with a hand-written update.
/// </remarks>
public class ProDiffPaperTests
{
    private const int MelBins = 8;

    private static ProDiff<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new ProDiffOptions
        {
            EncoderDim = 16, HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FftFilterSize = 32, PrenetLayers = 1,
            DropoutRate = 0.0, VariancePredictorFilterSize = 16, VariancePredictorDropout = 0.0, NumPitchBins = 16,
            NumEnergyBins = 16, DenoiserLayers = 2, DenoiserChannels = 16, MelChannels = MelBins, FftSize = 64,
            HopSize = 16, SampleRate = 8000,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static TtsTrainingSample<double> Sample()
    {
        var tokens = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < 4; i++) tokens[i] = 10 + i * 9;
        const int frames = 12;
        var mel = new Tensor<double>(new[] { frames, MelBins });
        var pitch = new double[frames];
        var energy = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
            pitch[f] = 150 + 30 * Math.Sin(0.5 * f);
            energy[f] = 1 + 0.5 * Math.Cos(0.3 * f);
        }
        return new TtsTrainingSample<double> { Tokens = tokens, Mel = mel, Durations = new[] { 3, 3, 3, 3 }, Pitch = pitch, Energy = energy };
    }

    [Fact(Timeout = 240000)]
    public async Task TeacherTraining_ReducesTheReconstructionSsimAndVarianceObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 240000)]
    public async Task Distillation_TrainsTheStudentAgainstTheFrozenTeacher()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        for (int i = 0; i < 5; i++) model.Train(sample);
        long parameters = model.ParameterCount;
        model.BeginDistillation();
        Assert.Equal(ProDiffTrainingPhase.Distillation, model.CurrentPhase);
        // The frozen teacher is not a trainable part of the student.
        Assert.Equal(parameters, model.ParameterCount);

        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Distillation objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Sampling_IsRepeatable_AndTheDistilledStepCountChangesTheSample()
    {
        await Task.Yield();
        var model = CreateModel();
        var teacherSample = model.Synthesize("ab");
        Assert.Equal(teacherSample.ToVector().ToArray(), model.Synthesize("ab").ToVector().ToArray());
        Assert.Equal(2, teacherSample.Rank);
        Assert.Equal(MelBins, teacherSample.Shape[1]);
        for (int i = 0; i < teacherSample.Length; i++) Assert.True(double.IsFinite(teacherSample[i]));

        model.BeginDistillation();
        var studentSample = model.Synthesize("ab");
        Assert.Equal(teacherSample.Shape, studentSample.Shape);
        Assert.True(Enumerable.Range(0, studentSample.Length).Any(i => Math.Abs(studentSample[i] - teacherSample[i]) > 1e-12),
            "Two student steps reproduced the four teacher steps exactly.");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseThePaperTrainsOnAlignedVariances()
    {
        await Task.Yield();
        var sample = Sample();
        Assert.Throws<NotSupportedException>(() => CreateModel().Train(sample.Tokens, sample.Mel!));
    }
}
