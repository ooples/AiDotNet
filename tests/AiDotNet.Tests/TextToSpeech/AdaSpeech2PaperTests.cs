using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// AdaSpeech 2 follows its paper's adaptation pipeline (Yan et al. 2021, §2): source training, mel encoder aligning
/// with only the mel encoder trainable, and untranscribed adaptation that reconstructs speech through the mel encoder
/// and decoder while adapting only the conditional layer normalizations.
/// </summary>
/// <remarks>
/// Before this change AdaSpeech 2 had no mel encoder and no untranscribed path: its synthesis averaged encoder values
/// into a "mel-to-phoneme" scalar, computed durations from a fixed formula, and threw on every call because its
/// encoder boundary pointed past the layer stack.
/// </remarks>
public class AdaSpeech2PaperTests
{
    private const int MelBins = 8;

    private static AdaSpeech2<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new AdaSpeech2Options
        {
            EncoderDim = 16, HiddenDim = 16, MelChannels = MelBins, NumHeads = 2, NumEncoderLayers = 1,
            NumDecoderLayers = 1, NumMelEncoderLayers = 1, FftFilterSize = 32, VariancePredictorFilterSize = 16,
            VariancePredictorDropout = 0.0, AcousticConditionFilterSize = 16, AcousticConditionDropout = 0.0,
            DropoutRate = 0.0, NumSpeakers = 3, LearningRate = 1e-3,
        });

    private static Tensor<double> Mel(int frames, double phase)
    {
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c + phase);
        return mel;
    }

    private static double[] Pitch(int frames) => Enumerable.Range(0, frames).Select(f => 140.0 + 30.0 * Math.Sin(0.5 * f)).ToArray();
    private static double[] Energy(int frames) => Enumerable.Range(0, frames).Select(f => 5.0 + Math.Cos(0.3 * f)).ToArray();

    private static TtsTrainingSample<double> Sample(int speaker)
    {
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + (i * 7) % 60;
        return new TtsTrainingSample<double>
        {
            Tokens = tokens, Mel = Mel(14, 0.0), Durations = new[] { 3, 3, 3, 3, 2 },
            Pitch = Pitch(14), Energy = Energy(14), SpeakerId = speaker,
        };
    }

    private static double[] Snapshot(IEnumerable<ILayer<double>> layers)
        => layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 120000)]
    public async Task MelEncoderAligning_TrainsOnlyTheMelEncoder_TowardThePhonemeHiddenSpace()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(speaker: 0);
        for (int i = 0; i < 5; i++) model.Train(sample);

        model.CurrentStep = AdaSpeech2TrainingStep.MelEncoderAligning;
        var melEncoderBefore = Snapshot(model.MelEncoderLayers);
        var allBefore = model.GetParameters().ToArray();
        double before = model.EvaluateTrainingObjective(sample);

        for (int i = 0; i < 25; i++) model.Train(sample);

        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Aligning loss did not fall ({before} -> {after}).");
        var allAfter = model.GetParameters().ToArray();
        int changed = allBefore.Zip(allAfter, (a, b) => a != b).Count(c => c);
        Assert.True(changed > 0 && changed <= melEncoderBefore.Length,
            $"Aligning changed {changed} parameters; only the mel encoder's {melEncoderBefore.Length} may move.");
        Assert.NotEqual(melEncoderBefore, Snapshot(model.MelEncoderLayers));
    }

    [Fact(Timeout = 120000)]
    public async Task UntranscribedAdaptation_ReconstructsSpeech_AdaptingOnlyTheConditionalLayerNorms()
    {
        await Task.Yield();
        var model = CreateModel();
        for (int i = 0; i < 5; i++) model.Train(Sample(speaker: 0));

        var speech = Mel(12, 0.9);
        Assert.Throws<InvalidOperationException>(() => model.TrainUntranscribed(speech, 2, Pitch(12), Energy(12)));

        model.CurrentStep = AdaSpeech2TrainingStep.UntranscribedAdaptation;
        // Evaluate first: source training never ran the mel encoder, so its lazy input projection materializes here.
        double before = model.EvaluateUntranscribed(speech, 2, Pitch(12), Energy(12));
        var melEncoderBefore = Snapshot(model.MelEncoderLayers);
        var allBefore = model.GetParameters().ToArray();

        for (int i = 0; i < 25; i++) model.TrainUntranscribed(speech, 2, Pitch(12), Energy(12));

        double after = model.EvaluateUntranscribed(speech, 2, Pitch(12), Energy(12));
        Assert.True(after < before, $"Reconstruction loss did not fall ({before} -> {after}).");
        Assert.Equal(melEncoderBefore, Snapshot(model.MelEncoderLayers));

        // Speaker embedding (3 x 16) plus W_gamma and W_beta (16 x 16) of each of the C = 2L + 1 = 3 conditional norms.
        int adaptive = 3 * 16 + 3 * 2 * 16 * 16;
        var allAfter = model.GetParameters().ToArray();
        int changed = allBefore.Zip(allAfter, (a, b) => a != b).Count(c => c);
        Assert.True(changed > 0 && changed <= adaptive,
            $"Adaptation changed {changed} parameters; only the {adaptive} adaptive ones may move.");
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_UsesThePhonemeEncoderAndAdaptedDecoder_InTheGivenVoice()
    {
        await Task.Yield();
        var model = CreateModel();
        model.Voice = new TtsVoice<double> { SpeakerId = 1, Reference = Mel(12, 0.0) };
        var mel = model.Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        for (int i = 0; i < mel.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }
}
