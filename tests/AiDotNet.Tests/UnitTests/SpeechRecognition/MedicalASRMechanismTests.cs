using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SpeechRecognition;

/// <summary>
/// Verifies that <see cref="MedicalASR{T}"/> implements Chiu et al., "Speech recognition for medical
/// conversations" (Interspeech 2018, arXiv:1711.07274).
/// </summary>
/// <remarks>
/// That paper is a COMPARISON — "we explored both CTC and LAS systems" — whose conclusion is that
/// "the LAS was more resilient to noisy data and CTC required more data clean up". A CTC-only class
/// cannot express that finding, because the finding IS the difference between the two arms. These
/// tests pin both arms and the pyramidal reduction that makes the LAS arm what it is.
/// </remarks>
public class MedicalASRMechanismTests
{
    private static NeuralNetworkArchitecture<double> Arch() =>
        new(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 64, outputSize: 32);

    private static MedicalASROptions Small(Action<MedicalASROptions>? tweak = null)
    {
        var o = new MedicalASROptions
        {
            EncoderDim = 16,
            NumEncoderLayers = 1,
            NumAttentionHeads = 2,
            NumMels = 8,
            VocabSize = 24,
            DecoderDim = 16,
            NumDecoderLayers = 1,
            PyramidalReductions = 2,
            MaxTextLength = 16,
            DropoutRate = 0.0,
        };
        tweak?.Invoke(o);
        return o;
    }

    private static MedicalASR<double> Model(Action<MedicalASROptions>? tweak = null)
        => new(Arch(), Small(tweak));

    private static Tensor<double> Features(int frames = 8, int width = 8, int seed = 4)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([frames, width]);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    [Fact]
    public void BothArmsOfThePaperExist_AndLasIsTheDefault()
    {
        // "The LAS was more resilient to noisy data and CTC required more data clean up" — so LAS is
        // the default for spontaneous conversation, and CTC is the paper's other arm rather than a
        // lesser fallback.
        Assert.Equal(MedicalAsrDecoderType.ListenAttendSpell, new MedicalASROptions().DecoderType);
        Assert.Equal(MedicalAsrDecoderType.ListenAttendSpell, Model().DecoderType);
        Assert.Equal(MedicalAsrDecoderType.Ctc, Model(o => o.DecoderType = MedicalAsrDecoderType.Ctc).DecoderType);

        // Two members, no more: the paper compares exactly these two.
        Assert.Equal(2, Enum.GetValues<MedicalAsrDecoderType>().Length);
    }

    [Fact]
    public void LasListensPyramidally_HalvingTheSequencePerReduction()
    {
        // Without the reduction the speller would attend over every acoustic frame of a
        // consultation. This is the part that distinguishes a "listener" from an ordinary encoder.
        var model = Model(o => o.PyramidalReductions = 2);
        Assert.Equal(4, model.TimeReductionFactor);

        var encoded = Features(frames: 8, width: 16);
        var reduced = model.PyramidalReduce(encoded);

        Assert.Equal(2, reduced.Shape[0]);    // 8 -> 4 -> 2
        Assert.Equal(16, reduced.Shape[1]);   // width unchanged, so it can reduce repeatedly
    }

    [Fact]
    public void ReductionAveragesAdjacentFrames()
    {
        var model = Model(o => o.PyramidalReductions = 1);
        var input = new Tensor<double>([4, 2]);
        double[] values = [0, 10, 2, 12, 4, 14, 6, 16];
        for (int i = 0; i < values.Length; i++) input[i] = values[i];

        var reduced = model.PyramidalReduce(input);

        Assert.Equal(2, reduced.Shape[0]);
        Assert.Equal(1.0, reduced[0, 0], 12);    // mean(0, 2)
        Assert.Equal(11.0, reduced[0, 1], 12);   // mean(10, 12)
        Assert.Equal(5.0, reduced[1, 0], 12);    // mean(4, 6)
        Assert.Equal(15.0, reduced[1, 1], 12);   // mean(14, 16)
    }

    [Fact]
    public void CtcArmKeepsFullTimeResolution()
    {
        // CTC is frame-synchronous: reducing the time axis would destroy the alignment it depends on.
        var model = Model(o => { o.DecoderType = MedicalAsrDecoderType.Ctc; o.PyramidalReductions = 3; });

        Assert.Equal(1, model.TimeReductionFactor);
        var encoded = Features(frames: 8, width: 16);
        Assert.Equal(8, model.PyramidalReduce(encoded).Shape[0]);
    }

    [Fact]
    public void TheTwoArmsAreStructurallyDifferentModels()
    {
        // If both arms produced the same parameterization, only one of them would really exist.
        var las = Model();
        var ctc = Model(o => o.DecoderType = MedicalAsrDecoderType.Ctc);

        las.Predict(Features());
        ctc.Predict(Features());

        Assert.NotEqual(las.GetParameters().Length, ctc.GetParameters().Length);
    }

    [Fact]
    public void MoreReductionsGiveAShorterSummary()
    {
        var one = Model(o => o.PyramidalReductions = 1);
        var three = Model(o => o.PyramidalReductions = 3);
        var encoded = Features(frames: 16, width: 16);

        Assert.Equal(8, one.PyramidalReduce(encoded).Shape[0]);
        Assert.Equal(2, three.PyramidalReduce(encoded).Shape[0]);
        Assert.True(three.TimeReductionFactor > one.TimeReductionFactor);
    }

    [Fact]
    public void ReductionStopsRatherThanCollapsingToNothing()
    {
        // A short utterance must not be reduced away entirely.
        var model = Model(o => o.PyramidalReductions = 8);
        var reduced = model.PyramidalReduce(Features(frames: 2, width: 4));
        Assert.True(reduced.Shape[0] >= 1);
    }

    [Fact]
    public void NamedActivationsExposeTheThreeLasStages()
    {
        var model = Model();
        var activations = model.GetNamedLayerActivations(Features());

        Assert.Contains("Listener", activations.Keys);
        Assert.Contains("PyramidalSummary", activations.Keys);
        Assert.Contains("Speller", activations.Keys);

        // The summary must genuinely be shorter than what the listener produced.
        Assert.True(activations["PyramidalSummary"].Shape[0] < activations["Listener"].Shape[0]);
    }

    [Fact]
    public void BothArmsTrainAndProduceFiniteOutput()
    {
        foreach (var arm in Enum.GetValues<MedicalAsrDecoderType>())
        {
            var model = Model(o => o.DecoderType = arm);
            var input = Features();

            var output = model.Predict(input);
            for (int i = 0; i < output.Length; i++)
            {
                Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]),
                    $"{arm} produced a non-finite output.");
            }

            var before = model.GetParameters();
            var target = new Tensor<double>(output.Shape.ToArray());
            var rng = new Random(7);
            for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

            model.Train(input, target);
            var after = model.GetParameters();
            Assert.True(Enumerable.Range(0, Math.Min(before.Length, after.Length))
                    .Any(i => Math.Abs(before[i] - after[i]) > 1e-12),
                $"{arm} did not update any parameter.");
        }
    }

    [Fact]
    public void TranscriptionReportsFiniteConfidence()
    {
        var model = Model();
        var audio = new Tensor<double>([64]);
        var rng = new Random(11);
        for (int i = 0; i < audio.Length; i++) audio[i] = rng.NextDouble() - 0.5;

        var result = model.Transcribe(audio);
        Assert.NotNull(result);
        Assert.False(double.IsNaN(result.Confidence) || double.IsInfinity(result.Confidence));
    }
}
