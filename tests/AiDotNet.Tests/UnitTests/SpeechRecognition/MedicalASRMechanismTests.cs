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
/// tests pin both arms and the input both read: 80 filterbanks stacked three frames at a time with a
/// stride of three (section 4). The earlier revision's "pyramidal" averaging of encoder outputs is not
/// in the paper, whose only time reduction is that stacking.
/// </remarks>
public class MedicalASRMechanismTests
{
    private static NeuralNetworkArchitecture<double> Arch() =>
        new(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputSize: 3200, outputSize: 24);

    private static MedicalASROptions Small(Action<MedicalASROptions>? tweak = null)
    {
        var o = new MedicalASROptions
        {
            EncoderDim = 8,
            NumEncoderLayers = 1,
            NumAttentionHeads = 2,
            NumMels = 8,
            VocabSize = 24,
            DecoderDim = 8,
            NumDecoderLayers = 1,
            MaxTextLength = 6,
            Seed = 3,
        };
        tweak?.Invoke(o);
        return o;
    }

    private static MedicalASR<double> Model(Action<MedicalASROptions>? tweak = null)
        => new(Arch(), Small(tweak));

    private static Tensor<double> Audio(int samples = 3200, int seed = 4)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>([samples]);
        for (int i = 0; i < t.Length; i++) t[i] = 0.3 * Math.Sin(2 * Math.PI * 220 * i / 16000.0) + 0.05 * (rng.NextDouble() - 0.5);
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
        Assert.Equal(2, Enum.GetValues(typeof(MedicalAsrDecoderType)).Length);
    }

    [Fact]
    public void InputStacksThreeFramesWithAStrideOfThree()
    {
        // "As input to the model we stack 3 such frames and take a stride of 3, i.e., input is 240
        // filterbanks every 30 ms."
        var model = Model();
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(3, model.TimeReductionFactor);
        var features = network.Features(Audio());
        // 0.2 s is 21 frames of 10 ms: ceil(21 / 3) = 7 steps of 3 x 8 values.
        Assert.Equal(new[] { 7, 24 }, features.Shape.ToArray());

        var unstacked = new MedicalASR<double>(Arch(), Small(o => o.StackedFrames = 1)).Network
            ?? throw new InvalidOperationException("native model");
        var frames = unstacked.Features(Audio());
        Assert.Equal(new[] { 21, 8 }, frames.Shape.ToArray());
        // Step 2 holds frames 6, 7 and 8 side by side.
        for (int k = 0; k < 3; k++)
            for (int b = 0; b < 8; b++)
                Assert.Equal(frames[6 + k, b], features[2, k * 8 + b], 9);
    }

    [Fact]
    public void TheEncoderIsBidirectional()
    {
        var model = Model();
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var features = network.Features(Audio());
        var encoded = network.Encode(features);
        Assert.Equal(new[] { 7, 16 }, encoded.Shape.ToArray());
        // The first step's encoding depends on the last step's input.
        var changed = features.Clone();
        for (int b = 0; b < features.Shape[1]; b++) changed[6, b] += 2.0;
        var other = network.Encode(changed);
        Assert.Contains(Enumerable.Range(0, 16), k => Math.Abs(other[0, k] - encoded[0, k]) > 1e-9);
    }

    [Fact]
    public void TheTwoArmsAreStructurallyDifferentModels()
    {
        // If both arms produced the same parameterization, only one of them would really exist.
        var las = Model();
        var ctc = Model(o => o.DecoderType = MedicalAsrDecoderType.Ctc);
        var lasNetwork = las.Network ?? throw new InvalidOperationException("native model");
        var ctcNetwork = ctc.Network ?? throw new InvalidOperationException("native model");

        Assert.Null(lasNetwork.CtcHead);
        Assert.NotNull(lasNetwork.Embedding);
        Assert.NotNull(ctcNetwork.CtcHead);
        Assert.Null(ctcNetwork.Embedding);
        Assert.NotEqual(las.GetParameters().Length, ctc.GetParameters().Length);
        // CTC is frame-synchronous: one distribution per input step. LAS emits one per grapheme.
        Assert.Equal(new[] { 7, 24 }, ctc.Predict(Audio()).Shape.ToArray());
        Assert.Equal(new[] { 6, 24 }, las.Predict(Audio()).Shape.ToArray());
    }

    [Fact]
    public void BothArmsTrainAndProduceFiniteOutput()
    {
        foreach (var arm in (MedicalAsrDecoderType[])Enum.GetValues(typeof(MedicalAsrDecoderType)))
        {
            var model = Model(o => o.DecoderType = arm);
            var input = Audio();

            var output = model.Predict(input);
            for (int i = 0; i < output.Length; i++)
            {
                Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]),
                    $"{arm} produced a non-finite output.");
            }

            var before = model.GetParameters();
            var target = new Tensor<double>([3], new Vector<double>(new[] { 5.0, 9.0, 2.0 }));

            model.Train(input, target);
            var after = model.GetParameters();
            Assert.Equal(before.Length, after.Length);
            Assert.True(Enumerable.Range(0, before.Length).Any(i => Math.Abs(before[i] - after[i]) > 1e-12),
                $"{arm} did not update any parameter.");
        }
    }

    [Fact]
    public void TranscriptionReportsFiniteConfidence()
    {
        var model = Model();
        var result = model.Transcribe(Audio(64));
        Assert.NotNull(result);
        Assert.False(double.IsNaN(result.Confidence) || double.IsInfinity(result.Confidence));
    }
}
