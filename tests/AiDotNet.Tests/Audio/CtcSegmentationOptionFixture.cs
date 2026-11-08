using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.CTCVariants;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

internal static class CtcSegmentationOptionFixture
{
    internal static CTCSegmentation<T> Create<T>() => new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
        NeuralNetworkTaskType.SequenceToSequence, inputSize: 6400, outputSize: 8) { RandomSeed = 7 },
        new CTCSegmentationOptions { Seed = 7, NumMels = 8, EncoderDim = 16, NumEncoderLayers = 1,
            NumAttentionHeads = 2, FeedForwardMultiplier = 2, AttentionDecoderLayers = 1,
            AttentionDecoderFeedForward = 32, VocabSize = 8, ConfidenceWindow = 2 });

    internal static Tensor<T> Waveform<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var waveform = new Tensor<T>(new[] { 6400 });
        for (int i = 0; i < waveform.Length; i++) waveform[i] = ops.FromDouble(.1 * Math.Sin(.03 * i) + .05 * Math.Cos(.11 * i));
        return waveform;
    }

    internal static long Observe<T>(CTCSegmentation<T> model)
    {
        var result = model.Segment(Waveform<T>(), new[] { 1, 2, 1 });
        return TtsPaperNumerics.Fingerprint(result.LabelFrames.Select(frame => (double)frame).Concat(new[] { result.Confidence }));
    }
}

public class CtcSegmentationOptionTests
{
    [Fact(Timeout = 120000)]
    public void ConfidenceWindowChangesConfidenceWithoutChangingAlignment()
    {
        using var model = CtcSegmentationOptionFixture.Create<double>();
        var waveform = CtcSegmentationOptionFixture.Waveform<double>();
        var first = model.Segment(waveform, new[] { 1, 2, 1 });
        ((CTCSegmentationOptions)model.GetOptions()).ConfidenceWindow = 3;
        var second = model.Segment(waveform, new[] { 1, 2, 1 });
        Assert.Equal(first.LabelFrames, second.LabelFrames);
        Assert.NotEqual(first.Confidence, second.Confidence);
    }
}
