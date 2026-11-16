using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Audio.StableAudio;
using AiDotNet.Audio.Whisper;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class AudioUnresolvedReviewTests
{
    [Fact]
    public void Normalization_HandlesMoreFramesThanHalfCanRepresent()
    {
        var data = Enumerable.Range(0, 70000).Select(i => (Half)(i % 2 == 0 ? -2 : 2)).ToArray();
        var result = ConformerAsrNetwork<Half>.NormalizePerBin(AiDotNetEngine.Current,
            new Tensor<Half>(data, new[] { data.Length, 1 }));
        for (int i = 0; i < data.Length; i++) Assert.Equal(i % 2 == 0 ? -1.0 : 1.0, (double)result[i, 0]);
    }

    [Fact]
    public void Normalization_ScalesBeforeDoubleReductions()
    {
        var varying = ConformerAsrNetwork<double>.NormalizePerBin(AiDotNetEngine.Current,
            new Tensor<double>(new[] { -1e154, 1e154 }, new[] { 2, 1 }));
        Assert.Equal(-1, varying[0, 0], 12);
        Assert.Equal(1, varying[1, 0], 12);
        var constant = ConformerAsrNetwork<double>.NormalizePerBin(AiDotNetEngine.Current,
            new Tensor<double>(new[] { 1e308, 1e308 }, new[] { 2, 1 }));
        Assert.Equal(new[] { 0.0, 0.0 }, constant.ToArray());
    }

    [Fact]
    public void StereoLatentMask_CombinesMatchingTimesAcrossChannels()
    {
        var mask = new Tensor<double>(new[] { 1.0, 0, 0, 0, 0, 0, 0, 1 }, new[] { 2, 4 });
        var result = StableAudioModel<double>.LatentMask(AiDotNetEngine.Current, mask, new[] { 3, 4 });
        Assert.Equal(new[] { 1.0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 }, result.ToArray());
    }

    [Theory]
    [InlineData(double.NaN, 0.0, 1.0, 2, 0.7310585786300049)]
    [InlineData(0.0, double.PositiveInfinity, 1.0, 1, 1.0)]
    [InlineData(double.NaN, double.NegativeInfinity, double.NaN, 0, 0.3333333333333333)]
    public void Whisper_ConfidenceUsesTheSameFiniteLogitsAsSelection(double a, double b, double c, int token, double probability)
    {
        var result = WhisperModel<double>.GetNextTokenWithProbability(new Tensor<double>(new[] { a, b, c }, new[] { 1, 1, 3 }), 0);
        Assert.Equal(token, result.token);
        Assert.Equal(probability, result.probability, 12);
    }

    private static RNNTransducer<double> SmallTransducer() => new(
        new NeuralNetworkArchitecture<double>(inputFeatures: 16, outputSize: 3),
        new RNNTransducerOptions { NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumEncoderHeads = 2,
            PredictionDim = 8, NumPredictionLayers = 1, EmbeddingDim = 8, JointDim = 8, VocabSize = 3, DropoutRate = 0 });

    [Fact]
    public void Rnnt_TargetRows_DistinguishBatchOneFromAnUnbatchedDistribution()
    {
        using var model = SmallTransducer();
        var method = typeof(RNNTransducer<double>).GetMethod("RowLabels", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var hard = new Tensor<double>(new[] { 1.0, 2, 1 }, new[] { 1, 3 });
        var labels = (int[][])method.Invoke(model, new object[] { hard, 1, true })!;
        Assert.Equal(new[] { 1, 2, 1 }, labels[0]);
        var soft = new Tensor<double>(new[] { 0.0, 1, 0, 0, 0, 1 }, new[] { 2, 3 });
        labels = (int[][])method.Invoke(model, new object[] { soft, 1, false })!;
        Assert.Equal(new[] { 1, 2 }, labels[0]);
        labels = (int[][])method.Invoke(model, new object[] { soft.Reshape(new[] { 1, 2, 3 }), 1, true })!;
        Assert.Equal(new[] { 1, 2 }, labels[0]);
    }

    [Fact]
    public void Rnnt_WithoutPredictionNetwork_RejectsBeamSearch()
    {
        using var model = SmallTransducer();
        typeof(RNNTransducer<double>).GetField("_network", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, null);
        Assert.Throws<NotSupportedException>(() => model.TranscribeWithBeamSearch(new Tensor<double>(new[] { 3200 }), 2));
    }

    [Fact]
    public void Ctc_LongPrefixesKeepTheirStructuralIdentity()
    {
        const int frames = 80, vocabulary = 5;
        var data = new double[frames * vocabulary];
        var expected = new List<int>();
        for (int t = 0; t < frames; t++)
        {
            int winner = t % 2 == 0 ? 1 + t / 2 % 4 : 0;
            if (winner != 0) expected.Add(winner);
            for (int k = 0; k < vocabulary; k++) data[t * vocabulary + k] = k == winner ? 0 : -100;
        }
        Assert.Equal(expected, SpeechRecognitionEngineOps.CtcPrefixBeamSearch(new Tensor<double>(data, new[] { frames, vocabulary }), 4, 0));
    }
}
