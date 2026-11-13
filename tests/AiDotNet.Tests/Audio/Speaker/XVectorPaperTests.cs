using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Speaker;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio.Speaker;

/// <summary>
/// The x-vector extractor (Snyder et al. 2018): Table 1's TDNN with statistics pooling, and training through the softmax
/// over speakers (#2397).
/// </summary>
public class XVectorPaperTests
{
    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * pitch * t) + 0.2 * Math.Sin(2 * Math.PI * 2 * pitch * t) + 0.1 * Math.Sin(2 * Math.PI * 3.3 * pitch * t);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTableOne()
    {
        await Task.Yield();
        var o = new SpeakerEmbeddingOptions();
        // 512-unit frame layers, 1500-unit pooled layer, 512-dimensional x-vectors, 24 filterbanks, 3-second mean normalization.
        Assert.Equal((512, 1500, 512, 24, 300), (o.HiddenDim, o.PoolingDim, o.EmbeddingDimension, o.NumMfcc, o.NormalizationWindowFrames));
    }

    [Fact(Timeout = 300000)]
    public async Task SpeakerSoftmaxTraining_SeparatesTwoSpeakers()
    {
        await Task.Yield();
        var options = new SpeakerEmbeddingOptions { HiddenDim = 16, PoolingDim = 24, EmbeddingDimension = 8, NumTrainingSpeakers = 2, Seed = 3 };
        var model = new SpeakerEmbeddingExtractor<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 8, outputSize: 8), options: options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var low = Voice(8000, 120);
        var high = Voice(8000, 230);
        var lowFeatures = network.Features(low);
        // Table 1's TDNN consumes 14 frames of context; the pooled statistics are twice the 24-unit layer.
        Assert.Equal(48, network.Segment6.GetInputShape()[^1]);
        var batch = new Tensor<double>(new[] { 2, lowFeatures.Shape[0], lowFeatures.Shape[1] });
        var highFeatures = network.Features(high);
        for (int t = 0; t < lowFeatures.Shape[0]; t++)
            for (int b = 0; b < lowFeatures.Shape[1]; b++) { batch[0, t, b] = lowFeatures[t, b]; batch[1, t, b] = highFeatures[t, b]; }
        double Loss()
        {
            var logits = network.SpeakerLogits(network.Embed(batch, training: false));
            double total = 0;
            for (int r = 0; r < 2; r++)
            {
                double max = Math.Max(logits[r, 0], logits[r, 1]), sum = Math.Exp(logits[r, 0] - max) + Math.Exp(logits[r, 1] - max);
                total -= logits[r, r] - max - Math.Log(sum);
            }
            return total / 2;
        }
        double before = Loss();
        // A minibatch of both speakers, as the paper trains: the segment-level batch normalization needs more than one
        // utterance per step.
        var targets = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0 }));
        for (int step = 0; step < 30; step++) model.Train(batch, targets);
        Assert.True(Loss() < 0.5 * before, $"speaker cross-entropy {before} -> {Loss()}");
        var a = model.Extract(low).Vector;
        var c = model.Extract(high).Vector;
        Assert.Equal(8, a.Length);
        Assert.True(a.Zip(c, (x, y) => Math.Abs(x - y)).Sum() > 1e-3);
    }
}
