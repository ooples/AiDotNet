using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.Multilingual;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Google USM (Zhang et al. 2023): the Conformer-2B of Table 2 over 128-bin log-mel features, BEST-RQ pre-training with
/// the 16-codebook multi-softmax loss (sections 2.2.1 and 2.7) and 8-second chunk-wise attention (section 2.4). Google
/// released no code, so these check the paper directly (#2321).
/// </summary>
public class USMPaperTests
{
    private static USMOptions Tiny() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardMultiplier = 2, NumMels = 16,
        CodebookSize = 32, CodeDim = 8, NumCodebooks = 3, MaskProbability = 0.2, MaskLength = 4, VocabSize = 8,
        MaxAudioLengthSeconds = 1, Seed = 5,
    };

    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * pitch * t) + 0.15 * Math.Sin(2 * Math.PI * 3 * pitch * t) * Math.Sin(2 * Math.PI * 4 * t);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreConformer2B_WithSixteenCodebooks_AndEightSecondChunks()
    {
        await Task.Yield();
        var o = new USMOptions();
        Assert.Equal((32, 1536, 16, 5), (o.NumEncoderLayers, o.EncoderDim, o.NumAttentionHeads, o.ConvKernel));
        Assert.Equal((128, 16, 8192), (o.NumMels, o.NumCodebooks, o.CodebookSize));
        // 8 s of 40 ms encoder frames (10 ms frames reduced 4x).
        Assert.Equal(200, o.ToConfiguration().AttentionChunkSize);
    }

    [Fact(Timeout = 60000)]
    public async Task ChunkWiseAttention_KeepsAttentionInsideItsChunk()
    {
        await Task.Yield();
        // One block, 24 frames, chunks of 8. Changing frame 20 reaches frames 16-23 through attention and two frames either
        // side through the kernel-5 convolution, which can carry it into chunk 1 and from there two frames into chunk 0:
        // frames 0-5 cannot change. Without chunks attention carries it to every frame.
        var blocks = new System.Collections.Generic.Dictionary<int, ConformerBlockLayer<double>>();
        var initial = new ConformerBlockLayer<double>(16, 2, 2, 5, ropeTheta: 10000.0).GetParameters();
        Tensor<double> Run(int chunk, double bump)
        {
            if (!blocks.TryGetValue(chunk, out var block))
            {
                // Every chunk setting gets the same weights.
                block = new ConformerBlockLayer<double>(16, 2, 2, 5, ropeTheta: 10000.0, attentionChunkSize: chunk);
                block.SetParameters(initial);
                block.SetTrainingMode(false);
                blocks[chunk] = block;
            }
            var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(9);
            var x = new Tensor<double>(new[] { 1, 24, 16 });
            for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble() - 0.5;
            for (int d = 0; d < 16; d++) x[0, 20, d] += bump * Math.Sin(d + 1);   // not a constant shift, which LayerNorm removes
            return block.Forward(x);
        }
        var chunked = Run(8, 0.0);
        var chunkedBumped = Run(8, 1.0);
        for (int t = 0; t < 6; t++)
            for (int d = 0; d < 16; d++)
                Assert.Equal(chunked[0, t, d], chunkedBumped[0, t, d], 12);
        var whole = Run(0, 0.0);
        var wholeBumped = Run(0, 1.0);
        double moved = Enumerable.Range(0, 16).Sum(d => Math.Abs(whole[0, 0, d] - wholeBumped[0, 0, d]));
        Assert.True(moved > 1e-9, "global attention carries the change to frame 0");
        // A chunk covering the whole sequence is global attention.
        var oneChunk = Run(24, 1.0);
        for (int i = 0; i < oneChunk.Length; i++) Assert.Equal(wholeBumped[i], oneChunk[i], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task MultiSoftmaxPretraining_UsesIndependentCodebooks_AndLowersTheLoss()
    {
        await Task.Yield();
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 3e-3 });
        using var model = new USM<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.SequenceToSequence, inputSize: 4000, outputSize: 8), Tiny(), optimizer);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(3, network.PretrainingHeads.Count);
        var audio = Voice(4000, 210);
        var features = network.Features(Enumerable.Range(0, audio.Length).Select(i => audio[i]).ToArray());
        var labels = Enumerable.Range(0, 3).Select(n => network.Quantize(features, n)).ToArray();
        Assert.False(labels[0].SequenceEqual(labels[1]) && labels[1].SequenceEqual(labels[2]), "the quantizers are independent draws");
        double Loss() => model.PretrainingLoss(audio, Tensors.Helpers.RandomHelper.CreateSeededRandom(11));
        double before = Loss();
        for (int step = 0; step < 40; step++) model.Pretrain(audio);
        double after = Loss();
        Assert.True(after < 0.7 * before, $"multi-softmax loss {before} -> {after}");
        Assert.Equal(new[] { network.EncodedLength(features.GetLength(0)), 8 }, model.Predict(audio).Shape.ToArray());
    }
}
