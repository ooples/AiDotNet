using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Multilingual;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Google's Chirp recognizers. Google publishes no Chirp paper: Chirp is its deployment of USM (Zhang et al. 2023), and
/// Chirp 2 and 3 are presented as later USM-based generations with unpublished networks, so all three build USM-CTC.
/// These check that they are that network: the Conformer-2B defaults, the 40 ms CTC frame rate and BEST-RQ
/// pre-training (#2397).
/// </summary>
public class ChirpPaperTests
{
    private static Tensor<double> Voice(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * 180 * t) + 0.15 * Math.Sin(2 * Math.PI * 540 * t) * Math.Sin(2 * Math.PI * 4 * t);
        }
        return audio;
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 8);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreUsmConformer2B()
    {
        await Task.Yield();
        var c1 = new ChirpOptions();
        var c2 = new Chirp2Options();
        var c3 = new Chirp3Options();
        // USM Table 2: Conformer-2B is 32 layers, 1536 wide, 16 heads, kernel 5; section 2.7: 128 mel bins and 16 codebooks.
        Assert.Equal((32, 1536, 16, 5, 128, 16), (c1.NumEncoderLayers, c1.EncoderDim, c1.NumAttentionHeads, c1.ConvKernel, c1.NumMels, c1.NumCodebooks));
        Assert.Equal((32, 1536, 16, 5, 128, 16), (c2.NumEncoderLayers, c2.EncoderDim, c2.NumAttentionHeads, c2.ConvKernel, c2.NumMels, c2.NumCodebooks));
        Assert.Equal((32, 1536, 16, 5, 128, 16), (c3.NumEncoderLayers, c3.EncoderDim, c3.NumAttentionHeads, c3.ConvKernel, c3.NumMels, c3.NumCodebooks));
        // Section 2.4: 8-second chunks of 40 ms frames.
        Assert.Equal(200, c1.ToConfiguration().AttentionChunkSize);
    }

    [Fact(Timeout = 120000)]
    public async Task EveryGeneration_EmitsCtcFramesAtFortyMilliseconds()
    {
        await Task.Yield();
        var audio = Voice(8000);
        using var chirp = new Chirp<double>(Architecture(), new ChirpOptions { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, NumMels = 16, CodebookSize = 32, NumCodebooks = 2, VocabSize = 8, MaxAudioLengthSeconds = 1, Seed = 5 });
        using var chirp2 = new Chirp2<double>(Architecture(), new Chirp2Options { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, NumMels = 16, CodebookSize = 32, NumCodebooks = 2, VocabSize = 8, MaxAudioLengthSeconds = 1, Seed = 5 });
        using var chirp3 = new Chirp3<double>(Architecture(), new Chirp3Options { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, NumMels = 16, CodebookSize = 32, NumCodebooks = 2, VocabSize = 8, MaxAudioLengthSeconds = 1, Seed = 5 });
        // 0.5 s is 49 10 ms frames, two stride-2 convolutions reduce that to 12 frames of 40 ms.
        foreach (var logits in new[] { chirp.Predict(audio), chirp2.Predict(audio), chirp3.Predict(audio) })
            Assert.Equal(new[] { 12, 8 }, logits.Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task BestRqPretraining_LowersTheMultiSoftmaxLoss()
    {
        await Task.Yield();
        using var model = new Chirp<double>(Architecture(), new ChirpOptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardMultiplier = 2, NumMels = 16,
            CodebookSize = 32, CodeDim = 8, NumCodebooks = 3, MaskProbability = 0.2, MaskLength = 4, VocabSize = 8,
            MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var audio = Voice(8000);
        double Loss() => model.PretrainingLoss(audio, Tensors.Helpers.RandomHelper.CreateSeededRandom(3));
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Pretrain(audio);
        double after = Loss();
        Assert.True(after < 0.8 * before, $"multi-softmax loss {before} -> {after}");
    }
}
