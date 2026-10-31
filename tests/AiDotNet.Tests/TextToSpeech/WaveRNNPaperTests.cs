using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// WaveRNN follows its paper (Kalchbrenner et al. 2018): a single recurrent layer with a dual softmax over the coarse and
/// fine bytes of 16-bit samples, the current coarse byte reaching only the fine half of the state, teacher-forced
/// likelihood training, coarse-then-fine sampling and optional gradual magnitude pruning.
/// </summary>
/// <remarks>
/// Before this change WaveRNN ran a generic layer stack once over the mel spectrogram: no recurrence over samples, no
/// dual softmax and no sampling.
/// </remarks>
public class WaveRNNPaperTests
{
    private static WaveRNNOptions Options(double sparsity = 0) => new()
    {
        MelChannels = 8, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelMinFrequency = 0, MelMaxFrequency = 2000,
        UpsampleScales = [4, 4], RnnDim = 16, SequenceSamples = 64,
        SparsityTarget = sparsity, PruningStartStep = 0, PruningSteps = 4, PruningInterval = 1, PruneBlockRows = 4, PruneBlockColumns = 4,
    };

    private static WaveRNN<double> CreateModel(double sparsity = 0) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 16) { RandomSeed = 8 },
        Options(sparsity),
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio(int length = 512)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * 250 * i / 4000.0);
        return audio;
    }

    private static Tensor<double> Frames(Tensor<double> mel, int count)
    {
        var slice = new Tensor<double>(new[] { 1, mel.Shape[1], count });
        for (int c = 0; c < mel.Shape[1]; c++)
            for (int f = 0; f < count; f++) slice[0, c, f] = mel[0, c, f];
        return slice;
    }

    [Fact(Timeout = 60000)]
    public async Task SixteenBitSamples_SplitIntoCoarseAndFineBytes()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = new Tensor<double>(new[] { 3 });
        audio[0] = -1;
        audio[1] = 0;
        audio[2] = 1;
        var (coarse, fine) = model.Split(audio);
        Assert.Equal(new[] { 0, 128, 255 }, coarse);
        Assert.Equal(new[] { 0, 0, 255 }, fine);
        for (int s = 0; s < 65536; s += 997) Assert.Equal(s, WaveRNN<double>.ToSixteenBit(WaveRNN<double>.FromSixteenBit(s)));
    }

    [Fact(Timeout = 60000)]
    public async Task CurrentCoarseByte_ReachesOnlyTheFinePrediction()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = Frames(model.ComputeMel(Audio()), 4);
        var audio = Audio(64);
        var changed = new Tensor<double>(new[] { 64 });
        for (int i = 0; i < 64; i++) changed[i] = audio[i];
        // Change only the coarse byte of the last sample.
        changed[63] = WaveRNN<double>.FromSixteenBit(((WaveRNN<double>.ToSixteenBit(audio[63]) >> 8) ^ 0x40) << 8 | (WaveRNN<double>.ToSixteenBit(audio[63]) & 255));
        var (coarseA, fineA) = model.Logits(mel, audio);
        var (coarseB, fineB) = model.Logits(mel, changed);
        for (int c = 0; c < 256; c++) Assert.Equal(coarseA[0, c, 63], coarseB[0, c, 63], 12);
        Assert.Contains(Enumerable.Range(0, 256), c => Math.Abs(fineA[0, c, 63] - fineB[0, c, 63]) > 1e-9);
        // Earlier samples are untouched (causality).
        for (int c = 0; c < 256; c++) Assert.Equal(fineA[0, c, 62], fineB[0, c, 62], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheNegativeLogLikelihood()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = Audio(64);
        var mel = Frames(model.ComputeMel(Audio()), 4);
        double before = provider.EvaluateTrainingObjective(mel, audio);
        for (int i = 0; i < 15; i++) model.Train(mel, audio);
        double after = provider.EvaluateTrainingObjective(mel, audio);
        Assert.True(after < before, $"The negative log-likelihood did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 300000)]
    public async Task Pruning_ReachesTheTargetSparsityInBlocks()
    {
        await Task.Yield();
        var model = CreateModel(sparsity: 0.5);
        var audio = Audio(64);
        var mel = Frames(model.ComputeMel(Audio()), 4);
        Assert.Equal(0.0, model.RecurrentSparsity);
        for (int i = 0; i < 6; i++) model.Train(mel, audio);
        // Z = 0.5 of each gate's 4×4 blocks after S = 4 steps.
        Assert.Equal(0.5, model.RecurrentSparsity, 6);
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_DrawsSixteenBitSamples_Repeatably()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = Frames(model.ComputeMel(Audio()), 2);
        var wave = model.MelToWaveform(mel);
        Assert.Equal(32, wave.Length);
        for (int i = 0; i < wave.Length; i++)
        {
            int s = WaveRNN<double>.ToSixteenBit(wave[i]);
            Assert.Equal(WaveRNN<double>.FromSixteenBit(s), wave[i], 12);
        }
        Assert.Equal(wave.ToVector().ToArray(), model.MelToWaveform(mel).ToVector().ToArray());
    }
}
