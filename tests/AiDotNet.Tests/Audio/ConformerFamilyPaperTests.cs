using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.SpeechRecognition.CTCVariants;
using AiDotNet.SpeechRecognition.NeMo;
using AiDotNet.SpeechRecognition.Streaming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Gulati et al. 2020's Conformer recognizers: Table 1's sizes, the 4x convolution subsampling, and training with the CTC
/// and RNN-T objectives (#2397).
/// </summary>
public class ConformerFamilyPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 8);

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
    public async Task Defaults_AreTableOnesSizes()
    {
        await Task.Yield();
        // Table 1: Conformer (M) is 16 blocks, width 256, 4 heads, kernel 32, with a 640-unit LSTM; (L) is 17, 512, 8.
        var m = new ConformerOptions();
        Assert.Equal((16, 256, 4, 32, 640, 1), (m.NumEncoderLayers, m.EncoderDim, m.NumAttentionHeads, m.ConvKernel, m.PredictionDim, m.PredictionLayers));
        var l = new ConformerTransducerOptions();
        Assert.Equal((17, 512, 8, 32), (l.NumEncoderLayers, l.EncoderDim, l.NumAttentionHeads, l.ConvKernel));
        // Section 3.1: SpecAugment with 2 frequency masks (F = 27) and 10 time masks (pS = 0.05).
        Assert.Equal((2, 27, 10, 0.05), (m.FrequencyMasks, m.FrequencyMaskWidth, m.TimeMasks, m.TimeMaskRatio));
        Assert.Equal(80, m.NumMels);
    }

    private static ConformerCTCOptions TinyCtc() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, VocabSize = 8,
        FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
    };

    [Fact(Timeout = 120000)]
    public async Task Subsampling_ReducesTheFrameRateFourTimes()
    {
        await Task.Yield();
        using var model = new ConformerCTC<double>(Architecture(), TinyCtc());
        // 0.5 s is 51 10 ms frames; two stride-2 3x3 convolutions give 12 frames of 40 ms.
        Assert.Equal(new[] { 12, 8 }, model.Predict(Voice(8000, 180)).Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task CtcTraining_LowersTheCtcLoss()
    {
        await Task.Yield();
        using var model = new ConformerCTC<double>(Architecture(), TinyCtc());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        double Loss()
        {
            var logits = network.CtcLogits(network.Encode(network.Features(audio.ToArray()), training: false));
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            return engine.TensorCTCLoss(engine.Reshape(engine.TensorLogSoftmax(logits, axis: -1), new[] { logits.Shape[0], 1, 8 }),
                new Tensor<int>(labels, new[] { 3 }), new[] { logits.Shape[0] }, new[] { 3 }, 0)[0];
        }
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = Loss();
        Assert.True(after < 0.5 * before, $"CTC loss {before} -> {after}");
    }

    [Fact(Timeout = 300000)]
    public async Task TransducerTraining_LowersTheTransducerLoss_AndGreedyDecodingRuns()
    {
        await Task.Yield();
        using var model = new ConformerTransducer<double>(Architecture(), new ConformerTransducerOptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, PredictionDim = 16,
            JointDim = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        double Loss() => network.TransducerLoss(network.Encode(network.Features(audio.ToArray()), training: false), labels)[0];
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = Loss();
        Assert.True(after < 0.5 * before, $"transducer loss {before} -> {after}");
        _ = model.Transcribe(audio);
    }

    [Fact(Timeout = 120000)]
    public async Task StreamingConformer_AttendsWithinChunks()
    {
        await Task.Yield();
        Assert.Equal(10, new StreamingConformerOptions().AttentionChunkSize);
        Assert.Equal(10, new StreamingConformerOptions().ToConfiguration().AttentionChunkSize);
    }

    [Fact(Timeout = 60000)]
    public async Task BranchformerDefaults_AreTheLibriSpeechModels()
    {
        await Task.Yield();
        var b = new BranchformerOptions();
        Assert.Equal((18, 512, 8, 3072, 31), (b.NumEncoderLayers, b.EncoderDim, b.NumAttentionHeads, b.CgmlpDim, b.ConvKernel));
        // E-Branchformer-L (Kim et al. 2022, Table 1): 17 blocks, width 512, 8 heads, 3072 cgMLP units, 2048 FFN units.
        var e = new EBranchformerOptions();
        Assert.Equal((17, 512, 8, 3072, 2048, 31), (e.NumEncoderLayers, e.EncoderDim, e.NumAttentionHeads, e.CgmlpDim, e.EncoderDim * e.FeedForwardMultiplier, e.MergeKernel));
    }

    [Theory(Timeout = 300000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BranchformerTraining_LowersTheCtcLoss(bool enhanced)
    {
        await Task.Yield();
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        AiDotNet.NeuralNetworks.NeuralNetworkBase<double> model = enhanced
            ? new EBranchformer<double>(Architecture(), new EBranchformerOptions { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, MergeKernel = 3, CgmlpDim = 32, NumMels = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5 })
            : new Branchformer<double>(Architecture(), new BranchformerOptions { EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, CgmlpDim = 32, NumMels = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5 });
        using (model)
        {
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            double Loss()
            {
                var logits = model.Predict(audio);
                return engine.TensorCTCLoss(engine.Reshape(engine.TensorLogSoftmax(logits, axis: -1), new[] { logits.Shape[0], 1, 8 }),
                    new Tensor<int>(labels, new[] { 3 }), new[] { logits.Shape[0] }, new[] { 3 }, 0)[0];
            }
            double before = Loss();
            for (int step = 0; step < 30; step++) model.Train(audio, target);
            double after = Loss();
            Assert.True(after < 0.5 * before, $"CTC loss {before} -> {after}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FastConformer_SubsamplesEightTimes()
    {
        await Task.Yield();
        var o = new ParakeetCTCOptions();
        Assert.Equal((24, 1024, 8, 9, 256, 1024), (o.NumEncoderLayers, o.EncoderDim, o.NumAttentionHeads, o.ConvKernel, o.SubsamplingChannels, o.VocabSize));
        using var model = new ParakeetCTC<double>(Architecture(), new ParakeetCTCOptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, SubsamplingChannels = 4,
            VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        // 51 10 ms frames through three stride-2 convolutions with padding 1: 26, 13, 7 frames of 80 ms.
        Assert.Equal(new[] { 7, 8 }, model.Predict(Voice(8000, 180)).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task FastEmit_IsCausal_SoLaterAudioLeavesEarlierFramesUnchanged()
    {
        await Task.Yield();
        Assert.Equal((0.004, 16, 256), (new FastEmitOptions().FastEmitLambda, new FastEmitOptions().NumEncoderLayers, new FastEmitOptions().EncoderDim));
        using var model = new FastEmit<double>(Architecture(), new FastEmitOptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, PredictionDim = 8, JointDim = 8,
            VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var features = network.Features(Voice(8000, 180).ToArray());
        var changed = (double[,])features.Clone();
        for (int t = 40; t < changed.GetLength(0); t++)
            for (int b = 0; b < changed.GetLength(1); b++) changed[t, b] += 3.0;
        var a = network.Encode(features, training: false);
        var c = network.Encode(changed, training: false);
        // Feature frames 40 on reach encoder frames from (40 - 3) / 4 = 9 on through the subsampling's receptive field; the
        // causal blocks carry nothing back to frames 0 to 7.
        for (int t = 0; t < 8; t++)
            for (int k = 0; k < 16; k++) Assert.Equal(a[t, k], c[t, k], 10);
    }

    [Theory(Timeout = 300000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IntermediateCtc_TrainsAndMixesTheLosses(bool selfConditioned)
    {
        await Task.Yield();
        var interDefaults = new InterCTCOptions();
        Assert.Equal((12, 0.3), (interDefaults.NumEncoderLayers, interDefaults.IntermediateCtcWeight));
        Assert.Equal(new[] { 6 }, interDefaults.IntermediateCtcLayers);
        var selfDefaults = new SelfConditionedCTCOptions();
        Assert.Equal((18, 0.5), (selfDefaults.NumEncoderLayers, selfDefaults.IntermediateCtcWeight));
        Assert.Equal(new[] { 3, 6, 9, 12, 15 }, selfDefaults.IntermediateCtcLayers);
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        AiDotNet.NeuralNetworks.NeuralNetworkBase<double> model = selfConditioned
            ? new SelfConditionedCTC<double>(Architecture(), new SelfConditionedCTCOptions { EncoderDim = 16, NumEncoderLayers = 2, IntermediateCtcLayers = new[] { 1 }, NumAttentionHeads = 2, NumMels = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5 })
            : new InterCTC<double>(Architecture(), new InterCTCOptions { EncoderDim = 16, NumEncoderLayers = 2, IntermediateCtcLayers = new[] { 1 }, NumAttentionHeads = 2, NumMels = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5 });
        using (model)
        {
            var network = selfConditioned ? ((SelfConditionedCTC<double>)model).Network : ((InterCTC<double>)model).Network;
            if (network is null) throw new InvalidOperationException("native model");
            var features = network.Features(audio.ToArray());
            double Loss() => network.CtcTrainingLoss(features, labels, training: false)[0];
            var (output, intermediates) = network.EncodeWithIntermediates(features, training: false);
            double mixed = Loss(), final = network.CtcLoss(network.CtcLogits(output), labels)[0], inter = network.CtcLoss(intermediates[0], labels)[0];
            double w = selfConditioned ? 0.5 : 0.3;
            Assert.Equal((1 - w) * final + w * inter, mixed, 8);
            for (int step = 0; step < 30; step++) model.Train(audio, target);
            Assert.True(Loss() < 0.5 * mixed, $"mixed CTC loss {mixed} -> {Loss()}");
        }
    }
}
