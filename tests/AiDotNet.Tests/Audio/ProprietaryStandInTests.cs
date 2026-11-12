using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SpeechRecognition.ProprietaryAPI;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The hosted recognizers whose vendors publish no architecture, each built as the nearest disclosed one (#2397): Azure
/// as Microsoft's streaming Conformer transducer (Chen et al. 2021), Amazon Transcribe as a streaming Conformer
/// transducer, Deepgram Nova-2 as a Transformer-CTC model, Speechmatics as a Conformer with a phoneme CTC layer (Ursa's
/// disclosed acoustic model), and Sarvam as a Conformer-CTC model over Indic-script code points.
/// </summary>
public class ProprietaryStandInTests
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

    private static Tensor<double> Labels(params int[] labels) => new(new[] { labels.Length }, new Vector<double>(labels.Select(l => (double)l).ToArray()));

    [Fact(Timeout = 60000)]
    public async Task Azure_IsMicrosoftsConformerTransducer()
    {
        await Task.Yield();
        var o = new AzureSpeechSTTOptions();
        // Section 4.1: width 640, kernel 3, a two-layer 720-unit LSTM predictor, 4,000 sentence pieces; 1,024 feed-forward
        // units and 18 layers from the size-matched Transformer transducer.
        Assert.Equal((640, 18, 1024, 3), (o.EncoderDim, o.NumEncoderLayers, o.FeedForwardUnits, o.ConvKernel));
        Assert.Equal((720, 2, 4001), (o.PredictionDim, o.PredictionLayers, o.VocabSize));
        // 720 ms chunks (24 x 30 ms in the paper, 18 x 40 ms here) with at least 1.8 s of history.
        Assert.Equal(720, o.AttentionChunkSize * 40);
        Assert.True(o.AttentionChunkSize * o.AttentionLeftChunks * 40 >= 1800);
        var c = o.ToConfiguration();
        Assert.Equal((ConformerAsrDecoder.Transducer, 1024, 18, 3), (c.Decoder, c.FeedForwardUnits, c.AttentionChunkSize, c.AttentionLeftChunks));
    }

    [Fact(Timeout = 60000)]
    public async Task FeedForwardUnits_SetTheBlocksHiddenWidth()
    {
        await Task.Yield();
        // Two macaron modules of 16 -> 24 -> 16 instead of 16 -> 64 -> 16.
        var wide = new ConformerBlockLayer<double>(16, 2, 4, 3);
        var narrow = new ConformerBlockLayer<double>(16, 2, 4, 3, feedForwardUnits: 24);
        int perModule(int hidden) => 16 * hidden + hidden + hidden * 16 + 16;
        Assert.Equal(2 * (perModule(64) - perModule(24)), wide.ParameterCount - narrow.ParameterCount);
        Assert.Equal("24", narrow.GetMetadata()["FeedForwardUnits"]);
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheStandInsSizes()
    {
        await Task.Yield();
        var aws = new AWSTranscribeOptions();
        Assert.Equal((17, 512, 8, 32, 640, 1), (aws.NumEncoderLayers, aws.EncoderDim, aws.NumAttentionHeads, aws.ConvKernel, aws.PredictionDim, aws.PredictionLayers));
        var deepgram = new DeepgramNova2Options();
        Assert.Equal((12, 512, 8, 4), (deepgram.NumEncoderLayers, deepgram.EncoderDim, deepgram.NumAttentionHeads, deepgram.FeedForwardMultiplier));
        Assert.Equal(ConformerAsrBlock.Transformer, deepgram.ToConfiguration().Block);
        var speechmatics = new SpeechmaticsASROptions();
        Assert.Equal(41, speechmatics.VocabSize);
        Assert.Equal(speechmatics.VocabSize, speechmatics.Vocabulary.Length);
        var sarvam = new SarvamASROptions();
        Assert.Equal((0x0D80, "hi"), (sarvam.VocabSize, sarvam.Language));
    }

    [Fact(Timeout = 300000)]
    public async Task Azure_TrainsTheTransducer()
    {
        await Task.Yield();
        using var model = new AzureSpeechSTT<double>(Architecture(), new AzureSpeechSTTOptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardUnits = 24, ConvKernel = 3, NumMels = 16,
            AttentionChunkSize = 2, AttentionLeftChunks = 1, PredictionDim = 16, JointDim = 16, VocabSize = 8,
            FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(2, network.PredictionLstms.Count);
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var features = network.Features(audio.ToArray());
        double Loss() => network.TransducerLoss(network.Encode(features, training: false), labels)[0];
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, Labels(labels));
        Assert.True(Loss() < 0.5 * before, $"transducer loss {before} -> {Loss()}");
    }

    [Fact(Timeout = 300000)]
    public async Task Deepgram_TrainsTheTransformerCtc()
    {
        await Task.Yield();
        using var model = new DeepgramNova2<double>(Architecture(), new DeepgramNova2Options
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, NumMels = 16, VocabSize = 8,
            FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Single(network.TransformerBlocks);
        Assert.Empty(network.Blocks);
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var features = network.Features(audio.ToArray());
        double Loss() => network.CtcTrainingLoss(features, labels, training: false)[0];
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, Labels(labels));
        Assert.True(Loss() < 0.5 * before, $"CTC loss {before} -> {Loss()}");
    }

    [Fact(Timeout = 300000)]
    public async Task Speechmatics_TranscribesPhonemes()
    {
        await Task.Yield();
        using var model = new SpeechmaticsASR<double>(Architecture(), new SpeechmaticsASROptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16,
            FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var o = new SpeechmaticsASROptions();
        // HH AH L OW: tokens 2 + the phoneme's index.
        var labels = new[] { "HH ", "AH ", "L ", "OW " }.Select(p => Array.IndexOf(o.Vocabulary, p)).ToArray();
        Assert.DoesNotContain(-1, labels);
        var audio = Voice(8000, 180);
        for (int step = 0; step < 200; step++) model.Train(audio, Labels(labels));
        string text = model.Transcribe(audio).Text;
        Assert.True(text == "HH AH L OW", $"transcript '{text}'");
    }

    [Fact(Timeout = 300000)]
    public async Task Sarvam_TrainsOnCodePointLabels()
    {
        await Task.Yield();
        using var model = new SarvamASR<double>(Architecture(), new SarvamASROptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16,
            FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        // "नमस्ते" (namaste) in Devanagari: each label is the character's code point.
        var labels = "नमस्ते".Select(ch => (int)ch).ToArray();
        var audio = Voice(8000, 180);
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, Labels(labels));
        for (int step = 0; step < 100; step++) model.Train(audio, Labels(labels));
        double after = objective.EvaluateTrainingObjective(audio, Labels(labels));
        // The 3,456-way head starts near ln(3456) per frame and leaves the all-blank phase only after a few hundred
        // steps, so the check is the loss, which falls steadily.
        Assert.True(after < 0.5 * before, $"CTC loss {before} -> {after}");
        // A code point outside the vocabulary is rejected rather than wrapped.
        Assert.Throws<ArgumentException>(() => model.Train(audio, Labels(0x0D80)));
    }
}
