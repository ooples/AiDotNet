using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>ChatTTS (2noise, 2024) at a tiny size: the release's defaults and tokenizer, its sampling rules, what training
/// updates, and synthesis with and without a voice.</summary>
public class ChatTTSPaperTests
{
    private static ChatTTSOptions Tiny() => new()
    {
        HiddenDim = 16, NumHeads = 2, NumDecoderLayers = 1, IntermediateSize = 32, FsqLevels = new[] { 3, 3 }, DvaeDim = 8,
        DvaeEncoderHidden = 8, DvaeDecoderHidden = 8, DvaeLayers = 1, DvaeBottleneck = 8, DecoderDim = 8, DecoderHidden = 8,
        DecoderLayers = 1, DecoderBottleneck = 8, VocosDim = 8, VocosIntermediate = 16, VocosLayers = 1, FftSize = 64, HopSize = 16,
        MelChannels = 8, MaxNewRefineTokens = 3, MaxNewCodeTokens = 4, MaxTextLength = 32,
    };

    private static ChatTTS<double> Create(ChatTTSOptions? options = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
            { RandomSeed = 5 }, options ?? Tiny());

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) => layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheRelease()
    {
        await Task.Yield();
        var o = new ChatTTSOptions();
        Assert.Equal((768, 3072, 12, 20, 1e-6), (o.HiddenDim, o.IntermediateSize, o.NumHeads, o.NumDecoderLayers, o.RmsNormEpsilon));
        Assert.Equal((4, 2, 2, 625), (o.NumVq, o.FsqGroups, o.FsqResiduals, o.FsqLevels.Aggregate(1, (a, b) => a * b)));
        Assert.Equal((24000, 1024, 256, 100), (o.SampleRate, o.FftSize, o.HopSize, o.MelChannels));
        Assert.Equal((0.3, 0.7, 20, 1.05, 2048), (o.CodeTemperature, o.CodeTopP, o.CodeTopK, o.CodeRepetitionPenalty, o.MaxNewCodeTokens));
        Assert.Equal((0.7, 0.7, 20, 1.0, 384), (o.RefineTemperature, o.RefineTopP, o.RefineTopK, o.RefineRepetitionPenalty, o.MaxNewRefineTokens));
        Assert.True(o.RefineText && o.UseHiddenStateDecoder);
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenizer_IsBertBaseChinese_WithTheFiftyControlTokens()
    {
        await Task.Yield();
        using var model = Create();
        Assert.Equal(21178, model.TextVocabulary);
        // Control tokens take ids 21128.. in the release's order; text around them goes through WordPiece.
        var ids = model.Tokenize("[Stts][spk_emb]你好[uv_break]Hi[Ptts]");
        Assert.Equal(21131, ids[0]);
        Assert.Equal(21143, ids[1]);
        Assert.Equal(21137, ids[4]);
        Assert.Equal(21132, ids[^1]);
        // Each CJK character is its own word (bert-base-chinese: 你 872, 好 1962), and English is lowercased.
        Assert.Equal(new[] { 872, 1962 }, ids.Skip(2).Take(2));
        Assert.Equal(model.Tokenize("hi"), model.Tokenize("HI"));
    }

    [Fact(Timeout = 60000)]
    public async Task Sampling_FollowsTheReleasesWarpers()
    {
        await Task.Yield();
        var logits = Enumerable.Range(0, 30).Select(v => v * 0.5).ToArray();
        var random = new Random(3);
        var draws = Enumerable.Range(0, 400).Select(_ => ChatTTS<double>.Sample(logits, 1.0, 1.0, Array.Empty<int>(), 0.7, 20, -1, random)).ToList();
        // Top-p 0.7 over these logits keeps only the few most likely tokens, at least 3 (min_tokens_to_keep).
        Assert.True(draws.Distinct().Count() >= 3);
        Assert.All(draws, d => Assert.InRange(d, 26, 29));
        // A forbidden token (the end, before the first frame) is never drawn even when it is the most likely.
        Assert.DoesNotContain(29, Enumerable.Range(0, 200).Select(_ => ChatTTS<double>.Sample(logits, 1.0, 1.0, Array.Empty<int>(), 0.7, 20, 29, random)));
        // The repetition penalty lowers a repeated token: with 29 seen many times, 28 becomes the most likely.
        var penalized = Enumerable.Range(0, 400).Select(_ => ChatTTS<double>.Sample(logits, 1.0, 1.05, Enumerable.Repeat(29, 16), 0.7, 20, -1, random)).ToList();
        Assert.True(penalized.Count(d => d == 28) > penalized.Count(d => d == 29));
    }

    [Fact(Timeout = 120000)]
    public async Task Training_UpdatesTheGptOnly()
    {
        await Task.Yield();
        using var model = Create();
        var gpt = Snapshot(model.GptLayers);
        var codec = Snapshot(model.CodecLayers);
        var codes = new Tensor<double>(new[] { 6, 4 });
        for (int t = 0; t < 6; t++) for (int q = 0; q < 4; q++) codes[t, q] = (t + 2 * q) % 9;
        model.Train(new TtsTrainingSample<double> { Tokens = model.TextToTokens("你好 hello"), CodecTokens = codes });
        Assert.NotEqual(gpt, Snapshot(model.GptLayers));
        Assert.Equal(codec, Snapshot(model.CodecLayers));
    }

    [Theory(Timeout = 180000)]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public async Task Synthesis_IsDeterministic_WithAndWithoutAVoice(bool refine, bool hiddenDecoder)
    {
        await Task.Yield();
        var options = Tiny();
        options.RefineText = refine;
        options.UseHiddenStateDecoder = hiddenDecoder;
        using var model = Create(options);
        var first = model.Synthesize("你好 hello");
        Assert.True(first.Length > 0);
        Assert.Equal(first.ToArray(), model.Synthesize("你好 hello").ToArray());

        var recording = new Tensor<double>(new[] { 640 });
        for (int i = 0; i < recording.Length; i++) recording[i] = 0.2 * Math.Sin(i * 0.1);
        model.Voice = model.CreateVoice(model.SampleRandomSpeaker(new Random(1)), recording, "你好");
        Assert.Equal(21143, (int)model.TextToTokens("hi")[1]);              // [spk_emb] once the voice has an embedding
        var voiced = model.Synthesize("你好 hello");
        Assert.True(voiced.Length > 0);
    }

    [Fact(Timeout = 60000)]
    public async Task DvaeCodes_AreInTheCodebookAndDeterministic()
    {
        await Task.Yield();
        using var model = Create();
        var quantizer = model.Quantizer ?? throw new InvalidOperationException("native model");
        var audio = new Tensor<double>(new[] { 960 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.3 * Math.Sin(i * 0.07) + 0.1 * Math.Cos(i * 0.3);
        var codes = model.EncodeCodes(audio);
        Assert.NotEmpty(codes);
        Assert.All(codes, frame => Assert.All(frame, c => Assert.InRange(c, 0, quantizer.CodebookSize - 1)));
        Assert.Equal(codes, model.EncodeCodes(audio));
    }
}
