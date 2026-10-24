using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>SPEAR-TTS (Kharitonov et al. 2023) at a tiny size: the paper's defaults, its pretraining corruption, what each
/// training stage may update, S2's prompted sequence, and synthesis.</summary>
public class SPEARTTSPaperTests
{
    private const int Levels = 2, Codes = 16, SemanticCodes = 16;

    private static SPEARTTSOptions Tiny() => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 3, NumDecoderLayers = 3, ReadingFeedForwardDim = 32,
        ReadingKeyValueDim = 8, ReadingTrainableEncoderLayers = 1, BacktranslationTrainableDecoderLayers = 1,
        DropoutRate = 0.0, PretrainingDropoutRate = 0.0, SemanticCodes = SemanticCodes, SemanticFrameRate = 5,
        NumCodebooks = Levels, CodebookSize = Codes, SpeakingModelDim = 16, SpeakingLayers = 1, SpeakingHeads = 2,
        SpeakingKeyValueDim = 8, SpeakingFeedForwardDim = 32, SpeakingDropoutRate = 0.0, BeamSize = 2,
        MaxSemanticTokens = 4, MaxAcousticFrames = 4, MaxTextLength = 32, PromptSeconds = 0.4, HopSize = 1600,
        SoundStream = new SoundStreamOptions
        {
            SampleRate = 16000, Filters = 4, Dimension = 8, Ratios = [8, 5, 5, 4, 2], NumQuantizers = Levels,
            CodebookSize = Codes, TargetBandwidthKbps = Levels * 10 * 4 / 1000.0,
        },
    };

    private static SPEARTTS<double> Create(SPEARTTSOptions? options = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options ?? Tiny());

    private static int[,] CodeGrid(int frames)
    {
        var codes = new int[1 + Levels, frames];
        for (int t = 0; t < frames; t++)
        {
            codes[0, t] = (t / 2 * 7) % SemanticCodes;
            for (int l = 0; l < Levels; l++) codes[l + 1, t] = (3 * t + 5 * l + 1) % Codes;
        }
        return codes;
    }

    private static TtsTrainingSample<double> Sample(SPEARTTS<double> model, int frames = 16, string text = "hello there")
    {
        var grid = CodeGrid(frames);
        var tokens = new Tensor<double>(new[] { frames, 1 + Levels });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q <= Levels; q++) tokens[f, q] = grid[q, f];
        return new TtsTrainingSample<double> { Tokens = model.TextToTokens(text), CodecTokens = tokens };
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    private static IEnumerable<LayerBase<double>> Block(T5Block<double> block)
    {
        var a = block.SelfAttention;
        var layers = new List<LayerBase<double>> { block.SelfNorm, a.Query, a.Key, a.Value, a.Output, block.FeedForwardNorm, block.Wi, block.Wo };
        if (block.CrossAttention is { } c && block.CrossNorm is { } n) layers.AddRange(new LayerBase<double>[] { n, c.Query, c.Key, c.Value, c.Output });
        return layers;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new SPEARTTSOptions();
        // §7.1: 512 semantic clusters at 25 Hz; 3 SoundStream levels of 1024 codes at 50 Hz.
        Assert.Equal((512, 25, 3, 1024, 50), (o.SemanticCodes, o.SemanticFrameRate, o.NumCodebooks, o.CodebookSize, o.SampleRate / o.HopSize));
        // §7.2: T5-Large; deletion 0.6, dropout 0.5 in pretraining; label smoothing 0.1.
        Assert.Equal((1024, 4096, 64, 16, 24, 24), (o.HiddenDim, o.ReadingFeedForwardDim, o.ReadingKeyValueDim, o.NumHeads, o.NumEncoderLayers, o.NumDecoderLayers));
        Assert.Equal((0.6, 0.5, 0.1), (o.PretrainingDeletionProbability, o.PretrainingDropoutRate, o.LabelSmoothing));
        // §7.3: a 12-layer decoder-only Transformer, 12 heads of 64, width 768, feed-forward 2048.
        Assert.Equal((12, 12, 64, 768, 2048), (o.SpeakingLayers, o.SpeakingHeads, o.SpeakingKeyValueDim, o.SpeakingModelDim, o.SpeakingFeedForwardDim));
        // §7.4: beam 10 for S1, temperature 0.75 for S2; 3-second prompts.
        Assert.Equal((10, 0.75, 3.0), (o.BeamSize, o.Temperature, o.PromptSeconds));
        using var model = Create(new SPEARTTSOptions { SoundStream = new SoundStreamOptions { SampleRate = 16000, NumQuantizers = 12, CodebookSize = 1024, TargetBandwidthKbps = 1.5, Filters = 4, Dimension = 8 },
            HiddenDim = 8, NumEncoderLayers = 1, NumDecoderLayers = 1, ReadingFeedForwardDim = 8, ReadingKeyValueDim = 4, NumHeads = 2,
            ReadingTrainableEncoderLayers = 1, BacktranslationTrainableDecoderLayers = 1, SpeakingModelDim = 8, SpeakingLayers = 1,
            SpeakingHeads = 2, SpeakingKeyValueDim = 4, SpeakingFeedForwardDim = 8 });
        Assert.Equal(3, model.NumCodebooks);   // 1.5 kbps at 50 Hz selects the first three of the codec's 12 levels
    }

    [Fact(Timeout = 60000)]
    public async Task PretrainingCorruption_DeletesEachTokenIndependently_AndKeepsTheOrder()
    {
        await Task.Yield();
        using var model = Create();
        var ids = Enumerable.Range(0, 4000).Select(i => 100 + i % 50).ToArray();
        var corrupted = model.CorruptForPretraining(ids, new Random(11));
        Assert.Equal(1, corrupted[^1]);                                     // </s> closes the input
        var kept = corrupted.Take(corrupted.Length - 1).ToArray();
        // About 40% survive deletion at 0.6 (binomial standard deviation ~0.8% at n = 4000).
        Assert.InRange(kept.Length / 4000.0, 0.37, 0.43);
        int position = 0;
        foreach (int id in kept)
        {
            while (position < ids.Length && ids[position] != id) position++;
            Assert.True(position < ids.Length, "The kept tokens are a subsequence of the original, in order.");
            position++;
        }
        // Independent deletion, not an exact fraction: two seeds keep different numbers.
        var counts = Enumerable.Range(0, 6).Select(seed => model.CorruptForPretraining(ids.Take(200).ToArray(), new Random(seed)).Length).Distinct();
        Assert.True(counts.Count() > 1);
    }

    [Theory(Timeout = 180000)]
    [InlineData(SPEARTTSTrainingStage.Pretraining)]
    [InlineData(SPEARTTSTrainingStage.Reading)]
    [InlineData(SPEARTTSTrainingStage.Backtranslation)]
    [InlineData(SPEARTTSTrainingStage.Speaking)]
    public async Task EachStage_UpdatesWhatThePaperSays_AndNothingElse(SPEARTTSTrainingStage stage)
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = stage;
        var reading = model.ReadingModel ?? throw new InvalidOperationException("native model");
        var back = model.BacktranslationNetwork ?? throw new InvalidOperationException("native model");
        // T5 stores each stack's relative bias table in its first block, so the table goes with block 0.
        var groups = new Dictionary<string, List<LayerBase<double>>>
        {
            ["reading.encoderTable"] = new() { reading.EncoderEmbedding! },
            ["reading.lowerEncoder"] = Block(reading.Encoder[0]).ToList(),
            ["reading.encoderBias"] = new() { reading.EncoderBias },
            ["reading.upperEncoder"] = reading.Encoder.Skip(1).SelectMany(Block).Append(reading.EncoderFinalNorm).ToList(),
            ["reading.decoder"] = reading.Decoder.SelectMany(Block)
                .Concat(new LayerBase<double>[] { reading.Shared, reading.DecoderBias!, reading.DecoderFinalNorm! }).ToList(),
            ["back.encoder"] = back.Encoder.SelectMany(Block)
                .Concat(new LayerBase<double>[] { back.EncoderEmbedding!, back.EncoderBias, back.EncoderFinalNorm }).ToList(),
            ["back.outerDecoder"] = new[] { back.Decoder[0], back.Decoder[^1] }.SelectMany(Block)
                .Concat(new LayerBase<double>[] { back.Shared }).ToList(),
            ["back.decoderBias"] = new() { back.DecoderBias! },
            ["back.middleDecoder"] = Block(back.Decoder[1]).Append(back.DecoderFinalNorm!).ToList(),
            ["speaking"] = model.SpeakingLayers.ToList(),
        };
        // The groups cover every layer of the three networks exactly once.
        var covered = groups.Values.SelectMany(g => g).ToList();
        var all = model.ReadingLayers.Concat(model.BacktranslationLayers).Concat(model.SpeakingLayers).ToList();
        Assert.Equal(covered.Count, covered.Distinct().Count());
        Assert.Equal(all.Count, covered.Count);
        Assert.True(new HashSet<LayerBase<double>>(all).SetEquals(covered), "The groups name every layer.");
        var expected = stage switch
        {
            SPEARTTSTrainingStage.Pretraining => new[] { "reading.encoderTable", "reading.lowerEncoder", "reading.encoderBias", "reading.upperEncoder", "reading.decoder" },
            SPEARTTSTrainingStage.Reading => new[] { "reading.encoderTable", "reading.lowerEncoder", "reading.encoderBias" },
            SPEARTTSTrainingStage.Backtranslation => new[] { "back.outerDecoder", "back.decoderBias" },
            _ => new[] { "speaking" },
        };
        var before = groups.ToDictionary(g => g.Key, g => Snapshot(g.Value));
        for (int step = 0; step < 3; step++) model.Train(Sample(model));
        foreach (var group in groups)
        {
            bool moved = !before[group.Key].SequenceEqual(Snapshot(group.Value));
            Assert.True(moved == expected.Contains(group.Key), $"{stage}: {group.Key} {(moved ? "moved" : "stayed")}.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Backtranslation_StartsFromThePretrainedReader()
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = SPEARTTSTrainingStage.Pretraining;
        model.Train(Sample(model));
        Assert.NotEqual(Snapshot(model.ReadingLayers), Snapshot(model.BacktranslationLayers));
        model.StartBacktranslationFromPretrained();
        var pretrained = Snapshot(model.ReadingLayers);
        Assert.Equal(pretrained, Snapshot(model.BacktranslationLayers));
        // The copy is the values, not the tensors: training the backtranslation model leaves the reader as pretrained.
        model.CurrentStage = SPEARTTSTrainingStage.Backtranslation;
        model.Train(Sample(model));
        Assert.Equal(pretrained, Snapshot(model.ReadingLayers));
        Assert.NotEqual(pretrained, Snapshot(model.BacktranslationLayers));
    }

    [Fact(Timeout = 60000)]
    public async Task SpeakingSequence_IsPromptSemanticTargetSemanticPromptAcousticThenTarget()
    {
        await Task.Yield();
        using var model = Create();
        int frames = 16;
        var grid = CodeGrid(frames);
        for (int seed = 0; seed < 8; seed++)
        {
            var (ids, first) = model.SpeakingSequence(grid, new Random(seed));
            int begin = SemanticCodes + Levels * Codes, separator = begin + 1, end = begin + 2;
            Assert.Equal(begin, ids[0]);
            Assert.Equal(end, ids[^1]);
            int sep = ids.IndexOf(separator);
            Assert.True(sep > 1 && ids.LastIndexOf(separator) == sep, "One separator after the semantic tokens.");
            Assert.All(ids.Skip(1).Take(sep - 1), id => Assert.InRange(id, 0, SemanticCodes - 1));
            // A 0.4 s prompt at 10 frames a second is 4 frames of 2 levels; the target is the longer side, at least 6 frames.
            Assert.Equal(sep + 1 + 4 * Levels, first);
            int targetTokens = ids.Count - 1 - first;
            Assert.Equal(0, targetTokens % Levels);
            Assert.InRange(targetTokens / Levels, 6, 12);
            // Every acoustic token sits in its level's range, levels interleaved per frame.
            for (int i = sep + 1; i < ids.Count - 1; i++)
            {
                int level = (i - sep - 1) % Levels;
                Assert.InRange(ids[i], SemanticCodes + level * Codes, SemanticCodes + (level + 1) * Codes - 1);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_IsDeterministic_WholeCodecFrames_WithAndWithoutAVoice()
    {
        await Task.Yield();
        using var model = Create();
        var first = model.Synthesize("hello there");
        var second = model.Synthesize("hello there");
        Assert.True(first.Length > 0 && first.Length % 1600 == 0, $"{first.Length} samples");
        Assert.Equal(first.ToArray(), second.ToArray());

        var recording = new Tensor<double>(new[] { 16000 });
        for (int i = 0; i < recording.Length; i++) recording[i] = 0.3 * Math.Sin(i * 0.05);
        model.Voice = model.CreateVoice(recording, semanticTokens: new[] { 1, 1, 4, 9 });
        Assert.Equal(6400, model.Voice.ReferenceAudio!.Length);          // the first PromptSeconds (0.4 s) only
        var prompted = model.Synthesize("hello there");
        Assert.True(prompted.Length > 0 && prompted.Length % 1600 == 0, $"{prompted.Length} samples");
    }

    [Fact(Timeout = 60000)]
    public async Task Backtranscribe_ReadsOutPhonemes()
    {
        await Task.Yield();
        using var model = Create();
        var phonemes = model.Backtranscribe(new[] { 3, 3, 5, 8, 8, 2 });
        Assert.NotEmpty(phonemes);
        Assert.All(phonemes, p => Assert.Contains(p, AiDotNet.TextToSpeech.FrontEnd.LibriTtsPhonemeTable.Symbols));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Backtranscribe(new[] { SemanticCodes }));
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_IsRefused()
    {
        await Task.Yield();
        using var model = Create();
        Assert.ThrowsAny<Exception>(() => model.Train(model.TextToTokens("hi"), new Tensor<double>(new[] { 8, 80 })));
    }
}
