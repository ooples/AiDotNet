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

/// <summary>
/// SoundStorm (Borsos et al. 2023): the masking scheme of §3.2, the level-wise decoding of §3.3, the defaults of §4.1
/// and §5, and the two networks' separate training. The authors released no code, so these check the paper directly.
/// </summary>
public class SoundStormPaperTests
{
    private const int Levels = 3, Codes = 16;

    private static SoundStormOptions Tiny() => new()
    {
        HiddenDim = 16, NumHeads = 2, ConformerLayers = 1, ConformerHeadDim = 8, FeedForwardMultiplier = 2,
        NumCodebooks = Levels, CodebookSize = Codes, SemanticCodes = 16, SemanticFrameRate = 5, FirstLevelIterations = 4,
        TextModelDim = 16, TextFeedForwardDim = 32, TextKeyValueDim = 8, NumEncoderLayers = 1, NumDecoderLayers = 1,
        TextDropoutRate = 0.0, TopK = 8, MaxNewSemanticTokens = 4, MaxTextLength = 32, LearningRate = 3e-3,
        DropoutRate = 0.0, HopSize = 1600,
        SoundStream = new SoundStreamOptions
        {
            SampleRate = 16000, Filters = 4, Dimension = 8, Ratios = [8, 5, 5, 4, 2], NumQuantizers = Levels,
            CodebookSize = Codes, TargetBandwidthKbps = Levels * 10 * 4 / 1000.0,
        },
    };

    private static SoundStorm<double> Create(SoundStormOptions? options = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options ?? Tiny());

    private static int[,] CodeGrid(int frames)
    {
        var codes = new int[Levels, frames];
        for (int l = 0; l < Levels; l++)
            for (int t = 0; t < frames; t++) codes[l, t] = (3 * t + 5 * l + 1) % Codes;
        return codes;
    }

    private static int[] Semantic(int frames) => Enumerable.Range(0, frames).Select(t => (t / 2 * 7) % 16).ToArray();

    private static TtsTrainingSample<double> Sample(SoundStorm<double> model, int frames = 12)
    {
        var tokens = new Tensor<double>(new[] { frames, 1 + Levels });
        var semantic = Semantic(frames);
        var codes = CodeGrid(frames);
        for (int f = 0; f < frames; f++)
        {
            tokens[f, 0] = semantic[f];
            for (int l = 0; l < Levels; l++) tokens[f, l + 1] = codes[l, f];
        }
        return new TtsTrainingSample<double> { Tokens = model.TextToTokens("hello there"), CodecTokens = tokens };
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    private static double MaxDifference(Tensor<double> a, Tensor<double> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new SoundStormOptions();
        // §4.1: 12 layers, 16 heads, width 1024, feed-forward 4096, kernel 5; Q = 12 levels of 1024 codes at 50 Hz;
        // 1024 semantic tokens at 25 Hz; (16, 1, …, 1) decoding iterations.
        Assert.Equal((12, 16, 1024, 4096, 5), (o.ConformerLayers, o.NumHeads, o.HiddenDim, o.HiddenDim * o.FeedForwardMultiplier, o.ConvKernel));
        Assert.Equal((12, 1024, 50), (o.NumCodebooks, o.CodebookSize, o.SampleRate / o.HopSize));
        Assert.Equal((1024, 25, 16), (o.SemanticCodes, o.SemanticFrameRate, o.FirstLevelIterations));
        // §5: ByT5-large (36 + 12 layers, width 1536, feed-forward 3840), temperature 0.9, top-k 125.
        Assert.Equal((36, 12, 1536, 3840), (o.NumEncoderLayers, o.NumDecoderLayers, o.TextModelDim, o.TextFeedForwardDim));
        Assert.Equal((0.9, 125), (o.Temperature, o.TopK));
    }

    [Fact(Timeout = 60000)]
    public async Task Masking_HidesLevelQAfterThePromptAndEveryFinerLevel_ButReadsFinerLevelsInThePrompt()
    {
        await Task.Yield();
        using var model = Create();
        var core = model.AcousticCore ?? throw new InvalidOperationException("native model");
        int frames = 10, level = 1, prompt = 4;
        var masked = new bool[frames];
        for (int t = prompt; t < frames; t += 2) masked[t] = true;
        var visible = core.Visibility(frames, level, masked, prompt);
        for (int t = 0; t < frames; t++)
        {
            Assert.True(visible[0, t]);                                    // coarser levels: always read
            Assert.Equal(!masked[t], visible[level, t]);                   // level q: hidden where masked
            Assert.Equal(t < prompt, visible[2, t]);                       // finer levels: Y>t,>q masked after the prompt
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LevelLogits_IgnoreFinerLevelsAfterThePrompt_AndReadThemInsideIt()
    {
        await Task.Yield();
        using var model = Create();
        var core = model.AcousticCore ?? throw new InvalidOperationException("native model");
        int frames = 10, level = 1, prompt = 4;
        var semantic = Semantic(frames);
        var visible = core.Visibility(frames, level, null, prompt);
        var codes = CodeGrid(frames);
        var reference = core.Logits(codes, visible, semantic, level, null, training: false, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0));

        var laterChanged = (int[,])codes.Clone();
        for (int t = prompt; t < frames; t++) laterChanged[2, t] = (laterChanged[2, t] + 7) % Codes;
        Assert.Equal(0.0, MaxDifference(reference, core.Logits(laterChanged, visible, semantic, level, null, false, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0))));

        var promptChanged = (int[,])codes.Clone();
        promptChanged[2, 0] = (promptChanged[2, 0] + 7) % Codes;
        Assert.True(MaxDifference(reference, core.Logits(promptChanged, visible, semantic, level, null, false, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0))) > 1e-9);
    }

    [Fact(Timeout = 60000)]
    public async Task Decoding_KeepsThePrompt_AndFillsEveryLevelWithCodes()
    {
        await Task.Yield();
        using var model = Create();
        var core = model.AcousticCore ?? throw new InvalidOperationException("native model");
        int frames = 10, prompt = 3;
        var original = CodeGrid(frames);
        var codes = (int[,])original.Clone();
        for (int l = 0; l < Levels; l++)
            for (int t = prompt; t < frames; t++) codes[l, t] = core.MaskId;
        var again = (int[,])codes.Clone();
        core.Decode(codes, Semantic(frames), prompt, level => level == 0 ? 4 : 1, null, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3));
        core.Decode(again, Semantic(frames), prompt, level => level == 0 ? 4 : 1, null, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3));
        for (int l = 0; l < Levels; l++)
            for (int t = 0; t < frames; t++)
            {
                if (t < prompt) Assert.Equal(original[l, t], codes[l, t]);
                else Assert.InRange(codes[l, t], 0, Codes - 1);            // the mask id never survives decoding
                Assert.Equal(codes[l, t], again[l, t]);
            }
    }

    [Theory(Timeout = 120000)]
    [InlineData(SoundStormTrainingStage.TextToSemantic)]
    [InlineData(SoundStormTrainingStage.SemanticToAcoustic)]
    public async Task EachStage_TrainsOnlyItsOwnNetwork(SoundStormTrainingStage stage)
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = stage;
        var text = Snapshot(model.TextToSemanticLayers);
        var acoustic = Snapshot(model.SemanticToAcousticLayers);
        // An acoustic step can legitimately mask nothing (§3.2 draws t ~ U{0, T-1} and masks only frames after it, each
        // with the cosine-scheduled probability), and such a step has no loss to move anything. So train until the
        // stage's own network moves, and require the other one never to.
        bool textMoved = false, acousticMoved = false;
        for (int step = 0; step < 8 && !(textMoved || acousticMoved); step++)
        {
            model.Train(Sample(model));
            textMoved = !text.SequenceEqual(Snapshot(model.TextToSemanticLayers));
            acousticMoved = !acoustic.SequenceEqual(Snapshot(model.SemanticToAcousticLayers));
        }
        Assert.Equal(stage == SoundStormTrainingStage.TextToSemantic, textMoved);
        Assert.Equal(stage == SoundStormTrainingStage.SemanticToAcoustic, acousticMoved);
    }

    [Fact(Timeout = 60000)]
    public async Task TextToSemanticTraining_LeavesTheByT5EncoderFrozen()
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = SoundStormTrainingStage.TextToSemantic;
        // §5 trains only the decoder over a text-pretrained encoder.
        var t5 = model.TextModel ?? throw new InvalidOperationException("native model");
        var encoderLayers = new List<LayerBase<double>> { t5.EncoderBias, t5.EncoderFinalNorm };
        foreach (var block in t5.Encoder)
        {
            var a = block.SelfAttention;
            encoderLayers.AddRange(new LayerBase<double>[] { block.SelfNorm, a.Query, a.Key, a.Value, a.Output, block.FeedForwardNorm, block.Wi, block.Wo });
            if (block.WiLinear is not null) encoderLayers.Add(block.WiLinear);
        }
        var before = Snapshot(encoderLayers);
        var decoderBefore = Snapshot(t5.Decoder.Select(b => (LayerBase<double>)b.Wo));
        model.Train(Sample(model));
        Assert.Equal(before, Snapshot(encoderLayers));
        Assert.NotEqual(decoderBefore, Snapshot(t5.Decoder.Select(b => (LayerBase<double>)b.Wo)));
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_IsDeterministic_AndWholeCodecFrames()
    {
        await Task.Yield();
        using var model = Create();
        var first = model.Synthesize("hello there");
        var second = model.Synthesize("hello there");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.True(first.Length > 0);
        Assert.Equal(0, first.Length % 1600);
    }
}
