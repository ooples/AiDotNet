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
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VALL-E (Wang et al. 2023) as a model: its phoneme vocabulary, training each codec language model on its own, and
/// prompted synthesis. The networks themselves are checked against the reference in <see cref="VallECoreParityTests"/>.
/// </summary>
public class VALLEPaperTests
{
    private static VALLEOptions Tiny() => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, FeedForwardDim = 32, TextTokens = 128,
        NumCodebooks = 3, CodebookSize = 64, MaxCodesPerTextToken = 2, LearningRate = 3e-3, WarmupSteps = 0, DropoutRate = 0.0,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 3, CodebookSize = 64, Filters = 4, Ratios = [8, 5, 4, 2], Dimension = 8,
            ResidualKernelSizes = [3, 1],
            // 75 frames a second of log2(64) bits each: three codebooks.
            TargetBandwidthKbps = 3 * 75 * 6 / 1000.0,
        },
    };

    private static VALLE<double> Create(VALLEOptions options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    private static Tensor<double> Tone(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * 180.0 * i / 24000) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / 24000);
        return audio;
    }

    private static TtsTrainingSample<double> Sample(VALLE<double> valle, int frames = 12)
    {
        var codes = new Tensor<double>(new[] { frames, 3 });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < 3; q++) codes[f, q] = (5 * f + 11 * q) % 64;
        var phonemes = valle.EncodePhonemes(EnglishG2P.Default.Phonemize("hello there"));
        return new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { phonemes.Length }, new Vector<double>(phonemes.Select(p => (double)p).ToArray())),
            CodecTokens = codes,
        };
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Vocabulary_IsTheReferenceCollatersOrder()
    {
        await Task.Yield();
        // [<pad>, <bos>, <eos>] + sorted(the LibriTTS phoneme table): 93 tokens (lifeiteng/vall-e TextTokenCollater).
        using var valle = Create(Tiny());
        Assert.Equal(93, ((VALLEOptions)valle.GetOptions()).VocabSize);
        var expected = new Dictionary<string, int>
        {
            ["!"] = 3, ["<eps>"] = 11, ["_"] = 13, ["aɪ"] = 14, ["h"] = 29, ["l"] = 35, ["oʊ"] = 39, ["ə"] = 65, ["ɹ"] = 80,
            ["—"] = 92,
        };
        foreach (var (symbol, id) in expected)
            Assert.Equal(new[] { id }, valle.EncodePhonemes(new[] { symbol }));
        Assert.Equal(new[] { 29, 65, 35, 39 }, valle.EncodePhonemes(EnglishG2P.Default.Phonemize("hello")));
        // The reference asserts every symbol is in the table; so does the port.
        Assert.Throws<ArgumentException>(() => valle.EncodePhonemes(new[] { "q" }));
    }

    [Fact(Timeout = 180000)]
    public async Task EachStage_TrainsItsOwnModelOnly()
    {
        await Task.Yield();
        using var valle = Create(Tiny());
        var sample = Sample(valle);
        foreach (var stage in new[] { VallETrainingStage.AutoRegressive, VallETrainingStage.NonAutoRegressive })
        {
            valle.CurrentStage = stage;
            var own = stage == VallETrainingStage.AutoRegressive ? valle.AutoRegressiveLayers : valle.NonAutoRegressiveLayers;
            var other = stage == VallETrainingStage.AutoRegressive ? valle.NonAutoRegressiveLayers : valle.AutoRegressiveLayers;
            var ownBefore = Snapshot(own);
            var otherBefore = Snapshot(other);
            valle.Train(sample);
            Assert.NotEqual(ownBefore, Snapshot(own));
            Assert.True(otherBefore.SequenceEqual(Snapshot(other)), $"{stage} changed the other model.");
        }
    }

    [Theory(Timeout = 300000)]
    [InlineData(VallETrainingStage.AutoRegressive)]
    [InlineData(VallETrainingStage.NonAutoRegressive)]
    public async Task Training_LowersTheStageObjective(VallETrainingStage stage)
    {
        await Task.Yield();
        using var valle = Create(Tiny());
        valle.CurrentStage = stage;
        var sample = Sample(valle);
        double before = valle.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) valle.Train(sample);
        double after = valle.EvaluateTrainingObjective(sample);
        Assert.True(after < 0.8 * before, $"{stage}: the objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 180000)]
    public async Task Synthesis_ContinuesThePromptAndIsDeterministic()
    {
        await Task.Yield();
        using var valle = Create(Tiny());
        // Train the AR model a little so it continues past the prompt rather than ending at once.
        var sample = Sample(valle);
        for (int i = 0; i < 20; i++) valle.Train(sample);
        Assert.Throws<InvalidOperationException>(() => valle.Synthesize("hello"));

        valle.Voice = valle.CreateVoice(Tone(24000 / 3), "a prompt");
        var first = valle.Synthesize("hello there");
        var second = valle.Synthesize("hello there");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.All(first.ToArray(), v => Assert.True(double.IsFinite(v)));
        // Only the new frames are returned, 320 samples each.
        Assert.True(first.Length > 0 && first.Length % 320 == 0, $"{first.Length} samples.");

        // The text reaches the audio: a different sentence synthesizes differently.
        var other = valle.Synthesize("a completely different sentence");
        Assert.True(other.Length != first.Length || Enumerable.Range(0, first.Length).Any(i => other[i] != first[i]),
            "Different text synthesized identical audio.");
    }

    [Fact(Timeout = 60000)]
    public async Task Configuration_MustMatchTheCodec()
    {
        await Task.Yield();
        var codebooks = Tiny();
        codebooks.NumCodebooks = 2;
        Assert.Throws<ArgumentException>(() => Create(codebooks));
        var hop = Tiny();
        hop.HopSize = 256;
        Assert.Throws<ArgumentException>(() => Create(hop));
        var depth = Tiny();
        depth.NumEncoderLayers = 2;
        Assert.Throws<ArgumentException>(() => Create(depth));
    }
}
