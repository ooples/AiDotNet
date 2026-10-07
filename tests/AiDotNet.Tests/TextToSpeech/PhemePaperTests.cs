using System;
using System.Collections.Generic;
using System.IO;
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
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Pheme (Budzianowski et al. 2024) as its reference implementation builds it: the T5 text-to-semantic model over the
/// USLM phoneme table and the semantic codes, the SoundStorm acoustic model and SpeechTokenizer around them. The
/// component networks have their own parity tests (T5Seq2Seq, SoundStormConformer, PyannoteXVector, SpeechTokenizer);
/// these check how Pheme puts them together.
/// </summary>
public class PhemePaperTests
{
    private static PhemeOptions Tiny(int codes = 16) => new()
    {
        HiddenDim = 16, TextModelDim = 16, TextFeedForwardDim = 32, TextKeyValueDim = 8, NumHeads = 2,
        NumEncoderLayers = 1, NumDecoderLayers = 1, AcousticLayers = 1, AcousticHeads = 2, AcousticHeadDim = 8,
        AcousticCodebooks = 2, CodebookSize = codes, SemanticCodes = codes, MaxNewSemanticTokens = 6, TopK = 8,
        MaskGitSteps = 3, LearningRate = 3e-3, TextWarmupSteps = 0, AcousticWarmupSteps = 0, DropoutRate = 0.0,
        SpeechTokenizer = new SpeechTokenizerOptions
        {
            Filters = 4, Dimension = 8, SemanticDimension = 6, LstmLayers = 1, NumQuantizers = 3, CodebookSize = codes,
            // 50 frames a second of log2(codes) bits each: three codebooks.
            TargetBandwidthKbps = 3 * 50 * Math.Log(codes, 2) / 1000.0,
        },
    };

    private static Pheme<double> Create(PhemeOptions options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    private static Tensor<double> Tone(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * 180.0 * i / 16000) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / 16000);
        return audio;
    }

    private static TtsTrainingSample<double> Sample(Pheme<double> pheme, int frames = 6)
    {
        var codes = new Tensor<double>(new[] { frames, 3 });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < 3; q++) codes[f, q] = (3 * f + 5 * q) % 16;
        var phonemes = pheme.EncodePhonemes(EnglishG2P.Default.Phonemize("hello there"));
        return new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { phonemes.Length }, new Vector<double>(phonemes.Select(p => (double)p).ToArray())),
            CodecTokens = codes,
            SpeakerReference = Tone(8000),
        };
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 120000)]
    public async Task Vocabulary_MatchesTheReferenceTokenizer()
    {
        await Task.Yield();
        // [<pad>, <bos>, <eos>, spkr_1, spkr_2] + sorted(USLM symbols ∪ "0" … "1023"): 1119 tokens, the reference's
        // ids (Python's sorted() is code-point order, so "10" precedes "9" and the digits precede "<eps>").
        using var pheme = Create(Tiny(codes: 1024));
        Assert.Equal(1119, pheme.GetOptions() is PhemeOptions o ? o.VocabSize : 0);
        var expected = new Dictionary<string, int>
        {
            ["!"] = 5, ["0"] = 11, ["10"] = 13, ["1023"] = 40, ["9"] = 924, ["<eps>"] = 1037, ["_"] = 1039, ["aɪ"] = 1040,
            ["ə"] = 1091, ["ɹ"] = 1106, ["̃"] = 1114, ["ᵻ"] = 1117, ["—"] = 1118,
        };
        foreach (var (symbol, id) in expected)
            Assert.Equal(new[] { id }, pheme.EncodePhonemes(new[] { symbol }));
        // The reference's "hello" (espeak-ng en-us, USLM tokenizer): h ə l oʊ.
        Assert.Equal(new[] { 1055, 1091, 1061, 1065 }, pheme.EncodePhonemes(EnglishG2P.Default.Phonemize("hello")));
        // Special tokens are not phonemes, and unknown symbols are dropped as the reference drops them.
        Assert.Empty(pheme.EncodePhonemes(new[] { "<bos>", "spkr_1", "q" }));
    }

    [Fact(Timeout = 180000)]
    public async Task EachStage_TrainsItsOwnNetworkOnly()
    {
        await Task.Yield();
        using var pheme = Create(Tiny());
        var sample = Sample(pheme);
        var others = pheme.SpeakerEncoderLayers.Append(pheme.CodecLayer!).ToList();

        foreach (var stage in new[] { PhemeTrainingStage.TextToSemantic, PhemeTrainingStage.SemanticToAcoustic })
        {
            pheme.CurrentStage = stage;
            var own = stage == PhemeTrainingStage.TextToSemantic ? pheme.TextToSemanticLayers : pheme.SemanticToAcousticLayers;
            var other = stage == PhemeTrainingStage.TextToSemantic ? pheme.SemanticToAcousticLayers : pheme.TextToSemanticLayers;
            var ownBefore = Snapshot(own);
            var otherBefore = Snapshot(other);
            var frozenBefore = others.Select(l => Snapshot(new[] { l })).ToList();
            pheme.Train(sample);
            Assert.NotEqual(ownBefore, Snapshot(own));
            Assert.True(otherBefore.SequenceEqual(Snapshot(other)), $"{stage} changed the other network.");
            for (int i = 0; i < others.Count; i++)
                Assert.True(frozenBefore[i].SequenceEqual(Snapshot(new[] { others[i] })),
                    $"{stage} changed frozen {others[i].GetType().Name} #{i} of {others.Count}.");
        }
        Assert.Equal(1, pheme.TextToSemanticUpdates);
        Assert.Equal(1, pheme.SemanticToAcousticUpdates);
    }

    [Theory(Timeout = 300000)]
    [InlineData(PhemeTrainingStage.TextToSemantic)]
    [InlineData(PhemeTrainingStage.SemanticToAcoustic)]
    public async Task Training_LowersTheStageObjective(PhemeTrainingStage stage)
    {
        await Task.Yield();
        using var pheme = Create(Tiny());
        pheme.CurrentStage = stage;
        var sample = Sample(pheme);
        double before = pheme.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) pheme.Train(sample);
        double after = pheme.EvaluateTrainingObjective(sample);
        Assert.True(after < 0.8 * before, $"{stage}: the objective did not fall ({before} -> {after}).");
    }

    [Theory(Timeout = 180000)]
    [InlineData(PhemeAcousticDecoding.MaskGitFirstLevel)]
    [InlineData(PhemeAcousticDecoding.MaskGitRemainingLevels)]
    public async Task Synthesis_ContinuesThePromptAndIsDeterministic(PhemeAcousticDecoding decoding)
    {
        await Task.Yield();
        var options = Tiny();
        options.AcousticDecoding = decoding;
        using var pheme = Create(options);
        Assert.Throws<InvalidOperationException>(() => pheme.Synthesize("hello"));

        var prompt = Tone(16000);
        pheme.Voice = pheme.CreateVoice(prompt, "a prompt");
        var first = pheme.Synthesize("hello there");
        var second = pheme.Synthesize("hello there");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.All(first.ToArray(), v => Assert.True(double.IsFinite(v)));

        // The output is the regenerated last prompt frame plus at most MaxNewSemanticTokens frames, 320 samples each.
        Assert.Equal(0, first.Length % 320);
        int frames = first.Length / 320;
        Assert.InRange(frames, 1, 1 + 6);

        var other = pheme.Synthesize("a completely different sentence");
        Assert.True(other.Length != first.Length || Enumerable.Range(0, first.Length).Any(i => other[i] != first[i]));
    }

    [Fact(Timeout = 120000)]
    public async Task AcousticModel_MatchesTheReferenceAfterLoadingItsCheckpointLayout()
    {
        await Task.Yield();
        // ReferenceData/pheme_s2a_reference.json: a seeded TTSConformer from Pheme's code, its float32 state dictionary
        // under s2a.ckpt's "model." prefix, and its float64 logits per codebook level on the training path (the positions
        // its own mask chose, which it records) for fixed codes and a speaker embedding
        // (tools/reference-data/pheme_s2a_reference.py).
        string file = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "pheme_s2a_reference.json");
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(file) && dir is not null; dir = dir.Parent)
            file = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "pheme_s2a_reference.json");
        var fixture = JObject.Parse(File.ReadAllText(file));

        var options = Tiny();
        options.AcousticHeadDim = 64;                       // the reference fixes dim_head = 64
        options.SpeakerEmbeddingDropout = 0.0;
        using var pheme = Create(options);
        string path = Path.Combine(Path.GetTempPath(), $"pheme_s2a_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try { pheme.LoadSemanticToAcousticWeights(path); }
        finally { File.Delete(path); }

        int time = (int)fixture["time"]!;
        var rows = fixture["acoustic"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var acoustic = new int[2, time];
        for (int t = 0; t < time; t++)
            for (int l = 0; l < 2; l++) acoustic[l, t] = rows[t][l];
        var semantic = fixture["semantic"]!.Select(v => (int)v).ToArray();
        var speakerValues = fixture["speaker"]!.Select(v => (double)v).ToArray();
        var speaker = new Tensor<double>(new[] { speakerValues.Length }, new Vector<double>(speakerValues));

        pheme.SetTrainingMode(false);
        for (int level = 0; level < 2; level++)
        {
            var masked = new bool[time];
            foreach (var t in fixture["masked"]![level]!) masked[(int)t] = true;
            var logits = pheme.AcousticLogits(acoustic, semantic, level, masked, speaker, training: false, new Random(0));
            var expected = fixture["logits"]![level]!.Select(v => (double)v).ToArray();
            Assert.Equal(expected.Length, logits.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - logits[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                    $"level {level} logits[{i}]: reference {expected[i]:R}, port {logits[i]:R}.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Configuration_MustMatchTheCodec()
    {
        await Task.Yield();
        var mismatched = Tiny();
        mismatched.AcousticCodebooks = 3;
        Assert.Throws<ArgumentException>(() => Create(mismatched));
        var codes = Tiny();
        codes.SemanticCodes = 32;
        Assert.Throws<ArgumentException>(() => Create(codes));
        var hop = Tiny();
        hop.HopSize = 256;                                  // the codec's hop is 320
        Assert.Throws<ArgumentException>(() => Create(hop));
    }
}
