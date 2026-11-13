using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// FireRedTTS's streamable decoder (§3.3.2), its training stages and instruction tuning: the Mel codec against SoCodec's
/// code (tools/reference-data/fireredtts_codec_reference.py), its causality, the acoustic LM's lookahead conditioning,
/// synthesis through the streamable path, every training stage's objective falling on a repeated utterance, and the
/// paralinguistic markup.
/// </summary>
public class FireRedTtsStreamableTests
{
    private static IEngine Engine => AiDotNetEngine.Current;
    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
    private static Tensor<double> Tensor(JToken t, params int[] shape) => new(shape, new Vector<double>(Doubles(t)));

    private static JObject Fixture(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        var e = Doubles(expected);
        var a = actual.ToArray();
        Assert.True(e.Length == a.Length, $"{what}: {e.Length} values expected, {a.Length} computed");
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {a[i]}");
    }

    private static FireRedTtsMelCodec<double> SoCodecSized(out JObject f)
    {
        var fixture = Fixture("fireredtts_codec_reference.json");
        var codec = new FireRedTtsMelCodec<double>(Engine, new List<LayerBase<double>>(),
            new FireRedTtsMelCodecConfiguration(100, 32, new[] { 2 }, 8, 4, 4, 32, 0.99, 1.0), new Random(1));
        var state = (JObject)fixture["state"]!;
        codec.Load((name, _) => Doubles(state[name] ?? throw new InvalidDataException($"No tensor '{name}'.")), name => state[name] is not null,
            "encoder.", "decoder.", "quantizer.");
        f = fixture;
        return codec;
    }

    [Fact]
    public void MelCodec_MatchesSoCodec()
    {
        using var _ = new NoGradScope<double>();
        var codec = SoCodecSized(out var f);
        var mel = Tensor(f["mel"]!, 18, 100);
        var encoded = codec.Encode(mel);
        Close(f["encodings"]!, encoded, "encodings");
        var tokens = codec.Tokenize(mel);
        var want = f["tokens"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        for (int t = 0; t < want.Length; t++)
            for (int g = 0; g < 4; g++) Assert.Equal(want[t][g], tokens[t, g]);
        var quantized = codec.Quantizer.Decode(tokens);
        Close(f["quantized"]!, quantized, "quantized");
        Close(f["decoded"]!, codec.Decode(quantized, Tensor(f["speaker"]!, 32)), "decoded mel");
        var fixedTokens = f["fixed_tokens"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var grid = new int[fixedTokens.Length, 4];
        for (int t = 0; t < fixedTokens.Length; t++)
            for (int g = 0; g < 4; g++) grid[t, g] = fixedTokens[t][g];
        Close(f["fixed_quantized"]!, codec.Quantizer.Decode(grid), "decoded tokens");
    }

    [Fact]
    public void MelCodec_IsCausal()
    {
        using var _ = new NoGradScope<double>();
        var codec = SoCodecSized(out var f);
        var mel = Tensor(f["mel"]!, 18, 100);
        var changed = mel.Clone();
        for (int c = 0; c < 100; c++) changed[17, c] = changed[17, c] + 3.0;          // only the last frame
        var a = codec.Encode(mel);
        var b = codec.Encode(changed);
        for (int t = 0; t < 8; t++)                                                   // codec frames before frame 8 read frames ≤ 15
            for (int i = 0; i < 32; i++) Assert.Equal(a[t, i], b[t, i]);
        var speaker = Tensor(f["speaker"]!, 32);
        var qa = codec.Quantizer.Project(a);
        var decodedA = codec.Decode(codec.Quantizer.ProjectOut.Forward(qa), speaker);
        var qb = qa.Clone();
        for (int i = 0; i < qb.Shape[1]; i++) qb[8, i] = qb[8, i] + 1.0;              // only the last codec frame
        var decodedB = codec.Decode(codec.Quantizer.ProjectOut.Forward(qb), speaker);
        for (int t = 0; t < 16; t++)
            for (int c = 0; c < 100; c++) Assert.Equal(decodedA[t, c], decodedB[t, c]);
    }

    [Fact]
    public void AcousticLm_SemanticTokensArriveWithTheirLookahead()
    {
        using var _ = new NoGradScope<double>();
        var lm = new FireRedTtsAcousticLm<double>(Engine, new List<LayerBase<double>>(), new FireRedTtsAcousticLmConfiguration(4, 16, 16, 16, 2, 1,
            32, 0.0, new[] { 0, 1, 2, 3 }, 3, 8, 6), new Random(2));
        int[] semantic = { 3, 9, 1, 14 };
        int steps = 2 * semantic.Length + 1;
        var rows = lm.Conditioning(semantic, steps);
        for (int s = 0; s < steps; s++)
        {
            int frame = s - 1 - 3;
            for (int i = 0; i < 16; i++)
            {
                double expected = frame >= 0 && frame < 8 ? lm.Semantic.Forward(IndexIds(semantic[frame / 2]))[0, i] : 0.0;
                Assert.Equal(expected, rows[s, i], 12);
            }
        }
        // Changing token 2 (frames 4 and 5, steps 8 and 9) leaves every earlier step's logits as they were.
        var speaker = new Tensor<double>(new[] { 8 });
        var codes = new int[4, 8];
        for (int q = 0; q < 4; q++)
            for (int t = 0; t < 8; t++) codes[q, t] = (3 * q + t) % 16;
        var sequence = AiDotNet.Audio.MusicGen.MusicGenDelayPattern.TrainingSequence(codes, new[] { 0, 1, 2, 3 }, 16);
        var before = lm.Core.Logits(sequence, lm.Prefix(speaker), false, new Random(1), lm.Conditioning(semantic, 9));
        var after = lm.Core.Logits(sequence, lm.Prefix(speaker), false, new Random(1), lm.Conditioning(new[] { 3, 9, 7, 14 }, 9));
        bool laterDiffers = false;
        for (int q = 0; q < 4; q++)
            for (int s = 0; s < 9; s++)
                for (int v = 0; v < 16; v++)
                {
                    if (s < 8) Assert.Equal(before[q, s, v], after[q, s, v]);
                    else laterDiffers |= before[q, s, v] != after[q, s, v];
                }
        Assert.True(laterDiffers, "The changed token never reached the logits.");
    }

    private static Tensor<double> IndexIds(params int[] ids) =>
        new(new[] { ids.Length }, new Vector<double>(ids.Select(v => (double)v).ToArray()));

    private static FireRedTTS<double> Tiny(Action<FireRedTTSOptions>? configure = null)
    {
        var options = FireRedTtsFixture.Options();
        configure?.Invoke(options);
        return new FireRedTTS<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1), options);
    }

    private static Tensor<double> Recording(int samples, double hz)
    {
        var r = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) r[i] = 0.4 * Math.Sin(2 * Math.PI * hz * i / 24000) + 0.1 * Math.Sin(2 * Math.PI * 3.1 * hz * i / 24000);
        return r;
    }

    [Fact]
    public void Streamable_Synthesis_RunsThePaperPath()
    {
        var model = Tiny(o => o.MelDecoder = FireRedTtsMelDecoder.Streamable);
        model.Voice = model.CreateVoice(Recording(24000, 180), 24000);
        var tokens = new Tensor<double>(new[] { 5, 1 });
        for (int i = 0; i < 5; i++) tokens[i, 0] = 7 * i + 3;
        var audio = model.DecodeFromTokens(tokens);
        Assert.Equal(5 * 960, audio.Length);
        Assert.True(audio.ToArray().All(v => !double.IsNaN(v)));
    }

    [Theory]
    [InlineData(FireRedTtsStage.TextToSpeech)]
    [InlineData(FireRedTtsStage.FlowDecoder)]
    [InlineData(FireRedTtsStage.SpeechTokenizer)]
    [InlineData(FireRedTtsStage.MelCodec)]
    [InlineData(FireRedTtsStage.AcousticLanguageModel)]
    public void EveryStage_LearnsARepeatedUtterance(FireRedTtsStage stage)
    {
        var model = Tiny(o =>
        {
            o.Stage = stage;
            o.LearningRate = 3e-3;
            o.FlowWarmupSteps = 1;
            o.AcousticLmWarmupSteps = 1;
            o.DiscriminatorWindows = new[] { 16 };
            o.DiscriminatorHidden = 8;
            o.FlowDropout = 0;
            o.DropoutRate = 0;
            o.TrainingCfgRate = 0;
        });
        var audio = Recording(24000, 210);
        var ids = model.TokenizeText("the quick brown fox");
        var codes = new Tensor<double>(new[] { 25, 1 });
        for (int i = 0; i < 25; i++) codes[i, 0] = (11 * i + 5) % 64;
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { ids.Length }, new Vector<double>(ids.Select(v => (double)v).ToArray())),
            CodecTokens = codes,
            Audio = audio,
            SpeakerReference = Recording(12000, 160),
        };
        double first = Convert.ToDouble(model.EvaluateTrainingObjective(sample));
        for (int i = 0; i < 12; i++) model.Train(sample);
        double last = Convert.ToDouble(model.EvaluateTrainingObjective(sample));
        Assert.True(!double.IsNaN(first) && !double.IsInfinity(first) && !double.IsNaN(last) && !double.IsInfinity(last), $"{stage}: {first} → {last}");
        Assert.True(last < first, $"{stage}: the objective went from {first} to {last}.");
    }

    [Fact]
    public void InstructionMarkup_LabelsTheMarkedCharacters()
    {
        var tokenizer = FireRedTtsTokenizer.Get(instructionTokens: true);
        const string text = "真是笑ˆ死ˆ我了[laugh]你真@棒@";       // 真是笑ˆ死ˆ我了[laugh]你真@棒@
        string stripped = FireRedTtsInstruction.Strip(text);
        Assert.Equal("真是笑死我了[laugh]你真棒", stripped);
        var (ids, labels) = FireRedTtsInstruction.Encode(tokenizer, stripped, FireRedTtsInstruction.Parse(text));
        Assert.NotNull(labels);
        Assert.Equal(ids.Length, labels!.Length);
        Assert.Contains(tokenizer.Special("[laugh]"), ids);
        // Two laugh-marked characters (label 1) and two emphasized ones (label 2), the rest unlabeled.
        Assert.True(labels.Count(l => l == 1) >= 2 && labels.Count(l => l == 2) >= 2, string.Join(",", labels));
        Assert.Equal(0, labels[0]);                                                   // the language tag
        var model = Tiny(o => o.InstructionTuning = true);
        model.Voice = model.CreateVoice(Recording(24000, 180), 24000);
        var audio = model.SynthesizeInstruction(text, FireRedTtsEmotion.Happy);
        Assert.True(audio.Length > 0);
    }
}
