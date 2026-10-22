using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Effects;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.DescriptionBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Parler-TTS reproduces the open reproduction of Lyth &amp; King (2024) (huggingface/parler-tts
/// <c>ParlerTTSForConditionalGeneration</c>): the delayed training labels, the teacher-forced logits, the loss and each
/// codebook's loss, and greedy generation, for the released configuration and for the rotary and cross-attended
/// transcript variants.
/// </summary>
/// <remarks><c>ReferenceData/parler_tts_reference.json</c> holds tiny seeded models' float32 weights and their float64
/// results (<c>tools/reference-data/parler_tts_reference.py</c>).</remarks>
public class ParlerTTSParityTests
{
    private static JObject Reference()
    {
        const string fileName = "parler_tts_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static ParlerTTSOptions Options(JObject fixture, JToken testCase)
    {
        var decoder = fixture["decoder"]!;
        var encoder = fixture["text_encoder"]!;
        return new ParlerTTSOptions
        {
            HiddenDim = (int)decoder["hidden_size"]!,
            NumHeads = (int)decoder["num_attention_heads"]!,
            KeyValueHeads = (int)decoder["num_key_value_heads"]!,
            CrossAttentionKeyValueHeads = (int)decoder["num_cross_attention_key_value_heads"]!,
            NumDecoderLayers = (int)decoder["num_hidden_layers"]!,
            FeedForwardDim = (int)decoder["ffn_dim"]!,
            NumCodebooks = (int)decoder["num_codebooks"]!,
            AudioVocabulary = (int)decoder["vocab_size"]!,
            CodebookSize = (int)fixture["codebook_size"]!,
            PromptVocabulary = (int)fixture["prompt_vocabulary"]!,
            RopeTheta = (double)decoder["rope_theta"]!,
            DropoutRate = 0.0,
            RotaryEmbeddings = (bool)testCase["rope_embeddings"]!,
            PromptCrossAttention = (bool)testCase["prompt_cross_attention"]!,
            TextEncoderDim = (int)encoder["d_model"]!,
            TextEncoderKeyValueDim = (int)encoder["d_kv"]!,
            TextEncoderFeedForwardDim = (int)encoder["d_ff"]!,
            TextEncoderLayers = (int)encoder["num_layers"]!,
            TextEncoderHeads = (int)encoder["num_heads"]!,
            TextEncoderVocabulary = (int)encoder["vocab_size"]!,
            TextEncoderDropout = 0.0,
            MaxLength = (int)fixture["max_length"]!,
            MinNewTokens = (int)fixture["min_new_tokens"]!,
            DoSample = false,
            // A codec with the fixture's three 16-code codebooks: hop 8 at 16 kHz, 8 kbps a codebook.
            SampleRate = 16000,
            HopSize = 8,
            Codec = new DACOptions
            {
                SampleRate = 16000, EncoderDim = 4, EncoderRates = [2, 4], DecoderDim = 16, DecoderRates = [4, 2],
                NumQuantizers = 3, CodebookSize = 16, CodebookDim = 2, TargetBandwidthKbps = 24.0,
            },
        };
    }

    private static (ParlerTTS<double> Model, JObject Fixture, JToken Case) Load(string caseName)
    {
        var fixture = Reference();
        var testCase = fixture["cases"]![caseName]!;
        var model = new ParlerTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), Options(fixture, testCase));
        string path = Path.Combine(Path.GetTempPath(), $"parler_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)testCase["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadReferenceWeights((name, shape) =>
            {
                Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
                var tensor = weights[name];
                Assert.Equal(shape, tensor.Shape.ToArray());
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            }, weights.ContainsKey);
            // Every reference tensor has a place in the port (the encoder's embed_tokens is its shared table, tied).
            var unused = weights.Keys.Where(k => !used.Contains(k) && k != "text_encoder.encoder.embed_tokens.weight").ToList();
            Assert.True(unused.Count == 0, "Unread reference tensors: " + string.Join(", ", unused));
        }
        finally
        {
            File.Delete(path);
        }
        return (model, fixture, testCase);
    }

    private static int[] Ints(JToken values) => values.Select(v => (int)v).ToArray();

    private static int[,] Matrix(JToken rows)
    {
        var list = rows.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var m = new int[list.Length, list.Length == 0 ? 0 : list[0].Length];
        for (int i = 0; i < list.Length; i++)
            for (int j = 0; j < list[i].Length; j++) m[i, j] = list[i][j];
        return m;
    }

    private static void Close(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-8 * Math.Max(1, Math.Abs(expected)),
            $"{what}: reference {expected:R}, port {actual:R}.");

    [Theory(Timeout = 120000)]
    [InlineData("released")]
    [InlineData("variants")]
    public async Task TeacherForcedLogitsAndLosses_MatchTheReference(string caseName)
    {
        await Task.Yield();
        var (model, fixture, testCase) = Load(caseName);
        using var _ = model;
        var description = Ints(fixture["description"]!);
        var transcript = Ints(fixture["transcript"]!);
        var codes = Matrix(fixture["codes"]!);                                                            // [frames, Q]

        var labels = model.PaperCore!.DelayedLabels(codes);
        Assert.Equal(Matrix(testCase["labels"]!), labels);

        // The decoder reads the labels shifted right behind the start token (shift_tokens_right).
        int codebooks = labels.GetLength(0), length = labels.GetLength(1);
        var input = new int[codebooks, length];
        for (int q = 0; q < codebooks; q++)
        {
            input[q, 0] = model.PaperCore.Configuration.BeginToken;
            for (int j = 1; j < length; j++) input[q, j] = labels[q, j - 1];
        }
        var logits = model.EvaluateLogits(transcript, description, input);
        var expected = testCase["logits"]!.Select(v => (double)v).ToArray();
        int vocabulary = model.PaperCore.Configuration.AudioVocabulary, index = 0;
        Assert.Equal(expected.Length, codebooks * length * vocabulary);
        for (int q = 0; q < codebooks; q++)
            for (int t = 0; t < length; t++)
                for (int v = 0; v < vocabulary; v++)
                    Close(expected[index++], logits[q][t, v], $"logits[{q}, {t}, {v}]");

        var (loss, perCodebook) = model.EvaluateLosses(transcript, description, codes);
        var expectedPerCodebook = testCase["per_codebook_losses"]!.Select(v => (double)v).ToArray();
        for (int q = 0; q < codebooks; q++) Close(expectedPerCodebook[q], perCodebook[q], $"codebook {q} loss");
        // The reference accumulates the total into a float32 zero tensor (torch.zeros([])), so its total is the mean of
        // its per-codebook losses rounded to float32; the port keeps the mean in full precision.
        Close(expectedPerCodebook.Average(), loss, "loss");
        Assert.Equal((float)expectedPerCodebook.Average(), (float)(double)testCase["loss"]!);
    }

    [Theory(Timeout = 120000)]
    [InlineData("released")]
    [InlineData("variants")]
    public async Task GreedyGeneration_MatchesTheReference(string caseName)
    {
        await Task.Yield();
        var (model, fixture, testCase) = Load(caseName);
        using var _ = model;
        model.Voice = new AiDotNet.TextToSpeech.TtsVoice<double>
        {
            Description = new Tensor<double>(new[] { fixture["description"]!.Count() },
                new Vector<double>(fixture["description"]!.Select(v => (double)v).ToArray())),
        };
        var transcript = fixture["transcript"]!.Select(v => (double)v).ToArray();
        var codes = model.GenerateCodes(new Tensor<double>(new[] { transcript.Length }, new Vector<double>(transcript)));
        Assert.Equal(Matrix(testCase["generated"]!), codes);
    }
}
