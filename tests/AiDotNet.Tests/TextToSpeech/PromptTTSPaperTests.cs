using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.DescriptionBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// PromptTTS (Guo et al. 2023): its style encoder reproduces THUDM/P-tuning-v2's BERT with deep prefixes (the [CLS]
/// state and the classifier's logits and loss, with and without the prefix MLP), and the style representation reaches
/// every content-encoder and decoder block as a prepended position.
/// </summary>
/// <remarks><c>ReferenceData/prompttts_style_reference.json</c> holds tiny seeded models' float32 weights and their
/// float64 results (<c>tools/reference-data/prompttts_style_reference.py</c>).</remarks>
public class PromptTTSPaperTests
{
    private static JObject Reference()
    {
        const string fileName = "prompttts_style_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    /// <summary>A small PromptTTS: 16-wide content networks, one block each, and the given style encoder.</summary>
    private static PromptTTSOptions Small(Action<PromptTTSOptions>? style = null)
    {
        var options = new PromptTTSOptions
        {
            HiddenDim = 16, EncoderDim = 16, NumEncoderLayers = 1, NumDecoderLayers = 1, NumHeads = 2, FftFilterSize = 32,
            VariancePredictorFilterSize = 16, VocabSize = 32, MelChannels = 8, DropoutRate = 0.0, VariancePredictorDropout = 0.0,
            StyleVocabularySize = 50, StyleHiddenDim = 16, StyleLayers = 2, StyleHeads = 2, StyleIntermediateDim = 32,
            StyleMaxPositions = 32, StyleDropout = 0.0, StyleFactorClasses = new[] { 2, 3 },
        };
        style?.Invoke(options);
        return options;
    }

    [Theory(Timeout = 120000)]
    [InlineData("prefix")]
    [InlineData("prefix_projection")]
    public async Task StyleEncoder_MatchesPTuningV2(string caseName)
    {
        await Task.Yield();
        var fixture = Reference();
        var testCase = fixture["cases"]![caseName]!;
        var bert = fixture["bert"]!;
        using var model = new PromptTTS<double>(Architecture(), Small(o =>
        {
            o.StyleVocabularySize = (int)bert["vocab_size"]!;
            o.StyleHiddenDim = (int)bert["hidden_size"]!;
            o.StyleLayers = (int)bert["num_hidden_layers"]!;
            o.StyleHeads = (int)bert["num_attention_heads"]!;
            o.StyleIntermediateDim = (int)bert["intermediate_size"]!;
            o.StyleMaxPositions = (int)bert["max_position_embeddings"]!;
            o.PrefixLength = (int)fixture["pre_seq_len"]!;
            o.PrefixProjection = (bool)testCase["prefix_projection"]!;
            o.PrefixHiddenSize = (int)fixture["prefix_hidden_size"]!;
            o.StyleFactorClasses = new[] { (int)fixture["num_labels"]! };
        }));
        string path = Path.Combine(Path.GetTempPath(), $"prompttts_style_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)testCase["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadPTuningWeights((name, shape) =>
            {
                Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
                Assert.Equal(shape, weights[name].Shape.ToArray());
                used.Add(name);
                return weights[name].ToVector().Select(v => (double)v).ToArray();
            });
            // BERT's registered position-id buffer is not a weight.
            var unused = weights.Keys.Where(k => !used.Contains(k)).ToList();
            Assert.True(unused.Count == 0, "Unread reference tensors: " + string.Join(", ", unused));
        }
        finally
        {
            File.Delete(path);
        }

        var prompt = fixture["prompt"]!.Select(v => (int)v).ToArray();
        var cls = model.EvaluateClsState(prompt);
        var expectedCls = testCase["cls"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expectedCls.Length, cls.Length);
        for (int i = 0; i < expectedCls.Length; i++) Close(expectedCls[i], cls[i], $"[CLS][{i}]");

        var logits = model.EvaluateFactorLogits(prompt)[0];
        var expectedLogits = testCase["logits"]!.Select(v => (double)v).ToArray();
        for (int i = 0; i < expectedLogits.Length; i++) Close(expectedLogits[i], logits[0, i], $"logits[{i}]");

        var promptTensor = new Tensor<double>(new[] { prompt.Length }, new Vector<double>(prompt.Select(v => (double)v).ToArray()));
        Close((double)testCase["loss"]!, model.EvaluateStyleLoss(promptTensor, new[] { (int)fixture["label"]! }), "loss");
    }

    private static void Close(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-8 * Math.Max(1, Math.Abs(expected)), $"{what}: reference {expected:R}, port {actual:R}.");

    [Fact(Timeout = 120000)]
    public async Task PrependedStyle_IsAnExtraPositionDroppedFromTheOutput()
    {
        await Task.Yield();
        // §2.3: the style representation is prepended to the input of every Transformer block.
        var block = new FeedForwardTransformerBlock<double>(8, 2, 16, 3, 1, 0.0, prependCondition: true);
        var plain = new FeedForwardTransformerBlock<double>(8, 2, 16, 3, 1, 0.0);
        var random = new Random(3);
        var x = new Tensor<double>(new[] { 5, 8 });
        var style = new Tensor<double>(new[] { 1, 8 });
        for (int i = 0; i < x.Length; i++) x[i] = random.NextDouble() - 0.5;
        for (int i = 0; i < style.Length; i++) style[i] = random.NextDouble() - 0.5;
        // Materialize both, then give the plain block the prepending block's weights.
        block.Forward(x, style);
        plain.Forward(x);
        plain.SetParameters(block.GetParameters());

        var joined = new Tensor<double>(new[] { 6, 8 });
        for (int c = 0; c < 8; c++) joined[0, c] = style[0, c];
        for (int t = 0; t < 5; t++) for (int c = 0; c < 8; c++) joined[t + 1, c] = x[t, c];
        var expected = plain.Forward(joined);
        var actual = block.Forward(x, style);
        Assert.Equal(new[] { 5, 8 }, actual.Shape.ToArray());
        for (int t = 0; t < 5; t++)
            for (int c = 0; c < 8; c++) Assert.Equal(expected[t + 1, c], actual[t, c], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task StylePrompt_ChangesTheMelSpectrogram()
    {
        await Task.Yield();
        using var model = new PromptTTS<double>(Architecture(), Small());
        var tokens = new Tensor<double>(new[] { 6 }, new Vector<double>(new double[] { 3, 9, 4, 12, 7, 5 }));
        model.Voice = new TtsVoice<double> { Description = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 2, 11, 30, 3 })) };
        var first = model.Predict(tokens);
        model.Voice = new TtsVoice<double> { Description = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 2, 41, 7, 3 })) };
        var second = model.Predict(tokens);
        Assert.Equal(first.Shape.ToArray()[1], second.Shape.ToArray()[1]);
        bool differs = first.Shape.ToArray().SequenceEqual(second.Shape.ToArray()) == false
            || Enumerable.Range(0, first.Length).Any(i => Math.Abs(first[i] - second[i]) > 1e-9);
        Assert.True(differs, "Two style prompts gave the same mel spectrogram.");
    }

    [Fact(Timeout = 120000)]
    public async Task StyleStage_TrainsOnlyThePrefixesAndHeads_UnderPTuningV2()
    {
        await Task.Yield();
        using var model = new PromptTTS<double>(Architecture(), Small(o => o.StyleLearningRate = 1e-2));
        var prompt = new Tensor<double>(new[] { 5 }, new Vector<double>(new double[] { 2, 14, 21, 8, 3 }));
        var labels = new[] { 1, 2 };
        var bertBefore = model.StyleEncoderLayers.Select(l => l.GetParameters().ToArray()).ToList();
        var prefixBefore = model.PrefixLayers.Select(l => l.GetParameters().ToArray()).ToList();
        double before = model.EvaluateStyleLoss(prompt, labels);
        for (int step = 0; step < 15; step++) model.TrainStyle(prompt, labels);
        double after = model.EvaluateStyleLoss(prompt, labels);

        Assert.True(after < before, $"The style loss did not fall: {before} -> {after}.");
        for (int i = 0; i < bertBefore.Count; i++)
            Assert.Equal(bertBefore[i], model.StyleEncoderLayers[i].GetParameters().ToArray());
        Assert.Contains(Enumerable.Range(0, prefixBefore.Count),
            i => !prefixBefore[i].SequenceEqual(model.PrefixLayers[i].GetParameters().ToArray()));
        Assert.Equal(labels, model.PredictStyleFactors(prompt));
    }
}
