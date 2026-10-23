using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.StyleEmotion;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// StyleTTS (Li et al. 2022) reproduces yl4579/StyleTTS: the text encoder, style encoder, duration encoder and durations,
/// inference with predicted durations, pitch and energy through the decoder, the text aligner's attention and token
/// logits, the pitch extractor, the discriminator, and monotonic alignment search (resemble-ai/monotonic_align).
/// </summary>
/// <remarks><c>ReferenceData/styletts_reference.json</c> holds a tiny seeded model's float32 weights and its float64
/// results (<c>tools/reference-data/styletts_reference.py</c>).</remarks>
public class StyleTTSPaperTests
{
    private static JObject Reference()
    {
        const string fileName = "styletts_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    private static (StyleTTS<double> Model, JObject Fixture) Load()
    {
        var fixture = Reference();
        var m = fixture["model"]!;
        var asr = fixture["asr"]!;
        var model = new StyleTTS<double>(Architecture(), new StyleTTSOptions
        {
            HiddenDim = (int)m["hidden_dim"]!, VocabSize = (int)m["n_token"]!, StyleDim = (int)m["style_dim"]!,
            TextLayers = (int)m["n_layer"]!, StyleEncoderDim = (int)m["dim_in"]!, MaxConvDim = (int)m["max_conv_dim"]!,
            MelChannels = (int)m["n_mels"]!, DropoutRate = (double)m["dropout"]!,
            AlignerHidden = (int)asr["hidden_dim"]!, AlignerTokenEmbedding = (int)asr["token_embedding_dim"]!,
            AlignerConvLayers = (int)asr["n_layers"]!,
            PitchExtractorChannels = fixture["jdc_widths"]!.Select(v => (int)v).ToArray(), PitchExtractorLstm = (int)fixture["jdc_lstm"]!,
        });
        string path = Path.Combine(Path.GetTempPath(), $"styletts_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadReferenceWeights((name, shape) =>
            {
                Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
                var tensor = weights[name];
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            }, includeDiscriminator: true);
            // Batch-norm step counters are not weights.
            var unused = weights.Keys.Where(k => !used.Contains(k) && !k.EndsWith("num_batches_tracked", StringComparison.Ordinal)).ToList();
            Assert.True(unused.Count == 0, "Unread reference tensors: " + string.Join(", ", unused));
        }
        finally
        {
            File.Delete(path);
        }
        // The reference runs in evaluation mode: stored spectral-norm vectors, batch-norm running statistics.
        model.SetTrainingMode(false);
        return (model, fixture);
    }

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static Tensor<double> Matrix(double[] values, int rows, int columns) =>
        new(new[] { rows, columns }, new Vector<double>(values));

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 300000)]
    public async Task Inference_MatchesTheReference()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        using var _ = model;
        var random = new Random(0);
        var ids = fixture["tokens"]!.Select(v => (int)v).ToArray();
        var tokens = new Tensor<double>(new[] { ids.Length }, new Vector<double>(ids.Select(v => (double)v).ToArray()));
        var reference = Matrix(Values(fixture["reference"]!), 80, (int)fixture["reference_frames"]!);

        var text = model.TextEncoderNetwork.Forward(tokens, false, random);
        Close(Values(fixture["text"]!), text, "text features");
        var style = model.StyleEncoderNetwork.Forward(reference);
        Close(Values(fixture["style"]!), style, "style");
        var encoded = model.PredictorNetwork.EncodeText(text, style, false, random);
        Close(Values(fixture["duration_encoder"]!), encoded, "duration encoder");
        Close(Values(fixture["durations"]!), model.PredictorNetwork.Durations(encoded, false, random), "durations");
        Close(Values(fixture["mel"]!), model.SynthesizeMel(ids, style), "synthesized mel");
    }

    [Fact(Timeout = 300000)]
    public async Task TrainingModules_MatchTheReference()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        using var _ = model;
        var random = new Random(0);
        var ids = fixture["tokens"]!.Select(v => (int)v).ToArray();
        int frames = (int)fixture["frames"]!;
        var mel = Matrix(Values(fixture["input_mel"]!), 80, frames);

        var reference = Matrix(Values(fixture["reference"]!), 80, (int)fixture["reference_frames"]!);
        var (logit, features) = model.DiscriminatorNetwork.Forward(new Tensor<double>(new[] { 1, 1, 80, reference.Shape[1] }, reference.ToVector()));
        var expectedFeatures = fixture["discriminator_features"]!.ToArray();
        Assert.Equal(expectedFeatures.Length, features.Count);
        for (int i = 0; i < features.Count; i++)
        {
            var expected = Values(expectedFeatures[i]);
            var flat = features[i].ToVector().ToArray();
            int head = expected.Length - 2;
            var summary = flat.Take(head).Concat(new[] { flat.Sum(), flat.Sum(v => v * v) }).ToArray();
            Close(expected, new Tensor<double>(new[] { summary.Length }, new Vector<double>(summary)), $"discriminator feature {i}");
        }
        Close(Values(fixture["discriminator_logit"]!), logit, "discriminator logit");

        var (logits, energies) = model.AlignerNetwork.Decode(mel, ids, false, random, maskRate: 0.0);
        Close(Values(fixture["aligner_logits"]!), logits, "aligner logits");
        Close(Values(fixture["aligner_energies"]!), energies, "aligner energies");
        Close(Values(fixture["aligner_ctc"]!), model.AlignerNetwork.CtcLogits(mel, false, random), "CTC logits");
        Close(Values(fixture["pitch"]!), model.PitchExtractorNetwork.Forward(mel, false, random), "pitch");
    }

    [Fact(Timeout = 120000)]
    public async Task MonotonicAlignmentSearch_MatchesTheReference()
    {
        await Task.Yield();
        var fixture = Reference();
        var rows = fixture["scores"]!.Select(r => r.Select(v => (double)v).ToArray()).ToArray();
        var scores = new double[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[0].Length; j++) scores[i, j] = (float)rows[i][j];   // the reference searches in float32
        var path = StyleTTS<double>.HardPath(scores);
        var expected = fixture["mas_path"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        for (int i = 0; i < expected.Length; i++)
            for (int j = 0; j < expected[0].Length; j++) Assert.Equal(expected[i][j], (int)path[i, j]);
    }

    private static StyleTTS<double> Small(StyleTtsTrainingStage stage, int tmaStart) => new(Architecture(), new StyleTTSOptions
    {
        HiddenDim = 16, StyleDim = 8, TextLayers = 2, StyleEncoderDim = 4, MaxConvDim = 16, ResidualDim = 8, AlignerHidden = 16,
        AlignerTokenEmbedding = 8, AlignerConvLayers = 1, PitchExtractorChannels = new[] { 4, 8, 8, 8 }, PitchExtractorLstm = 8,
        TmaStartStep = tmaStart,
    }) { CurrentStage = stage };

    private static TtsTrainingSample<double> Utterance()
    {
        var mel = new Tensor<double>(new[] { 96, 80 });
        for (int f = 0; f < 96; f++)
            for (int c = 0; c < 80; c++) mel[f, c] = Math.Sin(0.17 * f + 0.05 * c) - 1.5;
        var ids = Enumerable.Range(0, 24).Select(i => (double)((i * 7) % 178)).ToArray();
        return new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 24 }, new Vector<double>(ids)), Mel = mel };
    }

    // Batch norms' running statistics (in their parameter vectors) update in every training-mode forward, as the
    // reference's pitch extractor's do; they are statistics, not trained weights.
    // Likewise a spectral norm's singular-vector estimates u and v (after its kernel, gain and bias) take a power
    // iteration in every training-mode forward.
    private static double[] Trainable(AiDotNet.NeuralNetworks.Layers.LayerBase<double> layer)
    {
        if (layer is AiDotNet.NeuralNetworks.Layers.BatchNormalizationLayer<double>) return Array.Empty<double>();
        var all = layer.GetParameters().ToArray();
        if (layer is AiDotNet.NeuralNetworks.Layers.NormedConv2DLayer<double> conv)
        {
            var shape = conv.TorchWeightShape;
            int estimates = shape[0] + shape[1] * shape[2] * shape[3];
            return all.Take(all.Length - estimates).ToArray();
        }
        return all;
    }

    private static List<double[]> Snapshot(IEnumerable<AiDotNet.NeuralNetworks.Layers.LayerBase<double>> layers) =>
        layers.Select(Trainable).ToList();

    private static bool Changed(List<double[]> before, IReadOnlyList<AiDotNet.NeuralNetworks.Layers.LayerBase<double>> layers) =>
        Enumerable.Range(0, before.Count).Any(i => !before[i].SequenceEqual(Trainable(layers[i])));

    [Fact(Timeout = 300000)]
    public async Task ReconstructionStage_TrainsTheGeneratorAndDiscriminator_AndHoldsTheAlignerUntilTmaStarts()
    {
        await Task.Yield();
        // §II-B stage 1: text encoder, style encoder and decoder; the aligner and pitch extractor once the aligner's
        // objectives start; the predictors never.
        using var model = Small(StyleTtsTrainingStage.Reconstruction, tmaStart: 2);
        var sample = Utterance();
        var aligner = Snapshot(model.AlignerLayers);
        var predictor = Snapshot(model.PredictorLayers);
        var decoder = Snapshot(model.DecoderLayers);
        var discriminator = Snapshot(model.DiscriminatorLayers);
        model.Train(sample);
        Assert.False(Changed(aligner, model.AlignerLayers), "The aligner trained before its objectives started.");
        Assert.True(Changed(decoder, model.DecoderLayers), "The decoder did not train.");
        Assert.True(Changed(discriminator, model.DiscriminatorLayers), "The discriminator did not train.");
        model.Train(sample);
        model.Train(sample);
        Assert.True(Changed(aligner, model.AlignerLayers), "The aligner did not train once its objectives started.");
        Assert.False(Changed(predictor, model.PredictorLayers), "The predictors trained in the reconstruction stage.");
    }

    [Fact(Timeout = 300000)]
    public async Task ProsodyStage_TrainsOnlyThePredictors()
    {
        await Task.Yield();
        // §II-B stage 2 fixes every module except the duration and prosody predictors (the paper's objective has no
        // adversarial term, so the discriminator is fixed too).
        using var model = Small(StyleTtsTrainingStage.Prosody, tmaStart: 0);
        var sample = Utterance();
        var fixedLayers = model.TextEncoderLayers.Concat(model.StyleEncoderLayers).Concat(model.DecoderLayers).Concat(model.AlignerLayers)
            .Concat(model.PitchExtractorLayers).Concat(model.DiscriminatorLayers).ToList();
        var before = Snapshot(fixedLayers);
        var predictor = Snapshot(model.PredictorLayers);
        double loss = 0;
        for (int i = 0; i < 3; i++) loss = model.Train(sample);
        Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss) && loss > 0, $"Stage 2 loss {loss}.");
        Assert.True(Changed(predictor, model.PredictorLayers), "The predictors did not train.");
        var moved = Enumerable.Range(0, before.Count).Where(i => !before[i].SequenceEqual(Trainable(fixedLayers[i])))
            .Select(i => $"{i}:{fixedLayers[i].GetType().Name}").ToList();
        Assert.True(moved.Count == 0, "Modules outside the predictors trained in stage 2: " + string.Join(", ", moved));
    }
}
