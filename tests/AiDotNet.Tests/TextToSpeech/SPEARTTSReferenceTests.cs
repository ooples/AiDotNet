using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SPEAR-TTS's reading model against Hugging Face's <c>T5ForConditionalGeneration</c> with the encoder reading its own
/// table (<c>tools/reference-data/speartts_reference.py</c>): the label-smoothed loss of §7.2 as PyTorch's
/// <c>cross_entropy(label_smoothing=0.1)</c> computes it, one SGD step on the reading stage's parameters only (the
/// encoder's table and its lowest block), and the beam search of §7.4.
/// </summary>
public class SPEARTTSReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "speartts_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static (T5Seq2Seq<double> Model, JObject Fixture) Load()
    {
        var fixture = Reference();
        var c = fixture["config"]!;
        var configuration = new T5Configuration(
            VocabularySize: (int)c["vocab_size"]!, ModelDim: (int)c["d_model"]!, FeedForwardDim: (int)c["d_ff"]!,
            KeyValueDim: (int)c["d_kv"]!, Heads: (int)c["num_heads"]!, EncoderLayers: (int)c["num_layers"]!,
            DecoderLayers: (int)c["num_decoder_layers"]!, RelativeBuckets: (int)c["relative_attention_num_buckets"]!,
            RelativeMaxDistance: (int)c["relative_attention_max_distance"]!, Dropout: 0.0,
            DecoderStartTokenId: (int)c["decoder_start_token_id"]!, EndTokenId: (int)c["eos_token_id"]!,
            SeparateEncoderEmbedding: true);
        var model = new T5Seq2Seq<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration);

        string path = Path.Combine(Path.GetTempPath(), $"speartts_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadHuggingFaceWeights((name, shape) =>
            {
                var tensor = weights[name];
                Assert.Equal(shape, tensor.Shape.ToArray());
                used.Add(name);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
            // Every stored tensor is read, the encoder's own table among them.
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
            Assert.Contains("encoder.embed_tokens.weight", used);
        }
        finally
        {
            File.Delete(path);
        }
        return (model, fixture);
    }

    private static Tensor<double> Ids(IEnumerable<int> ids)
    {
        var values = ids.Select(i => (double)i).ToArray();
        return new Tensor<double>(new[] { values.Length }, new Vector<double>(values));
    }

    private static void AssertClose(double expected, double actual, string what) =>
        Assert.True(Math.Abs(expected - actual) <= 1e-9 * Math.Max(1, Math.Abs(expected)), $"{what}: reference {expected:R}, port {actual:R}.");

    private static int[] Ints(JToken token) => token.Select(v => (int)v).ToArray();

    [Fact(Timeout = 120000)]
    public async Task Loss_WithAndWithoutLabelSmoothing_MatchesPyTorch()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        var input = Ids(Ints(fixture["input_ids"]!));
        var labels = Ints(fixture["labels"]!);
        var random = new Random(0);
        AssertClose((double)fixture["loss"]!, model.Loss(input, labels, training: false, random)[0], "loss");
        AssertClose((double)fixture["smoothed_loss"]!,
            model.Loss(input, labels, training: false, random, labelSmoothing: (double)fixture["label_smoothing"]!)[0], "smoothed loss");
    }

    [Fact(Timeout = 120000)]
    public async Task OneSgdStep_OnTheEncoderTableAndLowestBlock_MatchesPyTorch()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        var input = Ids(Ints(fixture["input_ids"]!));
        var labels = Ints(fixture["labels"]!);
        double smoothing = (double)fixture["label_smoothing"]!, rate = (double)fixture["learning_rate"]!;
        var random = new Random(0);

        // Hugging Face's encoder.block[0].parameters(): the block, and the relative bias table T5 stores in it.
        var block = model.Encoder[0];
        var a = block.SelfAttention;
        var trainable = new List<ILayer<double>>
        {
            model.EncoderEmbedding ?? throw new InvalidOperationException("The encoder reads its own table."),
            block.SelfNorm, a.Query, a.Key, a.Value, a.Output, block.FeedForwardNorm, block.Wi, block.Wo, model.EncoderBias,
        };
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(trainable);
        Tensor<double> loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            loss = model.Loss(input, labels, training: false, random, labelSmoothing: smoothing);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        AssertClose((double)fixture["smoothed_loss"]!, loss[0], "smoothed loss");
        foreach (var p in parameters)
        {
            Assert.True(grads.TryGetValue(p, out var g), "Every reading parameter receives a gradient.");
            for (int i = 0; i < p.Length; i++) p[i] -= rate * g![i];
            AiDotNetEngine.Current.InvalidatePersistentTensor(p);
        }
        AssertClose((double)fixture["smoothed_loss_after_sgd"]!,
            model.Loss(input, labels, training: false, random, labelSmoothing: smoothing)[0], "smoothed loss after the step");
    }

    [Fact(Timeout = 120000)]
    public async Task BeamSearch_FindsTheReferenceSequence()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        int offset = (int)fixture["semantic_offset"]!, end = (int)fixture["config"]!["eos_token_id"]!;
        var memory = model.Encode(Ids(Ints(fixture["input_ids"]!)), training: false, new Random(0));
        Func<int, int, bool> allowed = (token, step) => token >= offset || (token == end && step > 0);
        int maxNew = (int)fixture["max_new_tokens"]!;
        Assert.Equal(Ints(fixture["beam"]!), model.BeamSearch(memory, null, Array.Empty<int>(), (int)fixture["beam_size"]!, maxNew, allowed));
        // A beam of one is greedy search; the fixture's seed makes it differ from the beam's sequence, so the beam
        // search above is not passing as a greedy one.
        var greedy = Ints(fixture["greedy"]!);
        Assert.NotEqual(Ints(fixture["beam"]!), greedy);
        Assert.Equal(greedy, model.BeamSearch(memory, null, Array.Empty<int>(), 1, maxNew, allowed));
    }
}
