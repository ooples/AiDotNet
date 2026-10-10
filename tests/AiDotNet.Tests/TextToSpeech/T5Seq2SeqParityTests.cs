using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The T5 encoder–decoder behind Pheme's text-to-semantic stage reproduces Hugging Face's
/// <c>T5ForConditionalGeneration</c> (v1.0: ReLU feed-forward, tied embeddings).
/// </summary>
/// <remarks><c>ReferenceData/t5_reference.json</c> holds a tiny seeded T5's float32 weights and its float64 encoder
/// states, logits and loss (<c>tools/reference-data/t5_reference.py</c>). Eight relative buckets with a maximum distance
/// of 16 put the 12-token input's larger offsets in the logarithmic buckets.</remarks>
public class T5Seq2SeqParityTests
{
    private static JObject Reference()
    {
        const string fileName = "t5_reference.json";
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
            RelativeMaxDistance: (int)c["relative_attention_max_distance"]!, Dropout: 0.0);
        var model = new T5Seq2Seq<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration);

        string path = Path.Combine(Path.GetTempPath(), $"t5_reference_{Guid.NewGuid():N}.safetensors");
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
            Assert.Equal(weights.Keys.OrderBy(k => k, StringComparer.Ordinal), used.OrderBy(k => k, StringComparer.Ordinal));
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

    private static void AssertClose(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task EncoderStates_MatchHuggingFace()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        var inputIds = fixture["input_ids"]!.Select(v => (int)v).ToArray();
        var encoder = model.Encode(Ids(inputIds), training: false, new Random(0));
        AssertClose(fixture["encoder"]!.Select(v => (double)v).ToArray(), encoder, "encoder");
    }

    [Fact(Timeout = 120000)]
    public async Task DecoderLogitsAndLoss_MatchHuggingFace()
    {
        await Task.Yield();
        var (model, fixture) = Load();
        var inputIds = fixture["input_ids"]!.Select(v => (int)v).ToArray();
        var labels = fixture["labels"]!.Select(v => (int)v).ToArray();
        var random = new Random(0);
        var memory = model.Encode(Ids(inputIds), training: false, random);
        var shifted = new[] { 0 }.Concat(labels.Take(labels.Length - 1));      // decoder_start_token_id 0, then the labels
        var logits = model.Decode(Ids(shifted), memory, training: false, random);
        AssertClose(fixture["logits"]!.Select(v => (double)v).ToArray(), logits, "logits");

        var loss = model.Loss(Ids(inputIds), labels, training: false, random);
        double expected = (double)fixture["loss"]!;
        Assert.True(Math.Abs(expected - loss[0]) <= 1e-9 * expected, $"loss: reference {expected:R}, port {loss[0]:R}.");
    }

    [Theory(Timeout = 120000)]
    [InlineData(0, true, 0)]
    [InlineData(3, true, 19)]
    [InlineData(-3, true, 3)]
    [InlineData(40, true, 28)]
    [InlineData(-17, true, 10)]
    [InlineData(-5, false, 5)]
    [InlineData(5, false, 0)]
    [InlineData(-200, false, 31)]
    [InlineData(-40, false, 23)]
    public async Task RelativePositionBuckets_MatchTheReference(int relative, bool bidirectional, int expected)
    {
        await Task.Yield();
        // Hugging Face T5Attention._relative_position_bucket with 32 buckets and a maximum distance of 128.
        Assert.Equal(expected, T5RelativePosition.Bucket(relative, bidirectional, 32, 128));
    }
}
