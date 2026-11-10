using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Scalar reference for HF Llama causal GQA: dropout on softmax probabilities, then V.
/// It reads the sampled mask because Torch and the native engine have different random generators.</summary>
public class LlamaAttentionDropoutReferenceTests
{
    private static readonly double[][] Input =
    [
        [0.5, -0.2, 0.8, 0.1], [0.1, 0.7, -0.3, 0.5], [0.6, 0.2, 0.4, -0.5],
    ];

    private static LlamaNetwork<double> Create(double dropout)
    {
        var model = new LlamaNetwork<double>(new CpuEngine(), new List<LayerBase<double>>(), new LlamaConfiguration
        {
            VocabSize = 0, IncludeHead = false, IncludeFinalNorm = false,
            Hidden = 4, Heads = 2, KeyValueHeads = 1, HeadDim = 2, Layers = 1, Intermediate = 8,
            RmsEpsilon = 1e-5, AttentionDropout = dropout,
        });
        // Zero positions make RoPE the identity; a zero MLP isolates native attention and its residual.
        model.LoadTorchWeights((name, shape) => name switch
        {
            "model.layers.0.input_layernorm.weight" or "model.layers.0.post_attention_layernorm.weight" => [1, 1, 1, 1],
            "model.layers.0.self_attn.q_proj.weight" or "model.layers.0.self_attn.o_proj.weight" =>
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            "model.layers.0.self_attn.k_proj.weight" => [1, 0, 0, 0, 0, 1, 0, 0],
            "model.layers.0.self_attn.v_proj.weight" => [0, 0, 1, 0, 0, 0, 0, 1],
            "model.layers.0.mlp.gate_proj.weight" or "model.layers.0.mlp.up_proj.weight" or "model.layers.0.mlp.down_proj.weight"
                => new double[shape.Aggregate(1, (a, b) => a * b)],
            _ => throw new InvalidOperationException($"Unexpected checkpoint tensor {name}."),
        });
        return model;
    }

    [Theory]
    [InlineData(0.0, true)]
    [InlineData(0.5, false)]
    [InlineData(0.5, true)]
    public void NativeAttentionMatchesScalarGqaWithProbabilityDropout(double rate, bool training)
    {
        var model = Create(rate);
        model.Training = training;
        var dropout = model.Layers[0].AttentionDropout;
        if (dropout is not null) dropout.RandomSeed = 123;
        var input = new Tensor<double>(new[] { 1, 3, 4 }, new Vector<double>(Input.SelectMany(row => row).ToArray()));
        var actual = model.Forward(input, positionIds: new int[1, 3]).Hidden;
        Tensor<double>? mask = null;
        if (training && rate > 0)
        {
            mask = Assert.IsType<Tensor<double>>(typeof(DropoutLayer<double>)
                .GetField("_dropoutMask", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(dropout));
            Assert.Equal(new[] { 2, 3, 3 }, mask.Shape.ToArray());
            Assert.All(mask, value => Assert.True(value == 0 || value == 2));
            var eligible = Enumerable.Range(0, 2).SelectMany(head => Enumerable.Range(0, 3)
                .SelectMany(query => Enumerable.Range(0, query + 1).Select(key => mask[head, query, key]))).ToArray();
            Assert.Contains(0.0, eligible);
            Assert.Contains(2.0, eligible);
        }
        var normalized = Input.Select(row => row.Select(value => value / Math.Sqrt(row.Sum(x => x * x) / 4 + 1e-5)).ToArray()).ToArray();
        var expected = Input.SelectMany(row => row).ToArray();
        for (int head = 0; head < 2; head++)
            for (int query = 0; query < 3; query++)
            {
                var scores = Enumerable.Range(0, query + 1).Select(key =>
                    (normalized[query][2 * head] * normalized[key][0]
                     + normalized[query][2 * head + 1] * normalized[key][1]) / Math.Sqrt(2)).ToArray();
                double maximum = scores.Max();
                var weights = scores.Select(score => Math.Exp(score - maximum)).ToArray();
                double denominator = weights.Sum();
                for (int key = 0; key <= query; key++)
                    for (int feature = 0; feature < 2; feature++)
                        expected[query * 4 + head * 2 + feature] += weights[key] / denominator
                            * (mask is null ? 1 : mask[head, query, key]) * normalized[key][feature + 2];
            }
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-12, $"Attention output {i}: {expected[i]:R} vs {actual[i]:R}.");
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidAttentionDropoutIsRejectedBeforeAllocatingWeights(double rate)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => Create(rate));
        Assert.Contains("AttentionDropout", error.Message);
    }
}
