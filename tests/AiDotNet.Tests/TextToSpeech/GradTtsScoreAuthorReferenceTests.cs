using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Full-network numerics exported from the unmodified, pinned Huawei Grad-TTS implementation.</summary>
public class GradTtsScoreAuthorReferenceTests
{
    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static T Field<T>(object owner, string name) => (T)owner.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;

    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "grad_tts_score_reference.json")));

    private static (GradTtsScoreEstimator<double> Network, Tensor<double> X, Tensor<double> Mu, Tensor<double> Mask) Create(JsonElement reference)
    {
        int frames = reference.GetProperty("frames").GetInt32(), bins = reference.GetProperty("mel_bins").GetInt32();
        var x = new Tensor<double>(new[] { frames, bins }, new Vector<double>(Values(reference.GetProperty("xt"))));
        var mu = new Tensor<double>(new[] { frames, bins }, new Vector<double>(Values(reference.GetProperty("mu"))));
        var mask = new Tensor<double>(new[] { 1, 1, 1, frames });
        for (int i = 0; i < reference.GetProperty("valid_frames").GetInt32(); i++) mask[i] = 1;
        var network = new GradTtsScoreEstimator<double>(AiDotNetEngine.Current, 8, new[] { 1, 2, 4 }, 1000);
        network.Estimate(x, mu, .37, mask); // Materialize the real lazy time projections before loading weights.
        foreach (var layer in network.Layers) Initialize(layer);
        return (network, x, mu, mask);
    }

    private static void Initialize(LayerBase<double> layer)
    {
        var parameters = layer.GetParameters();
        int offset = 0;
        void Bias(int count)
        {
            for (int i = 0; i < count; i++) parameters[offset++] = .01 * Math.Cos((i + 1) * .17);
        }
        if (layer is DenseLayer<double>)
        {
            var weight = Field<Tensor<double>>(layer, "_weights");
            int input = weight.Shape[0], output = weight.Shape[1];
            for (int i = 0; i < input; i++)
                for (int o = 0; o < output; o++)
                    parameters[offset++] = .04 * Math.Sin((i + 1) * (o + 1) * .13) / Math.Sqrt(input);
            Bias(output);
        }
        else if (layer is TorchConv2DLayer<double>)
        {
            var weight = Field<Tensor<double>>(layer, "_weight");
            int a = weight.Shape[0], b = weight.Shape[1], height = weight.Shape[2], width = weight.Shape[3];
            for (int ia = 0; ia < a; ia++) for (int ib = 0; ib < b; ib++)
                for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
                    parameters[offset++] = .03 * Math.Sin((ia + 1) * (ib + 1) * .13 + y * .17 + x * .19) / Math.Sqrt(b * height * width);
            Bias(Field<Tensor<double>>(layer, "_bias").Length);
        }
        else if (layer is GroupNormalizationLayer<double>)
        {
            int count = parameters.Length / 2;
            for (int i = 0; i < count; i++) parameters[offset++] = 1 + .03 * Math.Sin((i + 1) * .13);
            Bias(count);
        }
        else if (layer is RezeroGateLayer<double>)
            parameters[offset++] = .15;
        else throw new InvalidOperationException($"Unrecognized author-network layer: {layer.GetType().Name}.");
        Assert.Equal(parameters.Length, offset);
        layer.SetParameters(parameters);
    }

    private static void Close(Tensor<double> actual, double[] expected, string name, List<string>? failures = null)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.All(actual.ToVector().ToArray(), x => Assert.True(!double.IsNaN(x) && !double.IsInfinity(x), $"{name} is nonfinite."));
        double tolerance = 1e-12 + 2e-8 * expected.Select(Math.Abs).DefaultIfEmpty(0).Max();
        int worst = Enumerable.Range(0, expected.Length).OrderByDescending(i => Math.Abs(actual[i] - expected[i])).First();
        double error = Math.Abs(actual[worst] - expected[worst]);
        string message = $"{name}[{worst}]: {actual[worst]:R} != {expected[worst]:R}, error {error:R}, tolerance {tolerance:R}.";
        if (error <= tolerance) return;
        if (failures is null) Assert.Fail(message);
        else failures.Add(message);
    }

    [Fact]
    public void NativeTrainableCount_MatchesTheAuthorNetwork()
    {
        using var fixture = Reference();
        var (network, _, _, _) = Create(fixture.RootElement);
        Assert.Equal(fixture.RootElement.GetProperty("parameter_count").GetInt32(), network.Layers.Sum(layer => layer.ParameterCount));
    }

    [Fact]
    public void CompleteScoreForwardLossAndGradients_MatchAuthorWithMaskedFramesAndActiveAttention()
    {
        using var fixture = Reference(); var reference = fixture.RootElement;
        var failures = new List<string>();
        var (network, x, mu, mask) = Create(reference);
        var convs = network.Layers.OfType<TorchConv2DLayer<double>>().ToArray();
        var time = network.Layers.OfType<DenseLayer<double>>().First();
        var norm = network.Layers.OfType<GroupNormalizationLayer<double>>().First();
        var gates = network.Layers.OfType<RezeroGateLayer<double>>().Select(g => Field<Tensor<double>>(g, "_gate")).ToArray();
        var selected = new Dictionary<string, Tensor<double>>
        {
            ["xt_gradient"] = x, ["mu_gradient"] = mu,
            ["first_conv_weight_gradient"] = Field<Tensor<double>>(convs.First(), "_weight"),
            ["final_conv_weight_gradient"] = Field<Tensor<double>>(convs.Last(), "_weight"),
            ["final_conv_bias_gradient"] = Field<Tensor<double>>(convs.Last(), "_bias"),
            ["time_mlp_weight_gradient"] = Field<Tensor<double>>(time, "_weights"),
            ["time_mlp_bias_gradient"] = Field<Tensor<double>>(time, "_biases"),
            ["first_norm_weight_gradient"] = Field<Tensor<double>>(norm, "_gamma"),
            ["first_norm_bias_gradient"] = Field<Tensor<double>>(norm, "_beta"),
        };
        using var tape = new GradientTape<double>();
        var output = network.Estimate(x, mu, .37, mask);
        Close(output, Values(reference.GetProperty("output")), "complete score output", failures);
        var engine = AiDotNetEngine.Current;
        var weights = new Tensor<double>(output.Shape.ToArray(), new Vector<double>(Values(reference.GetProperty("loss_weights"))));
        var loss = engine.ReduceSum(engine.TensorMultiply(weights, engine.TensorMultiply(output, output)), new[] { 0, 1 }, keepDims: false);
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "complete weighted loss", failures);
        var gradients = tape.ComputeGradients(loss, sources: selected.Values.Concat(gates).ToList());
        foreach (var (name, parameter) in selected)
            Close(gradients[parameter], Values(reference.GetProperty(name)), name, failures);
        var gateGradients = Values(reference.GetProperty("gate_gradients"));
        Assert.Equal(gateGradients.Length, gates.Length);
        for (int i = 0; i < gates.Length; i++)
            Close(gradients[gates[i]], new[] { gateGradients[i] }, $"attention gate {i}", failures);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
