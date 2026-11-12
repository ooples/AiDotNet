using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.StyleEmotion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class KokoroHarmonicSourceTests
{
    [Fact]
    public void OriginalNonzeroExcitation_MatchesSourceAndEveryMergeGradient_WithDetachedPitch()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "kokoro_harmonic_source.json")));
        var root = document.RootElement;
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var layers = new List<LayerBase<double>>();
        var engine = AiDotNetEngine.Current;
        var source = new StyleTts2Source<double>(engine, layers, 24000, 300);
        source.Load(AiDotNet.ModelLoading.TorchTensorSource.From((string name, int[] shape) => name.EndsWith("weight", StringComparison.Ordinal)
            ? Values(root.GetProperty("weight")) : Values(root.GetProperty("bias"))), "source");
        var draws = root.GetProperty("draws").EnumerateArray().ToArray();
        source.InitialPhases = Values(draws.Single(x => x.GetProperty("kind").GetString() == "uniform").GetProperty("values"));
        source.GaussianNoise = Values(draws.Single(x => x.GetProperty("kind").GetString() == "normal" &&
            x.GetProperty("shape").EnumerateArray().Last().GetInt32() == 9).GetProperty("values"));
        var pitchValues = Values(root.GetProperty("pitch"));
        var pitch = new Tensor<double>(new[] { 1, pitchValues.Length }, new Vector<double>(pitchValues));
        var objective = new Tensor<double>(new[] { 1, 1, 300 * pitchValues.Length },
            new Vector<double>(Values(root.GetProperty("objective_weights"))));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(2, parameters.Count);
        using var tape = new GradientTape<double>();
        var harmonic = source.Forward(pitch, new Random(7));
        Close(harmonic, Values(root.GetProperty("harmonic")), "harmonic source", 1e-12);
        var loss = engine.ReduceSum(engine.TensorMultiply(harmonic, objective), new[] { 0, 1, 2 }, true);
        Close(loss, new[] { root.GetProperty("loss").GetDouble() }, "source objective", 1e-10);
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { pitch }).ToList());
        Close(gradients[source.Merge.GetWeights()], Values(root.GetProperty("weight_gradient")), "merge weight gradient", 1e-9);
        Close(gradients[source.Merge.GetBiases()], Values(root.GetProperty("bias_gradient")), "merge bias gradient", 1e-9);
        if (gradients.TryGetValue(pitch, out var pitchGradient))
            Assert.All(pitchGradient.ToArray(), value => Assert.Equal(0, value));
        // Explicit draws make the result independent of the model's normal sampling stream.
        Close(source.Forward(pitch, new Random(931)), harmonic.ToArray(), "replayed excitation", 1e-12);
    }

    private static void Close(Tensor<double> actual, double[] expected, string name, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < tolerance, $"Kokoro {name} maximum error {error:R}.");
    }
}
