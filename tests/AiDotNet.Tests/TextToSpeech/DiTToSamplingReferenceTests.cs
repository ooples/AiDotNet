using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class DiTToSamplingReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void CompleteDdimWithBothNeuralGuidanceBranches_MatchesIndependentTorchRollout(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "ditto_diffusion_reference.json")));
        var root = document.RootElement;
        var example = root.GetProperty("sampling_cases")[index];
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        int promptLength = example.GetProperty("prompt_length").GetInt32();
        var options = DiTToTTSPaperTests.TinyOptions();
        options.TextHiddenDim = 6;
        options.LatentDim = 4;
        options.DiTLayers = 2;
        options.InferenceSteps = example.GetProperty("steps").GetInt32();
        options.GuidanceScale = example.GetProperty("guidance").GetDouble();
        options.NoiseScheduleScale = example.GetProperty("scale").GetDouble();
        using var inference = new NoGradScope<double>();
        using var model = new DiTToTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        model.SetTrainingMode(false);
        int imported = 0;
        model.DiffusionTransformer!.Load((name, shape) =>
        {
            var values = Values(root.GetProperty("weights").GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            imported++;
            return values;
        }, "");
        Assert.Equal(63, imported);
        // Control only the length posterior; every diffusion layer remains real.
        var biases = Enumerable.Repeat(-40.0, options.MaxLatentLength + 1).ToArray();
        biases[4 - promptLength] = 40;
        TorchParameters.Linear(AiDotNetEngine.Current, model.LengthPredictor!.Classifier, options.LengthHiddenDim,
            options.MaxLatentLength + 1, new double[(options.MaxLatentLength + 1) * options.LengthHiddenDim], biases);
        var promptValues = Values(example.GetProperty("prompt"));
        var prompt = promptLength == 0 ? null : new Tensor<double>(new[] { promptLength, 4 }, new Vector<double>(promptValues));
        var text = new Tensor<double>(new[] { 3, 6 }, new Vector<double>(Values(root.GetProperty("text"))));
        var random = new ReplayGaussian(Values(example.GetProperty("initial")));
        var actual = model.GenerateLatentsFromEncoding(text, prompt, random);
        Assert.Equal(32, random.Consumed);
        Assert.Equal(new[] { 4, 4 }, actual.Shape.ToArray());
        var expected = Values(example.GetProperty("final"));
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"Complete DiTTo sampling case {index}: error {error:R}.");
        for (int i = 0; i < promptValues.Length; i++) Assert.Equal(promptValues[i], actual[i]);
    }

    // Invert the production Box-Muller transform. Only the initial Gaussian
    // samples are replayed; no diffusion prediction or sampler step is replaced.
    private sealed class ReplayGaussian : Random
    {
        private readonly double[] _values;
        internal int Consumed { get; private set; }
        internal ReplayGaussian(double[] values) => _values = values;
        public override double NextDouble()
        {
            Assert.True(Consumed < 2 * _values.Length, "Unexpected extra sampler randomness.");
            double value = _values[Consumed / 2];
            return Consumed++ % 2 == 0 ? 1 - Math.Exp(-value * value / 2) : value < 0 ? .5 : 0;
        }
    }
}
