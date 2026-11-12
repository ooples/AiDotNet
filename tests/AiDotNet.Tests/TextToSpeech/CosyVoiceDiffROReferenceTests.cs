using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class CosyVoiceDiffROReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void CompleteQwenAndToken2TextReward_MatchIndependentLossAndEveryGradient(int index)
    {
        using var document = Read();
        var root = document.RootElement;
        var example = root.GetProperty("cases")[index];
        var (policy, policyLayers) = Policy(root.GetProperty("policy_weights"));
        var (reference, referenceLayers) = Policy(root.GetProperty("reference_weights"));
        var reward = Reward(root.GetProperty("reward_weights"));
        var policyParameters = Parameters(policyLayers);
        var referenceParameters = Parameters(referenceLayers);
        Assert.Equal(16, policyParameters.Count);
        Assert.Equal(16, referenceParameters.Count);
        Assert.Equal(27, reward.Parameters.Count);
        var expected = MatchParameters(policyParameters, root.GetProperty("policy_weights"), example.GetProperty("policy_gradients"))
            .Concat(MatchParameters(referenceParameters, root.GetProperty("reference_weights"), example.GetProperty("reference_gradients")))
            .Concat(MatchParameters(reward.Parameters, root.GetProperty("reward_weights"), example.GetProperty("reward_gradients"))).ToList();
        var random = new UniformReplay(Values(example.GetProperty("uniforms")));
        using var tape = new GradientTape<double>();
        var policyLogits = policy.SpeechLogits(Ids(root.GetProperty("text")), Ids(root.GetProperty("speech")));
        // Deliberately record the reference too: the objective must stop its gradient.
        var referenceLogits = reference.SpeechLogits(Ids(root.GetProperty("text")), Ids(root.GetProperty("speech")));
        var result = CosyVoiceDiffRO<double>.Objective(AiDotNetEngine.Current, policyLogits, referenceLogits,
            soft => reward.LogReward(soft, Ids(root.GetProperty("transcript"))), new CosyVoiceDiffROOptions
            {
                KlCoefficient = example.GetProperty("beta").GetDouble(),
                Temperature = example.GetProperty("temperature").GetDouble()
            }, random);
        Assert.Equal(20, random.Consumed);
        Close(policyLogits, Values(example.GetProperty("policy_logits")), "teacher-forced acoustic logits");
        Close(referenceLogits, Values(example.GetProperty("reference_logits")), "reference acoustic logits");
        Close(result.SoftTokens, Values(example.GetProperty("soft_tokens")), "Gumbel soft tokens");
        Close(result.Kl, new[] { example.GetProperty("kl").GetDouble() }, "summed forward token KL");
        Close(result.Reward, new[] { example.GetProperty("reward").GetDouble() }, "complete ASR log reward");
        Close(result.Loss, new[] { example.GetProperty("loss").GetDouble() }, "DiffRO objective");
        var gradients = tape.ComputeGradients(result.Loss,
            sources: expected.Select(item => item.Parameter).Concat(new[] { result.SoftTokens }).ToList());
        foreach (var item in expected)
        {
            if (gradients.TryGetValue(item.Parameter, out var gradient)) Close(gradient, item.Gradient, item.Name);
            else Assert.All(item.Gradient, value => Assert.Equal(0, value));
        }
        Close(gradients[result.SoftTokens], Values(example.GetProperty("soft_gradient")), "reward-to-soft-token gradient");
    }

    [Fact]
    public void Token2TextPretraining_PerformsTheIndependentFullRewardSgdStep()
    {
        using var document = Read();
        var root = document.RootElement;
        var expected = root.GetProperty("reward_training");
        var reward = Reward(root.GetProperty("reward_weights"));
        double loss = reward.Train(Ids(root.GetProperty("speech")), Ids(root.GetProperty("transcript")),
            expected.GetProperty("learning_rate").GetDouble());
        Assert.True(Math.Abs(loss - expected.GetProperty("loss_before").GetDouble()) < 1e-9);
        MatchParameters(reward.Parameters, expected.GetProperty("weights_after"), null);
        var speech = Ids(root.GetProperty("speech"));
        var hard = new Tensor<double>(new[] { speech.Length, reward.SpeechVocabularySize });
        for (int i = 0; i < speech.Length; i++) hard[i, speech[i]] = 1;
        using var noGrad = new NoGradScope<double>();
        Assert.True(Math.Abs(-reward.LogReward(hard, Ids(root.GetProperty("transcript"))).GetFlat(0)
            - expected.GetProperty("loss_after").GetDouble()) < 1e-9);
    }

    private static (CosyVoice2Lm<double> Model, List<LayerBase<double>> Layers) Policy(JsonElement weights)
    {
        var layers = new List<LayerBase<double>>();
        var model = new CosyVoice2Lm<double>(AiDotNetEngine.Current, layers, new CosyVoice2LmConfiguration(new LlamaConfiguration
        {
            VocabSize = 32, Hidden = 8, Intermediate = 16, Layers = 1, Heads = 2, KeyValueHeads = 1,
            RmsEpsilon = 1e-6, RopeTheta = 1_000_000, AttentionBias = true, IncludeHead = false
        }, SpeechVocabulary: 5, SpeechSpecialTokens: 200, SpecialsInSpeechEmbedding: true, BiasFreeDecoder: true), new Random(7));
        var used = new HashSet<string>();
        model.LoadTorchWeights((name, shape) =>
        {
            used.Add(name);
            var values = Values(weights.GetProperty(name).GetProperty("values"));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        }, "");
        Assert.Equal(weights.EnumerateObject().Select(x => x.Name).OrderBy(x => x), used.OrderBy(x => x));
        foreach (var layer in layers) layer.SetTrainingMode(false);
        return (model, layers);
    }

    private static CosyVoiceToken2TextReward<double> Reward(JsonElement weights)
    {
        var model = new CosyVoiceToken2TextReward<double>(5, 11, 8, 16, 2, 1, 1);
        var used = new HashSet<string>();
        model.LoadWeights((name, shape) =>
        {
            used.Add(name);
            var values = Values(weights.GetProperty(name).GetProperty("values"));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        });
        Assert.Equal(weights.EnumerateObject().Select(x => x.Name).OrderBy(x => x), used.OrderBy(x => x));
        return model;
    }

    private static IReadOnlyList<Tensor<double>> Parameters(List<LayerBase<double>> layers) =>
        TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList(), -1);

    // Unique fixed random values identify every imported tensor. Both ordinary
    // and Torch-linear-transposed storage are checked; no parameter is omitted.
    private static List<(Tensor<double> Parameter, string Name, double[] Gradient)> MatchParameters(
        IReadOnlyList<Tensor<double>> parameters, JsonElement weights, JsonElement? gradients)
    {
        Assert.Equal(weights.EnumerateObject().Count(), parameters.Count);
        var result = new List<(Tensor<double>, string, double[])>();
        var used = new HashSet<string>();
        foreach (var parameter in parameters)
        {
            var candidates = new List<(string Name, double[] Gradient)>();
            foreach (var entry in weights.EnumerateObject())
            {
                double[] values = Values(entry.Value.GetProperty("values"));
                if (values.Length != parameter.Length) continue;
                double[] gradient = gradients.HasValue ? Values(gradients.Value.GetProperty(entry.Name)) : new double[values.Length];
                if (Equal(parameter, values)) candidates.Add((entry.Name, gradient));
                int[] shape = Ids(entry.Value.GetProperty("shape"));
                if (shape.Length == 2 && shape[0] > 1 && shape[1] > 1)
                {
                    var transposed = Transpose(values, shape[0], shape[1]);
                    if (!values.SequenceEqual(transposed) && Equal(parameter, transposed))
                        candidates.Add((entry.Name, Transpose(gradient, shape[0], shape[1])));
                }
            }
            var matched = Assert.Single(candidates);
            Assert.True(used.Add(matched.Name), "Imported tensor alias counted twice: " + matched.Name);
            result.Add((parameter, matched.Name, matched.Gradient));
        }
        return result;
    }

    private static bool Equal(Tensor<double> actual, double[] expected) => actual.ToArray().Zip(expected,
        (a, b) => Math.Abs(a - b)).All(error => error < 1e-12);

    private static double[] Transpose(double[] values, int rows, int columns)
    {
        var result = new double[values.Length];
        for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) result[c * rows + r] = values[r * columns + c];
        return result;
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"DiffRO {name}: error {error:R}.");
    }

    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static int[] Ids(JsonElement value) => value.EnumerateArray().Select(x => (int)x.GetDouble()).ToArray();
    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "cosyvoice_diffro_reference.json")));

    private sealed class UniformReplay : Random
    {
        private readonly double[] _values;
        internal int Consumed { get; private set; }
        internal UniformReplay(double[] values) => _values = values;
        public override double NextDouble()
        {
            Assert.True(Consumed < _values.Length, "Unexpected extra Gumbel samples.");
            return _values[Consumed++];
        }
    }
}
