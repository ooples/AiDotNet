using System.Reflection;
using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class VoiceboxNetworkReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void CompleteAudioAndDurationNetworks_MatchTorchOutputsMaskedObjectiveAndAllGradients(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "voicebox_network_reference.json")));
        var reference = document.RootElement; var example = reference.GetProperty("cases")[index];
        double[] Values(JsonElement values) => values.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => Values(example.GetProperty("gradients").GetProperty(key));
        var engine = AiDotNetEngine.Current;
        var layers = new List<LayerBase<double>>();
        var options = new VoiceboxOptions
        {
            MelChannels = 3, PhoneVocabularySize = 8, PhoneEmbeddingDim = 4,
            AudioModelDim = 8, AudioHeads = 2, AudioFeedForwardDim = 16, AudioLayers = 4,
            AudioConvPositionWidth = 3, AudioConvPositionGroups = 2, AudioConvPositionDepth = 2,
            AudioAttentionDropout = 0, AudioActivationDropout = 0,
            DurationModelDim = 8, DurationHeads = 2, DurationFeedForwardDim = 16, DurationLayers = 2,
            DurationConvPositionWidth = 3, DurationConvPositionGroups = 2, DurationConvPositionDepth = 2,
            DurationAttentionDropout = 0, DurationActivationDropout = 0
        };
        var audio = new VoiceboxAudioModel<double>(engine, layers, options);
        var duration = new VoiceboxDurationModel<double>(engine, layers, options);
        foreach (var layer in layers) layer.SetTrainingMode(false);
        int[] audioIds = [2, 3, 4, 1, 3, 5], durationIds = [2, 3, 4, 1];
        audio.Forward(new Tensor<double>(new[] { 6, 3 }), new Tensor<double>(new[] { 6, 3 }), audioIds, .37);
        duration.Forward(durationIds, new Tensor<double>(new[] { 4, 1 }));
        static T Field<T>(object owner, string name) => (T)owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;
        static double[] Transpose(double[] values, int output, int input)
        {
            var result = new double[values.Length];
            for (int o = 0; o < output; o++) for (int i = 0; i < input; i++) result[i * output + o] = values[o * input + i];
            return result;
        }
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        void Copy(Tensor<double> tensor, string name, bool transpose = false)
        {
            var weight = Weight(name); var gradient = Gradient(name);
            if (transpose)
            {
                int output = tensor.Shape[1], input = tensor.Shape[0];
                weight = Transpose(weight, output, input); gradient = Transpose(gradient, output, input);
            }
            Assert.Equal(tensor.Length, weight.Length);
            for (int i = 0; i < weight.Length; i++) tensor.SetFlat(i, weight[i]);
            engine.InvalidatePersistentTensor(tensor);
            expected.Add(tensor, (name, gradient));
        }
        void Map(LayerBase<double> layer, string name)
        {
            if (layer is DenseLayer<double> dense)
            {
                Copy(dense.GetWeights(), name + ".weight", true); Copy(dense.GetBiases(), name + ".bias");
            }
            else if (layer is EmbeddingLayer<double> embedding) Copy(Field<Tensor<double>>(embedding, "_embeddingTensor"), name + ".weight");
            else if (layer is NormedConv1DLayer<double> conv)
            {
                Copy(Field<Tensor<double>>(conv, "_direction"), name + ".weight_v");
                Copy(Field<Tensor<double>>(conv, "_gain"), name + ".weight_g");
                Copy(Field<Tensor<double>>(conv, "_bias"), name + ".bias");
            }
            else
            {
                Assert.IsType<LayerNormalizationLayer<double>>(layer);
                var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
                Assert.Equal(2, owned.Count); Copy(owned[0], name + ".weight"); Copy(owned[1], name + ".bias");
            }
        }
        void Transformer(VoiceboxTransformer<double> transformer, string name)
        {
            var position = Field<VoiceboxConvPosition<double>>(transformer, "_position");
            var convolutions = Field<List<NormedConv1DLayer<double>>>(position, "_convolutions");
            for (int i = 0; i < convolutions.Count; i++) Map(convolutions[i], name + ".position." + i);
            var blocks = Field<List<VoiceboxTransformerBlock<double>>>(transformer, "_blocks");
            for (int i = 0; i < blocks.Count; i++)
                foreach (var (field, suffix) in new[] { ("_attentionNorm", "attention_norm"), ("_query", "query"), ("_key", "key"),
                    ("_value", "value"), ("_attentionOut", "attention_out"), ("_feedForwardNorm", "ff_norm"),
                    ("_feedForwardIn", "ff_in"), ("_feedForwardOut", "ff_out") })
                    Map(Field<LayerBase<double>>(blocks[i], field), $"{name}.blocks.{i}.{suffix}");
            var skips = Field<List<DenseLayer<double>>>(transformer, "_skipProjections");
            for (int i = 0; i < skips.Count; i++) Map(skips[i], name + ".skips." + i);
            Map(Field<LayerNormalizationLayer<double>>(transformer, "_finalNorm"), name + ".final_norm");
        }
        foreach (var (model, name) in new[] { ((object)audio, "audio"), ((object)duration, "duration") })
        {
            Map(Field<EmbeddingLayer<double>>(model, "_phones"), name + ".phones");
            Map(Field<DenseLayer<double>>(model, "_input"), name + ".input");
            Transformer(Field<VoiceboxTransformer<double>>(model, "_transformer"), name + ".transformer");
            Map(Field<DenseLayer<double>>(model, "_output"), name + ".output");
        }
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList());
        Assert.Equal(128, parameters.Count); Assert.Equal(parameters.Count, expected.Count);
        Assert.All(parameters, parameter => Assert.True(expected.ContainsKey(parameter)));
        Tensor<double> Tensor(string name, params int[] shape) => new(shape, new Vector<double>(Values(reference.GetProperty(name))));
        var state = Tensor("state", 6, 3); var audioContext = Tensor("audio_context", 6, 3); var durationContext = Tensor("duration_context", 4, 1);
        var audioTarget = Tensor("audio_target", 6, 3); var durationTarget = Tensor("duration_target", 4, 1);
        using var tape = new GradientTape<double>();
        bool conditioned = example.GetProperty("conditioned").GetBoolean();
        var audioOutput = audio.Forward(state, audioContext, audioIds, .37, conditioned);
        var durationOutput = duration.Forward(durationIds, durationContext, conditioned);
        Close(audioOutput, Values(example.GetProperty("audio")), "audio vector field");
        Close(durationOutput, Values(example.GetProperty("duration")), "duration regression");
        var audioLoss = VoiceboxObjectives<double>.MaskedMse(engine, engine.TensorSubtract(audioOutput, audioTarget),
            new[] { true, false, true, true, false, true });
        var durationMask = new Tensor<double>(new[] { 4, 1 }, new Vector<double>(new[] { 0.0, 1.0, 1.0, 1.0 }));
        var durationLoss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(durationMask,
            engine.TensorAbs(engine.TensorSubtract(durationOutput, durationTarget))), new[] { 0, 1 }, false), 1.0 / 3);
        var loss = engine.TensorAdd(engine.Reshape(audioLoss, new[] { 1 }), engine.Reshape(durationLoss, new[] { 1 }));
        Close(loss, new[] { example.GetProperty("loss").GetDouble() }, "combined masked objective");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { state, audioContext, durationContext }).ToList());
        foreach (var parameter in parameters)
        {
            var wanted = expected[parameter];
            if (gradients.TryGetValue(parameter, out var gradient)) Close(gradient, wanted.Gradient, wanted.Name);
            else Assert.All(wanted.Gradient, value => Assert.Equal(0, value));
        }
        foreach (var (tensor, key) in new[] { (state, "state_gradient"), (audioContext, "audio_context_gradient"), (durationContext, "duration_context_gradient") })
        {
            var wanted = Values(example.GetProperty(key));
            if (gradients.TryGetValue(tensor, out var gradient)) Close(gradient, wanted, key);
            else Assert.All(wanted, value => Assert.Equal(0, value));
        }
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"Voicebox full-network {name} error {error:R}.");
    }
}
