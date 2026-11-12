using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class AlignTtsAcousticReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CompleteGraphAndAllFourPhases_MatchEnumeratedTorchObjectivesAndGradients(int phase)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "aligntts_acoustic_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement values) => values.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        double[] Weight(string key) => Values(reference.GetProperty("weights").GetProperty(key));
        double[] Gradient(string key) => reference.GetProperty("phases")[phase].GetProperty("gradients").TryGetProperty(key, out var value)
            ? Values(value) : new double[Weight(key).Length];
        var tokens = new Tensor<double>(new[] { 4 }, new Vector<double>(Values(reference.GetProperty("tokens"))));
        var target = new Tensor<double>(new[] { 6, 5 }, new Vector<double>(Values(reference.GetProperty("target"))));
        using var model = new AlignTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 5), new AlignTTSOptions
        {
            VocabSize = 12, EncoderDim = 8, HiddenDim = 8, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1,
            FftFilterSize = 16, FftKernelSize = 3, DropoutRate = 0, MelChannels = 5,
            DurationPredictorDim = 8, DurationPredictorLayers = 1, MixDensityHiddenSize = 8, MixDensityHiddenLayers = 3
        });
        model.DisableAutoStreaming(); model.SetTrainingMode(false);
        for (int i = 1; i <= 4; i++)
        {
            model.CurrentPhase = (AlignTTSTrainingPhase)i;
            ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, target);
        }
        model.CurrentPhase = (AlignTTSTrainingPhase)(phase + 1);
        static T Field<T>(object owner, string name) => (T)owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;
        static double[] Transpose(double[] values, int output, int input)
        {
            var result = new double[values.Length];
            for (int o = 0; o < output; o++) for (int i = 0; i < input; i++) result[i * output + o] = values[o * input + i];
            return result;
        }
        var expected = new Dictionary<Tensor<double>, (string Name, double[] Gradient)>();
        void Add(Tensor<double> tensor, string name, double[] weight, double[] gradient)
        {
            Assert.Equal(tensor.Length, weight.Length);
            for (int i = 0; i < weight.Length; i++) tensor.SetFlat(i, weight[i]);
            AiDotNetEngine.Current.InvalidatePersistentTensor(tensor);
            expected.Add(tensor, (name, gradient));
        }
        void Map(LayerBase<double> layer, string name)
        {
            if (layer is FeedForwardTransformerBlock<double> fft)
            {
                var attention = Field<MultiHeadAttentionLayer<double>>(fft, "_attention");
                var qkv = Weight(name + ".attention.in_proj_weight"); var gradient = Gradient(name + ".attention.in_proj_weight");
                Tensor<double>[] projections = [attention.GetQueryWeights(), attention.GetKeyWeights(), attention.GetValueWeights()];
                for (int i = 0; i < 3; i++) Add(projections[i], name + ".qkv." + i,
                    Transpose(qkv.Skip(i * 64).Take(64).ToArray(), 8, 8), Transpose(gradient.Skip(i * 64).Take(64).ToArray(), 8, 8));
                Add(attention.GetOutputWeights(), name + ".attention.out_proj.weight", Transpose(Weight(name + ".attention.out_proj.weight"), 8, 8),
                    Transpose(Gradient(name + ".attention.out_proj.weight"), 8, 8));
                Add(Field<Tensor<double>>(attention, "_outputBias"), name + ".attention.out_proj.bias",
                    Weight(name + ".attention.out_proj.bias"), Gradient(name + ".attention.out_proj.bias"));
                Map(Field<LayerNormalizationLayer<double>>(fft, "_attentionNorm"), name + ".norm1");
                Map(Field<LayerNormalizationLayer<double>>(fft, "_convNorm"), name + ".norm2");
                Map(Field<Conv1DLayer<double>>(fft, "_conv1"), name + ".conv1");
                Map(Field<Conv1DLayer<double>>(fft, "_conv2"), name + ".conv2");
                return;
            }
            if (layer is DenseLayer<double> dense)
            {
                var weight = Weight(name + ".weight"); var bias = Weight(name + ".bias");
                int outputs = bias.Length, inputs = weight.Length / outputs;
                TorchParameters.Linear(AiDotNetEngine.Current, dense, inputs, outputs, weight, bias);
                expected.Add(dense.GetWeights(), (name + ".weight", Transpose(Gradient(name + ".weight"), outputs, inputs)));
                expected.Add(dense.GetBiases(), (name + ".bias", Gradient(name + ".bias")));
                return;
            }
            if (layer is EmbeddingLayer<double> embedding)
            {
                Add(Field<Tensor<double>>(embedding, "_embeddingTensor"), name + ".weight", Weight(name + ".weight"), Gradient(name + ".weight"));
                return;
            }
            if (layer is LayerNormalizationLayer<double> norm)
                TorchParameters.LayerNorm(AiDotNetEngine.Current, norm, Weight(name + ".weight"), Weight(name + ".bias"));
            else if (layer is Conv1DLayer<double> convolution)
                convolution.SetParameters(new Vector<double>(Weight(name + ".weight").Concat(Weight(name + ".bias")).ToArray()));
            else throw new InvalidOperationException("Unmapped reference layer: " + layer.GetType().Name);
            var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
            Assert.Equal(2, parameters.Count);
            expected.Add(parameters[0], (name + ".weight", Gradient(name + ".weight")));
            expected.Add(parameters[1], (name + ".bias", Gradient(name + ".bias")));
        }
        Map((LayerBase<double>)model.Layers[0], "embedding"); Map((LayerBase<double>)model.Layers[2], "encoder");
        Map((LayerBase<double>)model.Layers[4], "decoder"); Map((LayerBase<double>)model.Layers[5], "mel");
        var duration = Field<List<LayerBase<double>>>(model, "_durationPredictor");
        Map(duration[0], "duration_embedding"); Map(duration[2], "duration_encoder"); Map(duration[3], "duration_projection");
        var density = Field<List<LayerBase<double>>>(model, "_mixDensityNetwork");
        for (int i = 0; i < density.Count; i++) if (density[i] is not ActivationLayer<double>) Map(density[i], "density." + i);
        var components = (List<LayerBase<double>>)typeof(TtsModelBase<double>).GetField("ComponentLayers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var allParameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Concat(components.Cast<ILayer<double>>()).ToList());
        Assert.Equal(59, allParameters.Count); Assert.Equal(allParameters.Count, expected.Count);
        Assert.All(allParameters, parameter => Assert.True(expected.ContainsKey(parameter)));
        object Invoke(string name, params object[] arguments) => typeof(AlignTTS<double>).GetMethod(name,
            BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, arguments)!;
        var hidden = (Tensor<double>)Invoke("RunEncoder", tokens);
        Close(hidden, Values(reference.GetProperty("hidden")), "encoder");
        var likelihood = (Tensor<double>)Invoke("GaussianLogLikelihood", hidden, target);
        Close(likelihood, Values(reference.GetProperty("likelihood")), "density likelihood");
        var durations = (int[])Invoke("ViterbiDurations", likelihood);
        Assert.Equal(reference.GetProperty("durations").EnumerateArray().Select(x => x.GetInt32()).ToArray(), durations);
        Close((Tensor<double>)Invoke("RunDecoder", LengthRegulator.Expand(hidden, durations)), Values(reference.GetProperty("mel")), "mel decoder");
        Close((Tensor<double>)Invoke("RunDurationPredictor", tokens), Values(reference.GetProperty("log_duration")), "duration graph");
        using var tape = new GradientTape<double>();
        var loss = (Tensor<double>)Invoke("PhaseObjective", tokens, target);
        Close(loss, new[] { reference.GetProperty("phases")[phase].GetProperty("loss").GetDouble() }, "phase objective");
        var gradients = tape.ComputeGradients(loss, sources: allParameters);
        foreach (var parameter in allParameters)
        {
            var wanted = expected[parameter];
            if (gradients.TryGetValue(parameter, out var gradient)) Close(gradient, wanted.Gradient, wanted.Name);
            else Assert.All(wanted.Gradient, value => Assert.Equal(0, value));
        }
        var active = (IReadOnlyList<Tensor<double>>)Invoke("SelectTrainableParametersForTraining", allParameters);
        Assert.Equal(new[] { 28, 15, 43, 16 }[phase], active.Count);
        foreach (var parameter in active)
        {
            string name = expected[parameter].Name;
            Assert.True(phase switch
            {
                0 => name.StartsWith("embedding.") || name.StartsWith("encoder.") || name.StartsWith("density."),
                1 => name.StartsWith("decoder.") || name.StartsWith("mel."),
                2 => !name.StartsWith("duration_"),
                _ => name.StartsWith("duration_")
            }, "Unexpected phase-owned parameter " + name);
        }
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-9, $"Complete AlignTTS {name} error {error:R}.");
    }
}
