using System.Text.Json;
using AiDotNet.Models.Options;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class TorchLstmCellReferenceTests
{
    [Fact]
    public void OriginalLstmCell_MatchesEveryGradientAndAdamWUpdate_WithIndependentBiases()
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "torch_lstm_cell_reference.json")));
        var root = document.RootElement; var weights = root.GetProperty("parameters");
        static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        double[] Weight(string name) => Values(weights.GetProperty(name).GetProperty("values"));
        var model = new TorchLstmCellLayer<double>(3, 2);
        model.LoadTorchWeights(Weight("weight_ih"), Weight("weight_hh"), Weight("bias_ih"), Weight("bias_hh"));
        var parameters = new[] { model.InputProjection.GetWeights(), model.InputProjection.GetBiases(),
            model.RecurrentProjection.GetWeights(), model.RecurrentProjection.GetBiases() };
        var owned = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { model });
        Assert.Equal(4, owned.Count);
        foreach (var parameter in parameters) Assert.Contains(owned, value => ReferenceEquals(value, parameter));
        string[] names = { "weight_ih", "bias_ih", "weight_hh", "bias_hh" };
        double[] Convert(string name, string key)
        {
            var entry = weights.GetProperty(name); var values = Values(entry.GetProperty(key));
            if (!name.StartsWith("weight", StringComparison.Ordinal)) return values;
            var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            var result = new double[values.Length];
            for (int o = 0; o < shape[0]; o++) for (int i = 0; i < shape[1]; i++) result[i * shape[0] + o] = values[o * shape[1] + i];
            return result;
        }
        void Close(double[] actual, double[] expected, string name)
        {
            Assert.Equal(expected.Length, actual.Length);
            double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error < 2e-12, name + " error " + error.ToString("R"));
        }
        Tensor<double> Input(string name, params int[] shape) => new(shape, new Vector<double>(Values(root.GetProperty(name))));
        var input = Input("inputs", 4, 2, 3); var hidden = Input("hidden", 2, 2); var cell = Input("cell", 2, 2);
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var state = engine.TensorConcatenate(new[] { hidden, cell }, 1); var outputs = new List<Tensor<double>>();
        for (int step = 0; step < 4; step++)
        {
            var row = engine.Reshape(engine.TensorSlice(input, new[] { step, 0, 0 }, new[] { 1, 2, 3 }), new[] { 2, 3 });
            state = model.Forward(row, state); outputs.Add(state);
        }
        var output = engine.TensorConcatenate(outputs.ToArray(), 0);
        Close(output.ToArray(), Values(root.GetProperty("outputs")), "sequence");
        var loss = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(output, Input("probe", 8, 4)), new[] { 0, 1 }, true), new[] { 1 });
        Close(loss.ToArray(), new[] { root.GetProperty("loss").GetDouble() }, "loss");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { input, hidden, cell }).ToList());
        for (int i = 0; i < parameters.Length; i++) Close(gradients[parameters[i]].ToArray(), Convert(names[i], "gradient"), names[i]);
        Close(gradients[input].ToArray(), Values(root.GetProperty("input_gradient")), "input gradient");
        Close(gradients[hidden].ToArray(), Values(root.GetProperty("hidden_gradient")), "initial hidden gradient");
        Close(gradients[cell].ToArray(), Values(root.GetProperty("cell_gradient")), "initial cell gradient");
        var optimizer = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = .003, Beta1 = .8, Beta2 = .99, Epsilon = 1e-9, WeightDecay = .01,
                EnableGradientClipping = false
            });
        var updated = optimizer.UpdateParameters(new Vector<double>(parameters.SelectMany(x => x.ToArray())),
            new Vector<double>(parameters.SelectMany(x => gradients[x].ToArray())));
        Close(updated.ToArray(), names.SelectMany(name => Convert(name, "updated")).ToArray(), "real AdamW update");
    }
}
