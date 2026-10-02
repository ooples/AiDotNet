using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Regression tests for two defects that made WordCharEmbeddingLayer's gradient check fail for about one random
/// initialization in eight, which showed up as a test-order-dependent failure in CI:
/// <list type="bullet">
/// <item>BidirectionalLayer reversed its input with an untracked copy, so no gradient reached the input through the
/// backward direction, and it merged the backward direction's output without reversing it back into time order.</item>
/// <item>LSTMLayer's eval-mode stacked-weight cache ignored in-place writes to its weights.</item>
/// </list>
/// </summary>
public class BidirectionalRecurrentRegressionTests
{
    private static ILayer<double> BiLstm() =>
        new BidirectionalLayer<double>(new LSTMLayer<double>(5), mergeMode: true,
            (IActivationFunction<double>?)new IdentityActivation<double>());

    private static Tensor<double> Sequence(int seed, params int[] shape)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2 - 1;
        return tensor;
    }

    [Fact(Timeout = 120000)]
    public async Task The_input_gradient_flows_through_both_directions()
    {
        await Task.Yield();
        var layer = BiLstm();
        layer.SetTrainingMode(false);
        var input = Sequence(5, 4, 6, 3);
        layer.Forward(input);   // materialize lazy weights outside the tape
        var projection = Sequence(6, layer.Forward(input).Shape.ToArray());

        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var axes = Enumerable.Range(0, output.Shape.Length).ToArray();
            var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, projection), axes, keepDims: false);
            gradient = tape.ComputeGradients(loss, new[] { input })[input];
        }

        double Loss()
        {
            var output = layer.Forward(input);
            double sum = 0;
            for (int i = 0; i < output.Length; i++) sum += output[i] * projection[i];
            return sum;
        }

        // Every input element, not a direction: the backward direction reaches each one.
        const double eps = 1e-6;
        for (int i = 0; i < input.Length; i++)
        {
            double original = input[i];
            input[i] = original + eps;
            double plus = Loss();
            input[i] = original - eps;
            double minus = Loss();
            input[i] = original;
            double numerical = (plus - minus) / (2 * eps);
            Assert.True(Math.Abs(gradient[i] - numerical) <= 1e-5 * Math.Max(1, Math.Abs(numerical)),
                $"input[{i}]: analytical {gradient[i]:G9}, numerical {numerical:G9}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Position_t_pairs_the_forward_and_backward_states_at_t()
    {
        await Task.Yield();
        // With identical weights in both directions, reversing the sequence must reverse the output: the backward
        // state at t has read x[T-1..t], exactly the forward state at T-1-t of the reversed sequence.
        var inner = new LSTMLayer<double>(5);
        var layer = new BidirectionalLayer<double>(inner, mergeMode: true,
            (IActivationFunction<double>?)new IdentityActivation<double>());
        layer.SetTrainingMode(false);
        var input = Sequence(9, 1, 6, 3);
        layer.Forward(input);
        var parameters = layer.GetParameters();
        int half = parameters.Length / 2;
        for (int i = 0; i < half; i++) parameters[half + i] = parameters[i];
        layer.SetParameters(parameters);

        var output = layer.Forward(input);
        var reversedOutput = layer.Forward(AiDotNetEngine.Current.TensorFlip(input, new[] { 1 }));
        int steps = input.Shape[1], hidden = output.Shape[2];
        for (int t = 0; t < steps; t++)
            for (int h = 0; h < hidden; h++)
                Assert.Equal(output[0, t, h], reversedOutput[0, steps - 1 - t, h], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Eval_mode_lstm_sees_an_in_place_weight_write()
    {
        await Task.Yield();
        var layer = new LSTMLayer<float>(5);
        layer.SetTrainingMode(false);
        var rng = RandomHelper.CreateSeededRandom(3);
        var input = new Tensor<float>(new[] { 2, 4, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var before = layer.Forward(input).ToArray();
        var weights = AiDotNet.Training.TapeTrainingStep<float>.CollectParameters(new[] { layer }, structureVersion: -1);
        foreach (var tensor in weights)
            for (int i = 0; i < tensor.Length; i++) tensor[i] += 0.25f;   // an optimizer-style in-place step
        var after = layer.Forward(input).ToArray();

        Assert.NotEqual(before, after);
    }
}
