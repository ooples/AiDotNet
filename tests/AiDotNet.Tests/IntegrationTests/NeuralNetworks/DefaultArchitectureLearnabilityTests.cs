using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression guard for the degenerate default-architecture bug: for low-dimensional tabular inputs
/// the unfloored complexity formulas produced width-1/2 hidden layers ((inputSize+outputSize)/2 = 1
/// for a 1→1 Simple net). With a width-2 ReLU hidden layer the init frequently leaves EVERY path
/// dead over the whole input range (e.g. both second-layer weights negative), so no weight receives
/// a gradient and only the output bias trains — the network is permanently a constant function.
/// Observed concretely: a 1-input Simple/Medium regression net trained on y = 2x + 1 predicted the
/// identical constant for every probe input, with only the final bias parameter moving.
/// </summary>
public class DefaultArchitectureLearnabilityTests
{
    private static NeuralNetwork<double> BuildNet(NetworkComplexity complexity) => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: complexity,
            inputSize: 1,
            outputSize: 1));

    [Theory]
    [InlineData(NetworkComplexity.Simple)]
    [InlineData(NetworkComplexity.Medium)]
    public void Default_one_input_regression_net_is_not_a_constant_function(NetworkComplexity complexity)
    {
        var net = BuildNet(complexity);

        // Train y = 2x + 1 over x in [-1, 1], mini-batches of 10, 250 epochs — the exact setup that
        // exposed the dead net (predictions frozen at one constant for every input).
        const int n = 70;
        for (int epoch = 0; epoch < 250; epoch++)
        {
            for (int start = 0; start < n; start += 10)
            {
                int cnt = Math.Min(10, n - start);
                var bx = new double[cnt];
                var by = new double[cnt];
                for (int i = 0; i < cnt; i++)
                {
                    double xi = (start + i - 35) / 35.0;
                    bx[i] = xi;
                    by[i] = (2.0 * xi) + 1.0;
                }

                net.Train(
                    new Tensor<double>(new[] { cnt, 1 }, new Vector<double>(bx)),
                    new Tensor<double>(new[] { cnt, 1 }, new Vector<double>(by)));
            }
        }

        var probes = new[] { -0.8, -0.4, 0.0, 0.4, 0.8 };
        var preds = probes
            .Select(p => net.Predict(new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { p })))[0])
            .ToArray();

        // Statistical invariants, not exact values: the trained function must (1) actually depend on
        // its input and (2) be predominantly increasing on an increasing target. The dead net fails
        // both (identical constant at every probe).
        Assert.True(preds.Max() - preds.Min() > 0.5,
            $"{complexity}: net is (near-)constant over the input range — hidden layers dead, only the output bias trained. " +
            $"preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");

        int rising = 0;
        for (int i = 1; i < preds.Length; i++)
        {
            if (preds[i] > preds[i - 1])
            {
                rising++;
            }
        }

        Assert.True(rising >= 3,
            $"{complexity}: predictions not predominantly increasing in x. " +
            $"preds=[{string.Join(", ", preds.Select(v => v.ToString("F3")))}]");
    }

    [Fact]
    public void Default_hidden_layers_have_a_sane_minimum_width()
    {
        // A 1→1 Simple net previously had 4 trainable parameters total (width-1 hidden layer).
        // With the width floor the parameter count must reflect non-degenerate hidden layers.
        var net = BuildNet(NetworkComplexity.Simple);
        Assert.True(net.GetParameters().Length >= 25,
            $"Expected floored hidden width (>=25 params for 1→8→1), got {net.GetParameters().Length} — hidden sizing regressed to degenerate widths.");
    }
}
