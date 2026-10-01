using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// A network's own training step applies only a regularization the caller chose. The optimizer options default to
/// L2(0.01); applying that implicit term in Train changed what every network's training did - under Adam a 0.01 L2
/// gradient becomes a near-full learning-rate step, so frozen SeACo backbone parameters drifted and CUPS, Kairos,
/// Word2Vec and VideoMAE stopped reducing their loss. An explicitly configured L2 must still reach the step.
/// </summary>
public class TrainRegularizationIsExplicitOnlyTests
{
    private static FeedForwardNeuralNetwork<double> Network(IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null)
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        arch.RandomSeed = 7;
        return new FeedForwardNeuralNetwork<double>(arch, optimizer);
    }

    private static double[] TrainOnce(Vector<double> start, AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> options)
    {
        // FeedForwardNeuralNetwork trains with the optimizer it was constructed with.
        using var net = Network(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, options));
        net.UpdateParameters(start);
        var x = new Tensor<double>(new[] { 3, 4 });
        var y = new Tensor<double>(new[] { 3, 2 });
        for (int i = 0; i < x.Length; i++) x[i] = ((i % 5) - 2) * 0.3;
        for (int i = 0; i < y.Length; i++) y[i] = ((i % 3) - 1) * 0.5;
        net.Train(x, y);
        net.Train(x, y);
        return net.GetParameters().ToArray();
    }

    [Fact]
    public void The_default_regularization_is_not_applied_but_an_explicit_one_is()
    {
        Vector<double> start;
        using (var seed = Network()) start = seed.GetParameters();

        var implicitDefault = TrainOnce(start, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01 });
        var none = TrainOnce(start, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01,
            Regularization = new NoRegularization<double, Tensor<double>, Tensor<double>>(),
        });
        var explicitL2 = TrainOnce(start, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01,
            Regularization = new L2Regularization<double, Tensor<double>, Tensor<double>>(new RegularizationOptions { Strength = 0.5 }),
        });

        double moved = 0; var s0 = start.ToArray(); for (int i = 0; i < none.Length; i++) moved = System.Math.Max(moved, System.Math.Abs(none[i] - s0[i]));
        Assert.True(moved > 1e-6, $"training did not move the parameters at all (max {moved:E3})");
        Assert.Equal(none, implicitDefault);   // the implicit L2(0.01) default changed nothing
        double maxDiff = 0;
        for (int i = 0; i < none.Length; i++) maxDiff = System.Math.Max(maxDiff, System.Math.Abs(none[i] - explicitL2[i]));
        Assert.True(maxDiff > 1e-6, $"an explicitly configured L2 did not reach the training step (max diff {maxDiff:E3})");
    }

    [Fact]
    public void Copying_options_keeps_whether_the_regularization_was_chosen()
    {
        var defaults = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        var chosen = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            Regularization = new L2Regularization<double, Tensor<double>, Tensor<double>>(),
        };
        Assert.False(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>(defaults).RegularizationExplicitlySet);
        Assert.True(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>(chosen).RegularizationExplicitlySet);
    }

    // A model clone rebuilds its optimizer from a configuration copy, which assigns Regularization through its setter;
    // the copy of an untouched default must not come out "chosen", or the clone trains with an L2 its source never had.
    [Fact]
    public void A_configuration_clone_keeps_whether_the_regularization_was_chosen()
    {
        var defaults = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>();
        var chosen = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            Regularization = new L2Regularization<double, Tensor<double>, Tensor<double>>(),
        };
        Assert.False(((AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>)CloneEngine.CopyConfiguration(defaults)).RegularizationExplicitlySet);
        Assert.True(((AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>)CloneEngine.CopyConfiguration(chosen)).RegularizationExplicitlySet);
    }
}
