using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The optimizer's regularization goes on the side of the gradient clip its RegularizationOrder names. The default
/// is PyTorch's: <c>clip_grad_norm_(params, max_norm)</c> clips the loss gradient g and <c>optimizer.step()</c> then
/// adds <c>weight_decay * theta</c>, so the step uses <c>clip(g) + lambda * theta</c>. RegularizeThenClip is
/// TensorFlow/Keras's order, with the L2 term in the loss and <c>clipnorm</c> on the optimizer:
/// <c>clip(g + lambda * theta)</c>. Both references are written out here with the clip coefficient
/// <c>max_norm / (norm + 1e-6)</c>. Every case also asserts that the two references differ by far more than the
/// tolerance, so a path that ignored the order fails one of its two rows.
/// </summary>
public class RegularizationOrderTests
{
    private const double LearningRate = 0.1;
    private const double Lambda = 0.5;
    private const double MaxNorm = 1.0;

    // The loss gradient is C; its norm (about 10.2) is well over the clip. The second entry is where the orders part:
    // the clipped -0.196 plus lambda * theta = 1 is positive, the unclipped -2 + 1 is not.
    private static readonly double[] C = { 10.0, -2.0 };
    private static readonly double[] Theta0 = { 1.0, 2.0 };

    private static double Norm(IEnumerable<double> v) => Math.Sqrt(v.Sum(x => x * x));

    private static double[] Clip(double[] g)
    {
        double norm = Norm(g);
        double coefficient = norm > MaxNorm ? MaxNorm / (norm + 1e-6) : 1.0;
        return g.Select(x => x * coefficient).ToArray();
    }

    private static double[] AddL2(double[] g, double[] theta) => g.Select((x, i) => x + Lambda * theta[i]).ToArray();

    /// <summary>The gradient the step must use for <paramref name="order"/>.</summary>
    private static double[] EffectiveGradient(double[] g, double[] theta, RegularizationOrder order)
        => order == RegularizationOrder.ClipThenRegularize ? AddL2(Clip(g), theta) : Clip(AddL2(g, theta));

    private static void AssertOrdersDiscriminate(double[] g, double[] theta)
    {
        var torch = EffectiveGradient(g, theta, RegularizationOrder.ClipThenRegularize);
        var keras = EffectiveGradient(g, theta, RegularizationOrder.RegularizeThenClip);
        Assert.True(Norm(torch.Select((x, i) => x - keras[i])) > 1e-2,
            "the two orders give the same gradient here, so the rows cannot tell them apart");
    }

    private static T WithOrder<T>(T options, RegularizationOrder order) where T : GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>
    {
        options.InitialLearningRate = LearningRate;
        options.UseAdaptiveLearningRate = false;
        options.EnableGradientClipping = true;
        options.GradientClippingMethod = GradientClippingMethod.ByNorm;
        options.MaxGradientNorm = MaxNorm;
        options.Regularization = new L2Regularization<double, Tensor<double>, Tensor<double>>(new RegularizationOptions { Strength = Lambda });
        options.RegularizationOrder = order;
        return options;
    }

    private static Tensor<double> LinearObjective(Tensor<double> theta)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(engine.TensorMultiply(theta, new Tensor<double>(C, new[] { C.Length })), null, keepDims: false);
    }

    [Fact]
    public void The_default_order_is_pytorchs()
    {
        Assert.Equal(RegularizationOrder.ClipThenRegularize,
            new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>().RegularizationOrder);
        var chosen = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { RegularizationOrder = RegularizationOrder.RegularizeThenClip };
        Assert.Equal(RegularizationOrder.RegularizeThenClip,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>(chosen).RegularizationOrder);
    }

    [Theory]
    [InlineData(RegularizationOrder.ClipThenRegularize, false)]
    [InlineData(RegularizationOrder.RegularizeThenClip, false)]
    [InlineData(RegularizationOrder.ClipThenRegularize, true)]
    [InlineData(RegularizationOrder.RegularizeThenClip, true)]
    public void An_sgd_objective_step_regularizes_on_the_chosen_side_of_the_clip(RegularizationOrder order, bool accumulated)
    {
        AssertOrdersDiscriminate(C, Theta0);
        var theta = new Tensor<double>((double[])Theta0.Clone(), new[] { Theta0.Length });
        var sgd = new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
            WithOrder(new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>(), order));
        var parameters = new List<Tensor<double>> { theta };
        if (accumulated)
            TapeTrainingStepper<double>.EagerAccumulatedObjectiveStep(parameters, 3, _ => LinearObjective(theta), sgd);
        else
            TapeTrainingStepper<double>.EagerObjectiveStep(parameters, () => LinearObjective(theta), sgd);

        var g = EffectiveGradient(C, Theta0, order);
        for (int i = 0; i < Theta0.Length; i++)
            Assert.Equal(Theta0[i] - LearningRate * g[i], theta[i], 9);
    }

    /// <summary>
    /// Adam clips inside its own step (it shares the global norm with its anomaly guard), so the clip-ahead path must
    /// stop it clipping again. Adam's first step is <c>theta - lr * g / (|g| + eps)</c>: the sign of each effective
    /// gradient entry, which the order flips for the second entry.
    /// </summary>
    [Theory]
    [InlineData(RegularizationOrder.ClipThenRegularize)]
    [InlineData(RegularizationOrder.RegularizeThenClip)]
    public void An_adam_objective_step_regularizes_on_the_chosen_side_of_the_clip(RegularizationOrder order)
    {
        AssertOrdersDiscriminate(C, Theta0);
        var theta = new Tensor<double>((double[])Theta0.Clone(), new[] { Theta0.Length });
        var options = WithOrder(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>(), order);
        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, options);
        TapeTrainingStepper<double>.EagerObjectiveStep(new List<Tensor<double>> { theta }, () => LinearObjective(theta), adam);

        var g = EffectiveGradient(C, Theta0, order);
        for (int i = 0; i < Theta0.Length; i++)
            Assert.Equal(Theta0[i] - LearningRate * g[i] / (Math.Abs(g[i]) + options.Epsilon), theta[i], 9);
    }

    private static FeedForwardNeuralNetwork<double> Network(IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        arch.RandomSeed = 7;
        // No model clip: the optimizer's is the one being ordered, and two clips would compose to the tighter bound.
        return new FeedForwardNeuralNetwork<double>(arch, optimizer, options: new FeedForwardNeuralNetworkOptions { MaxGradNorm = 0.0 });
    }

    private static double[] TrainOnce(Vector<double> start, StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>> options,
        StreamingTrainingMode streaming)
    {
        using var net = Network(new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null, options));
        net.StreamingTraining = streaming;
        net.UpdateParameters(start);
        var x = new Tensor<double>(new[] { 3, 4 });
        var y = new Tensor<double>(new[] { 3, 2 });
        for (int i = 0; i < x.Length; i++) x[i] = ((i % 5) - 2) * 3.0;
        for (int i = 0; i < y.Length; i++) y[i] = ((i % 3) - 1) * 20.0;
        net.Train(x, y);
        return net.GetParameters().ToArray();
    }

    /// <summary>
    /// A network's Train in both orders, on the classic step (eager, or the fused plan when it takes the step) and on
    /// the streaming optimizer-in-backward step. The loss gradient g is read from an unclipped, unregularized SGD step
    /// from the same start (<c>theta - lr * g</c>), so the reference is exact for this network.
    /// </summary>
    [Theory]
    [InlineData(RegularizationOrder.ClipThenRegularize, StreamingTrainingMode.ForceOff)]
    [InlineData(RegularizationOrder.RegularizeThenClip, StreamingTrainingMode.ForceOff)]
    [InlineData(RegularizationOrder.ClipThenRegularize, StreamingTrainingMode.ForceOn)]
    [InlineData(RegularizationOrder.RegularizeThenClip, StreamingTrainingMode.ForceOn)]
    public void A_network_train_step_regularizes_on_the_chosen_side_of_the_clip(RegularizationOrder order, StreamingTrainingMode streaming)
    {
        Vector<double> start;
        using (var seed = Network(new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null)))
            start = seed.GetParameters();
        var theta0 = start.ToArray();

        var plain = new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = LearningRate,
            UseAdaptiveLearningRate = false,
            EnableGradientClipping = false,
        };
        var stepped = TrainOnce(start, plain, StreamingTrainingMode.ForceOff);
        var g = theta0.Select((t, i) => (t - stepped[i]) / LearningRate).ToArray();
        Assert.True(Norm(g) > 2 * MaxNorm, $"the loss gradient norm {Norm(g):G4} is under the clip, so no row discriminates");
        AssertOrdersDiscriminate(g, theta0);

        var actual = TrainOnce(start, WithOrder(new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>(), order), streaming);
        var expectedGradient = EffectiveGradient(g, theta0, order);
        for (int i = 0; i < theta0.Length; i++)
        {
            double expected = theta0[i] - LearningRate * expectedGradient[i];
            Assert.True(Math.Abs(actual[i] - expected) <= 1e-6 * (1 + Math.Abs(expected)),
                $"parameter {i}, {order}, {streaming}: trained {actual[i]:R}, reference {expected:R}");
        }
    }
}
