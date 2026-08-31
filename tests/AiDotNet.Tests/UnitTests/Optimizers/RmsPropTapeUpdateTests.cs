using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

public class RmsPropTapeUpdateTests
{
    [Fact]
    public async Task CenteredMomentumStep_MatchesGravesEquationsForTwoSteps()
    {
        await Task.Yield();

        var options = new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.1,
            Decay = 0.5,
            InitialMomentum = 0.9,
            UseAdaptiveMomentum = false,
            UseAdaptiveLearningRate = false,
            Epsilon = 0.01,
            Centered = true
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null, options);
        var parameter = Tensor(1.0);

        Step(optimizer, parameter, 2.0);
        Assert.Equal(0.8009925619580022, parameter[0], 12);

        Step(optimizer, parameter, 4.0);
        Assert.Equal(0.3811141615486657, parameter[0], 12);
    }

    [Fact]
    public async Task Reset_ClearsCenteredAveragesAndVelocity()
    {
        await Task.Yield();

        var options = new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.1,
            Decay = 0.5,
            InitialMomentum = 0.9,
            UseAdaptiveMomentum = false,
            UseAdaptiveLearningRate = false,
            Epsilon = 0.01,
            Centered = true
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null, options);
        var parameter = Tensor(1.0);

        Step(optimizer, parameter, 2.0);
        Step(optimizer, parameter, 4.0);
        optimizer.Reset();
        parameter[0] = 1.0;
        Step(optimizer, parameter, 2.0);

        Assert.Equal(0.8009925619580022, parameter[0], 12);
    }

    [Fact]
    public async Task UncenteredTapeStep_PreservesLegacyMomentumFreeBehavior()
    {
        await Task.Yield();

        var options = new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.1,
            Decay = 0.5,
            Epsilon = 0.01,
            Centered = false
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null, options);
        var parameter = Tensor(1.0);

        Step(optimizer, parameter, 2.0);
        Assert.Equal(0.8595716223438077, parameter[0], 12);

        Step(optimizer, parameter, 4.0);
        Assert.Equal(0.7266812568953027, parameter[0], 12);

        Assert.Equal(0.9, options.InitialMomentum);
        Assert.True(options.UseAdaptiveMomentum);
        Assert.False(options.Centered);
    }

    [Fact]
    public async Task UpdateParametersAndReverseUpdate_RoundTripBothVariants()
    {
        await Task.Yield();

        foreach (bool centered in new[] { false, true })
        {
            var options = new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.1,
                Decay = 0.5,
                InitialMomentum = 0.9,
                UseAdaptiveMomentum = false,
                UseAdaptiveLearningRate = false,
                Epsilon = 0.01,
                Centered = centered
            };
            var optimizer = new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null, options);
            var original = new Vector<double>(new[] { 1.0, -2.0 });
            var gradient = new Vector<double>(new[] { 2.0, -4.0 });

            var updated = optimizer.UpdateParameters(original, gradient);
            var restored = optimizer.ReverseUpdate(updated, gradient);

            Assert.Equal(original[0], restored[0], 12);
            Assert.Equal(original[1], restored[1], 12);
        }
    }

    private static void Step(
        RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>> optimizer,
        Tensor<double> parameter,
        double gradientValue)
    {
        var gradient = Tensor(gradientValue);
        optimizer.Step(new TapeStepContext<double>(
            [parameter],
            new Dictionary<Tensor<double>, Tensor<double>> { [parameter] = gradient },
            0.0));
    }

    private static Tensor<double> Tensor(double value)
    {
        var tensor = new Tensor<double>([1]);
        tensor[0] = value;
        return tensor;
    }
}
