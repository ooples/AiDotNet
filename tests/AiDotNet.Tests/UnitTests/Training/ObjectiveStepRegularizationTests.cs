using System.Collections.Generic;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The objective steps apply the optimizer's configured regularization, as the fused session and EagerStep do. A
/// paper's weight decay is configured as L2 on the optimizer, and the objective steps (diffusion, GAN and tabular
/// generators, Concerto, TensorModelTrainer) once dropped it: a parameter whose objective gradient is zero must still
/// decay by lr * lambda * theta.
/// </summary>
public class ObjectiveStepRegularizationTests
{
    private const double LearningRate = 0.1;
    private const double Lambda = 0.5;

    private static StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>> Sgd(bool withL2) =>
        new(null, withL2
            ? new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = LearningRate,
                UseAdaptiveLearningRate = false,
                Regularization = new L2Regularization<double, Tensor<double>, Tensor<double>>(new RegularizationOptions { Strength = Lambda }),
            }
            : new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = LearningRate,
                UseAdaptiveLearningRate = false,
            });

    // The objective reaches theta with a zero gradient: 0 * sum(theta).
    private static Tensor<double> ZeroGradientObjective(Tensor<double> theta)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMultiplyScalar(engine.ReduceSum(theta, null, keepDims: false), 0.0);
    }

    [Fact]
    public void ObjectiveStep_DecaysAParameterByTheConfiguredL2()
    {
        var theta = new Tensor<double>(new[] { 2.0, -4.0 }, new[] { 2 });
        TapeTrainingStepper<double>.EagerObjectiveStep(new List<Tensor<double>> { theta }, () => ZeroGradientObjective(theta), Sgd(withL2: true));
        Assert.Equal(2.0 * (1 - LearningRate * Lambda), theta[0], 12);
        Assert.Equal(-4.0 * (1 - LearningRate * Lambda), theta[1], 12);
    }

    [Fact]
    public void ObjectiveStep_LeavesTheParameterWithoutRegularization_OrWhenTheObjectiveCarriesIt()
    {
        var plain = new Tensor<double>(new[] { 2.0, -4.0 }, new[] { 2 });
        TapeTrainingStepper<double>.EagerObjectiveStep(new List<Tensor<double>> { plain }, () => ZeroGradientObjective(plain), Sgd(withL2: false));
        Assert.Equal(new[] { 2.0, -4.0 }, plain.ToArray());

        var carried = new Tensor<double>(new[] { 2.0, -4.0 }, new[] { 2 });
        TapeTrainingStepper<double>.EagerObjectiveStep(new List<Tensor<double>> { carried }, () => ZeroGradientObjective(carried),
            Sgd(withL2: true), applyOptimizerRegularization: false);
        Assert.Equal(new[] { 2.0, -4.0 }, carried.ToArray());
    }

    [Fact]
    public void AccumulatedObjectiveStep_DecaysAParameterByTheConfiguredL2()
    {
        var theta = new Tensor<double>(new[] { 2.0, -4.0 }, new[] { 2 });
        TapeTrainingStepper<double>.EagerAccumulatedObjectiveStep(new List<Tensor<double>> { theta }, 3, _ => ZeroGradientObjective(theta),
            Sgd(withL2: true));
        Assert.Equal(2.0 * (1 - LearningRate * Lambda), theta[0], 12);
        Assert.Equal(-4.0 * (1 - LearningRate * Lambda), theta[1], 12);
    }
}
