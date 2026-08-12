using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

public class OptimizerVectorUpdateStateSafetyTests
{
    private delegate IGradientBasedOptimizer<double, Matrix<double>, Vector<double>> OptimizerFactory();

    [Fact]
    public void UpdateParameters_LengthMismatchDoesNotMutateStateOrInputs()
    {
        foreach ((string name, OptimizerFactory create) in Factories())
        {
            var optimizer = create();
            var parameters = new Vector<double>(new[] { 1.0, -2.0, 3.0 });
            var shortGradient = new Vector<double>(new[] { 0.25, -0.5 });

            ArgumentException error = Assert.Throws<ArgumentException>(
                () => optimizer.UpdateParameters(parameters, shortGradient));
            Assert.Equal("gradient", error.ParamName);

            var gradient = new Vector<double>(new[] { 0.25, -0.5, 0.75 });
            Vector<double> actual = optimizer.UpdateParameters(parameters, gradient);
            Vector<double> expected = create().UpdateParameters(
                new Vector<double>(new[] { 1.0, -2.0, 3.0 }),
                new Vector<double>(new[] { 0.25, -0.5, 0.75 }));

            Assert.NotSame(parameters, actual);
            Assert.Equal(new[] { 1.0, -2.0, 3.0 }, parameters.ToArray());
            Assert.Equal(new[] { 0.25, -0.5, 0.75 }, gradient.ToArray());
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(expected[i].Equals(actual[i]),
                    $"{name} retained partial state after rejecting a mismatched gradient at index {i}: " +
                    $"expected {expected[i]:R}, actual {actual[i]:R}.");
            }
        }
    }

    private static (string Name, OptimizerFactory Create)[] Factories() =>
    [
        ("Adam", () => new AdamOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                UseAMSGrad = true
            })),
        ("AdamW", () => new AdamWOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                WeightDecay = 0.02,
                UseAMSGrad = true
            })),
        ("Adagrad", () => new AdagradOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new AdagradOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("Lion", () => new LionOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new LionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                WeightDecay = 0.02
            })),
        ("Nadam", () => new NadamOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new NadamOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("AdaMax", () => new AdaMaxOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new AdaMaxOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("AdaDelta", () => new AdaDeltaOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new AdaDeltaOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("RMSprop", () => new RootMeanSquarePropagationOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new RootMeanSquarePropagationOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("AMSGrad", () => new AMSGradOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("LAMB", () => new LAMBOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new LAMBOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                WeightDecay = 0.02,
                WarmupEpochs = 0
            }))
    ];
}
