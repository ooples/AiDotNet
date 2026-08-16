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
    public void FactoriesCoverEveryConcreteGradientBasedOptimizer()
    {
        Type openBaseType = typeof(GradientBasedOptimizerBase<,,>);
        Type[] discovered = openBaseType.Assembly.GetTypes()
            .Where(type => !type.IsAbstract && InheritsOpenGeneric(type, openBaseType))
            .Select(type => type.IsGenericType ? type.GetGenericTypeDefinition() : type)
            .OrderBy(type => type.FullName)
            .ToArray();
        Type[] covered = Factories()
            .Select(factory => factory.Create().GetType().GetGenericTypeDefinition())
            .Distinct()
            .OrderBy(type => type.FullName)
            .ToArray();

        Assert.Equal(discovered, covered);
    }

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

    [Fact]
    public void UpdateParameters_ParameterLengthChangeResetsVectorState()
    {
        foreach ((string name, OptimizerFactory create) in Factories())
        {
            var optimizer = create();
            optimizer.UpdateParameters(
                new Vector<double>(new[] { 1.0, -2.0, 3.0 }),
                new Vector<double>(new[] { 0.25, -0.5, 0.75 }));

            var parameters = new Vector<double>(new[] { 0.5, -0.25 });
            var gradient = new Vector<double>(new[] { 0.1, -0.2 });
            Vector<double> actual = optimizer.UpdateParameters(parameters, gradient);
            Vector<double> expected = create().UpdateParameters(
                new Vector<double>(new[] { 0.5, -0.25 }),
                new Vector<double>(new[] { 0.1, -0.2 }));

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(expected[i].Equals(actual[i]),
                    $"{name} retained state from a differently sized parameter vector at index {i}: " +
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
            })),
        ("Adam8Bit", () => new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                BlockSize = 2,
                QuantizationPercentile = 100,
                UseStochasticRounding = false
            })),
        ("Momentum", () => new MomentumOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("NAG", () => new NesterovAcceleratedGradientOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("FTRL", () => new FTRLOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>>())),
        ("LARS", () => new LARSOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new LARSOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                WarmupEpochs = 0
            })),
        ("LBFGS", () => new LBFGSOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("BFGS", () => new BFGSOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("DFP", () => new DFPOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new DFPOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("TrustRegion", () => new TrustRegionOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new TrustRegionOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("GradientDescent", () => new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("SGD", () => new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("ADMM", () => new ADMMOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new ADMMOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.01 })),
        ("ConjugateGradient", () => new ConjugateGradientOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("CoordinateDescent", () => new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("LevenbergMarquardt", () => new LevenbergMarquardtOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new LevenbergMarquardtOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("MiniBatchGradientDescent", () => new MiniBatchGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new MiniBatchGradientDescentOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("NewtonMethod", () => new NewtonMethodOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new NewtonMethodOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("ProximalGradientDescent", () => new ProximalGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new ProximalGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("RAdam", () => new RAdamOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("ASGD", () => new ASGDOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01
            })),
        ("Rprop", () => new RpropOptimizer<double, Matrix<double>, Vector<double>>(
            null!, new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialStepSize = 0.01
            }))
    ];

    private static bool InheritsOpenGeneric(Type type, Type openBaseType)
    {
        for (Type? current = type.BaseType; current is not null; current = current.BaseType)
        {
            if (current.IsGenericType && current.GetGenericTypeDefinition() == openBaseType)
            {
                return true;
            }
        }

        return false;
    }
}
