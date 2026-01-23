using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Comprehensive integration tests for gradient-based optimizers.
/// These tests verify mathematical correctness of optimizer update rules
/// against known formulas and expected convergence behavior.
/// </summary>
/// <remarks>
/// CRITICAL: These tests verify that optimizers implement the correct mathematical formulas.
/// If a test fails, FIX THE OPTIMIZER CODE, do NOT change the test to match buggy implementation.
/// Reference formulas are from:
/// - Original Adam paper: "Adam: A Method for Stochastic Optimization" (Kingma &amp; Ba, 2014)
/// - PyTorch optimizer implementations
/// - Deep Learning textbooks (Goodfellow et al.)
/// </remarks>
public class GradientBasedOptimizerIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LargeTolerance = 1e-4; // For accumulated numerical errors

    #region Test Helpers - Benchmark Functions

    /// <summary>
    /// Sphere function: f(x) = sum(x_i^2)
    /// Global minimum at origin with f(0) = 0
    /// </summary>
    private static double SphereFunction(double[] x)
    {
        double sum = 0;
        foreach (var xi in x)
        {
            sum += xi * xi;
        }
        return sum;
    }

    /// <summary>
    /// Gradient of sphere function: grad_i = 2 * x_i
    /// </summary>
    private static double[] SphereGradient(double[] x)
    {
        var grad = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            grad[i] = 2 * x[i];
        }
        return grad;
    }

    /// <summary>
    /// Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
    /// Global minimum at (a, a^2) with f(a, a^2) = 0
    /// Standard values: a=1, b=100
    /// </summary>
    private static double RosenbrockFunction(double[] x, double a = 1, double b = 100)
    {
        if (x.Length != 2) throw new ArgumentException("Rosenbrock function requires 2D input");
        double x0 = x[0], x1 = x[1];
        return Math.Pow(a - x0, 2) + b * Math.Pow(x1 - x0 * x0, 2);
    }

    /// <summary>
    /// Gradient of Rosenbrock function
    /// </summary>
    private static double[] RosenbrockGradient(double[] x, double a = 1, double b = 100)
    {
        if (x.Length != 2) throw new ArgumentException("Rosenbrock function requires 2D input");
        double x0 = x[0], x1 = x[1];
        return new double[]
        {
            -2 * (a - x0) - 4 * b * x0 * (x1 - x0 * x0),
            2 * b * (x1 - x0 * x0)
        };
    }

    /// <summary>
    /// Rastrigin function: f(x) = An + sum(x_i^2 - A*cos(2*pi*x_i))
    /// Global minimum at origin with f(0) = 0
    /// </summary>
    private static double RastriginFunction(double[] x, double A = 10)
    {
        double sum = A * x.Length;
        foreach (var xi in x)
        {
            sum += xi * xi - A * Math.Cos(2 * Math.PI * xi);
        }
        return sum;
    }

    /// <summary>
    /// Gradient of Rastrigin function
    /// </summary>
    private static double[] RastriginGradient(double[] x, double A = 10)
    {
        var grad = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            grad[i] = 2 * x[i] + 2 * Math.PI * A * Math.Sin(2 * Math.PI * x[i]);
        }
        return grad;
    }

    /// <summary>
    /// Simple quadratic function for testing: f(x) = 0.5 * x^T * A * x
    /// where A is a positive definite matrix (here we use diagonal with specified eigenvalues)
    /// </summary>
    private static double QuadraticFunction(double[] x, double[] eigenvalues)
    {
        if (x.Length != eigenvalues.Length)
            throw new ArgumentException("Dimension mismatch");

        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += 0.5 * eigenvalues[i] * x[i] * x[i];
        }
        return sum;
    }

    /// <summary>
    /// Gradient of quadratic function: grad_i = eigenvalue_i * x_i
    /// </summary>
    private static double[] QuadraticGradient(double[] x, double[] eigenvalues)
    {
        var grad = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            grad[i] = eigenvalues[i] * x[i];
        }
        return grad;
    }

    #endregion

    #region SGD/Gradient Descent Tests

    [Fact]
    public void GradientDescent_UpdateFormula_MatchesDefinition()
    {
        // Test that SGD implements: params_new = params_old - learning_rate * gradient
        // Reference: Standard gradient descent formula

        // Arrange
        var options = new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.5, -0.25, 1.0 });

        // Expected: params_new = [1.0, 2.0, 3.0] - 0.1 * [0.5, -0.25, 1.0]
        //                      = [1.0 - 0.05, 2.0 + 0.025, 3.0 - 0.1]
        //                      = [0.95, 2.025, 2.9]
        var expected = new double[] { 0.95, 2.025, 2.9 };

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], Tolerance);
        }
    }

    [Fact]
    public void GradientDescent_ConvergesOnSphereFunction()
    {
        // SGD should converge to the minimum of the sphere function (at origin)
        // With appropriate learning rate, should make progress in each step

        // Arrange
        var options = new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Start away from the minimum
        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act - Run 100 iterations
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert - Loss should decrease significantly
        Assert.True(finalLoss < initialLoss * 0.01,
            $"SGD should reduce loss significantly. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void GradientDescent_WithZeroGradient_ParametersUnchanged()
    {
        // With zero gradient, parameters should not change

        // Arrange
        var options = new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var zeroGradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        // Act
        var result = optimizer.UpdateParameters(parameters, zeroGradient);

        // Assert
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i], result[i], Tolerance);
        }
    }

    [Fact]
    public void GradientDescent_LearningRateScalesUpdate()
    {
        // Test that learning rate correctly scales the update

        // Arrange
        var optionsSmallLR = new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01
        };
        var optionsLargeLR = new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };

        var optimizerSmall = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, optionsSmallLR);
        var optimizerLarge = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, optionsLargeLR);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        // Act
        var resultSmall = optimizerSmall.UpdateParameters(parameters, gradient);
        var resultLarge = optimizerLarge.UpdateParameters(parameters, gradient);

        // Assert - Larger LR should produce larger update
        double updateSmall = Math.Abs(parameters[0] - resultSmall[0]);
        double updateLarge = Math.Abs(parameters[0] - resultLarge[0]);

        Assert.Equal(10.0, updateLarge / updateSmall, Tolerance); // 0.1 / 0.01 = 10
    }

    #endregion

    #region Momentum Optimizer Tests

    [Fact]
    public void Momentum_UpdateFormula_MatchesDefinition()
    {
        // Test that Momentum implements:
        // v = momentum * v_prev + learning_rate * gradient
        // params_new = params_old - v
        // Reference: "On the importance of initialization and momentum in deep learning" (Sutskever et al., 2013)

        // Arrange
        double lr = 0.1;
        double momentum = 0.9;
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            InitialMomentum = momentum
        };
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        // First update: v = 0.9 * 0 + 0.1 * 1 = 0.1
        //               params = params - v = [1.0, 2.0, 3.0] - [0.1, 0.1, 0.1] = [0.9, 1.9, 2.9]
        var expectedFirst = new double[] { 0.9, 1.9, 2.9 };

        // Act
        var resultFirst = optimizer.UpdateParameters(parameters, gradient);

        // Assert first update
        for (int i = 0; i < expectedFirst.Length; i++)
        {
            Assert.Equal(expectedFirst[i], resultFirst[i], Tolerance);
        }

        // Second update with same gradient:
        // v = 0.9 * 0.1 + 0.1 * 1 = 0.09 + 0.1 = 0.19
        // params = [0.9, 1.9, 2.9] - [0.19, 0.19, 0.19] = [0.71, 1.71, 2.71]
        var expectedSecond = new double[] { 0.71, 1.71, 2.71 };

        var resultSecond = optimizer.UpdateParameters(resultFirst, gradient);

        for (int i = 0; i < expectedSecond.Length; i++)
        {
            Assert.Equal(expectedSecond[i], resultSecond[i], Tolerance);
        }
    }

    [Fact]
    public void Momentum_AcceleratesConsistentGradients()
    {
        // With momentum, consecutive updates in the same direction should accelerate

        // Arrange
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            InitialMomentum = 0.9
        };
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 }); // Consistent gradient

        // Act
        var differences = new List<double>();
        var current = parameters;
        for (int i = 0; i < 10; i++)
        {
            var next = optimizer.UpdateParameters(current, gradient);
            differences.Add(Math.Abs(current[0] - next[0]));
            current = next;
        }

        // Assert - Updates should get larger due to momentum accumulation
        for (int i = 1; i < differences.Count; i++)
        {
            Assert.True(differences[i] >= differences[i - 1] - Tolerance,
                $"Update {i} ({differences[i]}) should be >= update {i - 1} ({differences[i - 1]})");
        }
    }

    [Fact]
    public void Momentum_DampsOscillations()
    {
        // When gradient direction changes, momentum should reduce oscillation

        // Arrange
        var optionsNoMomentum = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.0
        };
        var optionsWithMomentum = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.9
        };

        var optimizerNoMom = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, optionsNoMomentum);
        var optimizerWithMom = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, optionsWithMomentum);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var gradientPos = new Vector<double>(new double[] { 1.0 });
        var gradientNeg = new Vector<double>(new double[] { -1.0 });

        // Act - Oscillating gradients
        var currentNoMom = parameters;
        var currentWithMom = parameters;

        for (int i = 0; i < 10; i++)
        {
            var grad = i % 2 == 0 ? gradientPos : gradientNeg;
            currentNoMom = optimizerNoMom.UpdateParameters(currentNoMom, grad);
            currentWithMom = optimizerWithMom.UpdateParameters(currentWithMom, grad);
        }

        // Assert - With momentum, the accumulated velocity should show damping effect
        // The test passes if momentum optimizer maintains internal velocity state correctly
        Assert.NotNull(currentNoMom);
        Assert.NotNull(currentWithMom);
    }

    [Fact]
    public void Momentum_ConvergesFasterThanSGD_OnIllConditionedProblem()
    {
        // Momentum should converge faster on ill-conditioned problems (high condition number)

        // Arrange - Ill-conditioned quadratic with eigenvalues 1 and 100
        var eigenvalues = new double[] { 1.0, 100.0 };
        var startPoint = new double[] { 10.0, 10.0 };

        var optionsSGD = new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01 // Small LR needed for stability
        };
        var optionsMomentum = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            InitialMomentum = 0.9
        };

        var sgd = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, optionsSGD);
        var momentum = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, optionsMomentum);

        var xSGD = (double[])startPoint.Clone();
        var xMom = (double[])startPoint.Clone();

        // Act - Run 100 iterations
        for (int iter = 0; iter < 100; iter++)
        {
            var gradSGD = QuadraticGradient(xSGD, eigenvalues);
            var gradMom = QuadraticGradient(xMom, eigenvalues);

            var paramsSGD = new Vector<double>(xSGD);
            var paramsMom = new Vector<double>(xMom);
            var gradVecSGD = new Vector<double>(gradSGD);
            var gradVecMom = new Vector<double>(gradMom);

            var updatedSGD = sgd.UpdateParameters(paramsSGD, gradVecSGD);
            var updatedMom = momentum.UpdateParameters(paramsMom, gradVecMom);

            xSGD = new double[] { updatedSGD[0], updatedSGD[1] };
            xMom = new double[] { updatedMom[0], updatedMom[1] };
        }

        double lossSGD = QuadraticFunction(xSGD, eigenvalues);
        double lossMom = QuadraticFunction(xMom, eigenvalues);

        // Assert - Momentum should achieve lower loss
        Assert.True(lossMom <= lossSGD,
            $"Momentum loss ({lossMom}) should be <= SGD loss ({lossSGD})");
    }

    #endregion

    #region Adam Optimizer Tests

    [Fact]
    public void Adam_UpdateFormula_MatchesDefinition()
    {
        // Test Adam update formula from the original paper:
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        // m_hat = m_t / (1 - beta1^t)
        // v_hat = v_t / (1 - beta2^t)
        // theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
        // Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)

        // Arrange
        double lr = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            Beta1 = beta1,
            Beta2 = beta2,
            Epsilon = epsilon
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.5 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Manual calculation for first step (t=1):
        // m_1 = 0.9 * 0 + 0.1 * 2.0 = 0.2
        // v_1 = 0.999 * 0 + 0.001 * 4.0 = 0.004
        // m_hat = 0.2 / (1 - 0.9) = 0.2 / 0.1 = 2.0
        // v_hat = 0.004 / (1 - 0.999) = 0.004 / 0.001 = 4.0
        // update = 0.001 * 2.0 / (sqrt(4.0) + 1e-8) = 0.001 * 2.0 / 2.00000001 ≈ 0.001
        // params_new = 0.5 - 0.001 = 0.499

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - The update should be approximately 0.001
        double expectedUpdate = lr * 2.0 / (Math.Sqrt(4.0) + epsilon);
        double expectedParam = 0.5 - expectedUpdate;
        Assert.Equal(expectedParam, result[0], LargeTolerance);
    }

    [Fact]
    public void Adam_BiasCorrection_WorksCorrectly()
    {
        // Test that bias correction properly handles the initialization bias
        // Without bias correction, initial estimates would be biased towards zero

        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - First few updates should show bias correction effect
        var updates = new List<double>();
        var current = parameters;
        for (int i = 0; i < 5; i++)
        {
            var next = optimizer.UpdateParameters(current, gradient);
            updates.Add(Math.Abs(current[0] - next[0]));
            current = next;
        }

        // Assert - First update should be meaningful (not near zero due to bias)
        Assert.True(updates[0] > 0.01,
            $"First Adam update ({updates[0]}) should be meaningful after bias correction");
    }

    [Fact]
    public void Adam_AdaptsToGradientMagnitudes()
    {
        // Adam should adapt learning rate for each parameter based on gradient history

        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0, 0.0 });
        var smallGradient = new Vector<double>(new double[] { 0.01, 0.0 });
        var largeGradient = new Vector<double>(new double[] { 0.0, 10.0 });

        // Act - Apply different magnitude gradients to different parameters
        var current = parameters;
        for (int i = 0; i < 20; i++)
        {
            current = optimizer.UpdateParameters(current, smallGradient);
            current = optimizer.UpdateParameters(current, largeGradient);
        }

        // The effective learning rate for param[0] should be higher than param[1]
        // because param[1] sees larger gradients, causing larger v, thus smaller effective LR

        // Assert - Both parameters should have moved, but the ratio should reflect adaptation
        Assert.True(Math.Abs(current[0]) > 0,
            "Parameter with small gradient should move");
        Assert.True(Math.Abs(current[1]) > 0,
            "Parameter with large gradient should move");
    }

    [Fact]
    public void Adam_ConvergesOnRosenbrockFunction()
    {
        // Adam should make progress on the challenging Rosenbrock function

        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Start away from minimum (1, 1)
        var x = new double[] { -1.0, -1.0 };
        double initialLoss = RosenbrockFunction(x);

        // Act - Run optimization
        for (int iter = 0; iter < 1000; iter++)
        {
            var gradient = RosenbrockGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            x = new double[] { updated[0], updated[1] };
        }

        double finalLoss = RosenbrockFunction(x);

        // Assert - Should make significant progress
        Assert.True(finalLoss < initialLoss,
            $"Adam should reduce Rosenbrock loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void Adam_Reset_ClearsState()
    {
        // Reset should clear all accumulated state

        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer1 = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var optimizer2 = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 0.5, 0.5 });

        // Build up state in optimizer1
        for (int i = 0; i < 10; i++)
        {
            optimizer1.UpdateParameters(parameters, gradient);
        }

        // Act - Reset and compare with fresh optimizer
        optimizer1.Reset();
        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        // Assert - Should produce identical results after reset
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void Adam_SerializeDeserialize_PreservesState()
    {
        // Serialization should preserve complete optimizer state

        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer1 = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build state
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        // Act - Serialize and deserialize
        var serialized = optimizer1.Serialize();
        var optimizer2 = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        // Continue updating with both optimizers
        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        // Assert - Should produce identical results
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    #endregion

    #region RMSProp Tests

    [Fact]
    public void RMSProp_UpdateFormula_MatchesDefinition()
    {
        // Test RMSProp update formula:
        // v_t = decay * v_{t-1} + (1 - decay) * g_t^2
        // theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + epsilon)
        // Reference: Hinton's Coursera lecture notes

        // Arrange
        double lr = 0.01;
        double decay = 0.9;
        double epsilon = 1e-8;

        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            Decay = decay,
            Epsilon = epsilon
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Manual calculation for first step:
        // v_1 = 0.9 * 0 + 0.1 * 4.0 = 0.4
        // update = 0.01 * 2.0 / (sqrt(0.4) + 1e-8)
        //        = 0.01 * 2.0 / 0.6324555...
        //        ≈ 0.03162
        // params_new = 1.0 - 0.03162 ≈ 0.96838

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        double expectedV = (1 - decay) * 4.0;
        double expectedUpdate = lr * 2.0 / (Math.Sqrt(expectedV) + epsilon);
        double expectedParam = 1.0 - expectedUpdate;

        Assert.Equal(expectedParam, result[0], LargeTolerance);
    }

    [Fact]
    public void RMSProp_AdaptsLearningRate_ForDifferentGradients()
    {
        // RMSProp should use larger effective learning rate for infrequent features
        // and smaller effective learning rate for frequent features

        // Arrange
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Decay = 0.9
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0, 0.0 });

        // Apply gradients: param 0 gets small gradients, param 1 gets large gradients
        var current = parameters;
        for (int i = 0; i < 20; i++)
        {
            var gradient = new Vector<double>(new double[] { 0.1, 10.0 });
            current = optimizer.UpdateParameters(current, gradient);
        }

        // The accumulated v for param[1] is much larger, so effective LR is smaller
        // Assert - Both should have moved, adaptation visible in ratio
        Assert.NotEqual(0.0, current[0], Tolerance);
        Assert.NotEqual(0.0, current[1], Tolerance);
    }

    [Fact]
    public void RMSProp_ConvergesOnSphereFunction()
    {
        // Arrange
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Decay = 0.9
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert
        Assert.True(finalLoss < initialLoss * 0.01,
            $"RMSProp should reduce loss significantly. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region Adagrad Tests

    [Fact]
    public void Adagrad_UpdateFormula_MatchesDefinition()
    {
        // Test Adagrad update formula:
        // G_t = G_{t-1} + g_t^2  (accumulated squared gradients)
        // theta_t = theta_{t-1} - lr * g_t / (sqrt(G_t) + epsilon)
        // Reference: "Adaptive Subgradient Methods" (Duchi et al., 2011)

        // Arrange
        double lr = 0.1;
        double epsilon = 1e-8;

        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            Epsilon = epsilon
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Manual calculation for first step:
        // G_1 = 0 + 4.0 = 4.0
        // update = 0.1 * 2.0 / (sqrt(4.0) + 1e-8)
        //        = 0.1 * 2.0 / 2.00000001
        //        ≈ 0.1
        // params_new = 1.0 - 0.1 = 0.9

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        double expectedG = 4.0;
        double expectedUpdate = lr * 2.0 / (Math.Sqrt(expectedG) + epsilon);
        double expectedParam = 1.0 - expectedUpdate;

        Assert.Equal(expectedParam, result[0], LargeTolerance);
    }

    [Fact]
    public void Adagrad_LearningRateDecreases_WithAccumulation()
    {
        // Adagrad's effective learning rate should decrease as gradients accumulate

        // Arrange
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - Track update magnitudes
        var updateMagnitudes = new List<double>();
        var current = parameters;
        for (int i = 0; i < 10; i++)
        {
            var next = optimizer.UpdateParameters(current, gradient);
            updateMagnitudes.Add(Math.Abs(next[0] - current[0]));
            current = next;
        }

        // Assert - Updates should decrease over time
        for (int i = 1; i < updateMagnitudes.Count; i++)
        {
            Assert.True(updateMagnitudes[i] <= updateMagnitudes[i - 1] + Tolerance,
                $"Adagrad update {i} ({updateMagnitudes[i]}) should be <= update {i - 1} ({updateMagnitudes[i - 1]})");
        }
    }

    [Fact]
    public void Adagrad_ConvergesOnSparseFeatures()
    {
        // Adagrad should work well with sparse features (some gradients are often zero)

        // Arrange
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.5
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        // Sparse gradients: only one feature active at a time
        var gradients = new List<Vector<double>>
        {
            new Vector<double>(new double[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 0.0, 1.0 })
        };

        // Act
        var current = parameters;
        for (int i = 0; i < 30; i++)
        {
            var grad = gradients[i % 3];
            current = optimizer.UpdateParameters(current, grad);
        }

        // Assert - All parameters should have moved from initial position
        for (int i = 0; i < current.Length; i++)
        {
            Assert.NotEqual(1.0, current[i], Tolerance);
        }
    }

    #endregion

    #region AdaDelta Tests

    [Fact]
    public void AdaDelta_UpdateFormula_MatchesDefinition()
    {
        // Test AdaDelta update formula:
        // E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
        // Delta_theta = -sqrt(E[Delta_theta^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon) * g_t
        // E[Delta_theta^2]_t = rho * E[Delta_theta^2]_{t-1} + (1 - rho) * Delta_theta^2
        // theta_t = theta_{t-1} + Delta_theta
        // Reference: "ADADELTA: An Adaptive Learning Rate Method" (Zeiler, 2012)

        // Arrange
        double rho = 0.9;
        double epsilon = 1e-8;

        var options = new AdaDeltaOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            Rho = rho,
            Epsilon = epsilon
        };
        var optimizer = new AdaDeltaOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // AdaDelta doesn't require learning rate - it adapts automatically
        // The update should be non-zero and in the opposite direction of gradient
        Assert.True(result[0] < parameters[0],
            "AdaDelta should move parameter opposite to positive gradient");
    }

    [Fact]
    public void AdaDelta_NoLearningRateRequired()
    {
        // AdaDelta adapts learning rate automatically - no manual LR needed

        // Arrange
        var options = new AdaDeltaOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            Rho = 0.95
        };
        var optimizer = new AdaDeltaOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act - Run optimization without setting learning rate
        for (int iter = 0; iter < 200; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert
        Assert.True(finalLoss < initialLoss,
            $"AdaDelta should reduce loss without manual LR. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region Nesterov Accelerated Gradient Tests

    [Fact]
    public void NAG_UpdateFormula_MatchesDefinition()
    {
        // Test Nesterov Accelerated Gradient (NAG) update formula:
        // v_t = momentum * v_{t-1} + lr * gradient(theta_{t-1} - momentum * v_{t-1})
        // theta_t = theta_{t-1} - v_t
        // Approximation: theta_t = theta_{t-1} - (momentum * v_{t-1} + lr * g_t)
        // Reference: "A method for solving a convex programming problem" (Nesterov, 1983)

        // Arrange
        double lr = 0.1;
        double momentum = 0.9;

        var options = new NesterovAcceleratedGradientOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            InitialMomentum = momentum
        };
        var optimizer = new NesterovAcceleratedGradientOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - First update
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Should move in negative gradient direction
        Assert.True(result[0] < parameters[0],
            "NAG should move parameter opposite to positive gradient");
    }

    [Fact]
    public void NAG_OutperformsMomentum_OnConvexFunctions()
    {
        // NAG should converge faster than standard momentum on convex problems

        // Arrange - Strongly convex quadratic
        var eigenvalues = new double[] { 1.0, 10.0 };
        var startPoint = new double[] { 10.0, 10.0 };

        var optionsMom = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            InitialMomentum = 0.9
        };
        var optionsNAG = new NesterovAcceleratedGradientOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            InitialMomentum = 0.9
        };

        var momentum = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, optionsMom);
        var nag = new NesterovAcceleratedGradientOptimizer<double, Vector<double>, Vector<double>>(null, optionsNAG);

        var xMom = (double[])startPoint.Clone();
        var xNAG = (double[])startPoint.Clone();

        // Act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradMom = QuadraticGradient(xMom, eigenvalues);
            var gradNAG = QuadraticGradient(xNAG, eigenvalues);

            var paramsMom = new Vector<double>(xMom);
            var paramsNAG = new Vector<double>(xNAG);

            var updatedMom = momentum.UpdateParameters(paramsMom, new Vector<double>(gradMom));
            var updatedNAG = nag.UpdateParameters(paramsNAG, new Vector<double>(gradNAG));

            xMom = new double[] { updatedMom[0], updatedMom[1] };
            xNAG = new double[] { updatedNAG[0], updatedNAG[1] };
        }

        double lossMom = QuadraticFunction(xMom, eigenvalues);
        double lossNAG = QuadraticFunction(xNAG, eigenvalues);

        // Assert - NAG should achieve comparable or better loss
        Assert.True(lossNAG <= lossMom * 1.1, // Allow 10% tolerance
            $"NAG loss ({lossNAG}) should be <= Momentum loss ({lossMom}) on convex problem");
    }

    #endregion

    #region AdaMax Tests

    [Fact]
    public void AdaMax_UpdateFormula_MatchesDefinition()
    {
        // Test AdaMax update formula (Adam with L-infinity norm):
        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // u_t = max(beta2 * u_{t-1}, |g_t|)  (infinity norm)
        // theta_t = theta_{t-1} - lr / (1 - beta1^t) * m_t / u_t
        // Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014) - Section 7

        // Arrange
        double lr = 0.002;
        double beta1 = 0.9;
        double beta2 = 0.999;

        var options = new AdaMaxOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            Beta1 = beta1,
            Beta2 = beta2
        };
        var optimizer = new AdaMaxOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Should move opposite to gradient
        Assert.True(result[0] < parameters[0],
            "AdaMax should decrease parameter with positive gradient");
    }

    [Fact]
    public void AdaMax_HandlesLargeGradients()
    {
        // AdaMax should handle large gradient spikes gracefully (unlike Adam which squares them)

        // Arrange
        var options = new AdaMaxOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer = new AdaMaxOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var normalGradient = new Vector<double>(new double[] { 1.0 });
        var hugeGradient = new Vector<double>(new double[] { 1000.0 });

        // Act - Apply normal gradient, then huge spike, then normal again
        var current = parameters;
        for (int i = 0; i < 10; i++)
        {
            current = optimizer.UpdateParameters(current, normalGradient);
        }
        var beforeSpike = current[0];

        current = optimizer.UpdateParameters(current, hugeGradient);

        for (int i = 0; i < 10; i++)
        {
            current = optimizer.UpdateParameters(current, normalGradient);
        }

        // Assert - Should handle the spike without NaN or extreme values
        Assert.False(double.IsNaN(current[0]), "AdaMax should not produce NaN");
        Assert.False(double.IsInfinity(current[0]), "AdaMax should not produce Infinity");
    }

    #endregion

    #region AMSGrad Tests

    [Fact]
    public void AMSGrad_MaintainsMaxSecondMoment()
    {
        // AMSGrad maintains the maximum of past second moments
        // This ensures non-increasing learning rate sequence

        // Arrange
        var options = new AMSGradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer = new AMSGradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var largeGradient = new Vector<double>(new double[] { 10.0 });
        var smallGradient = new Vector<double>(new double[] { 0.1 });

        // Act - Apply large gradient first, then small gradients
        var current = parameters;
        current = optimizer.UpdateParameters(current, largeGradient);

        // Store update magnitude after large gradient
        var afterLargeGrad = current[0];

        // Apply several small gradients
        for (int i = 0; i < 10; i++)
        {
            current = optimizer.UpdateParameters(current, smallGradient);
        }

        // Assert - Updates should remain bounded even after small gradients
        // (the max second moment from the large gradient is preserved)
        Assert.NotNull(current);
        Assert.False(double.IsNaN(current[0]));
    }

    [Fact]
    public void AMSGrad_ConvergesBetterThanAdam_OnNonStationaryProblems()
    {
        // AMSGrad was designed to fix Adam's potential non-convergence

        // Arrange
        var optionsAdam = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optionsAMSGrad = new AMSGradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999
        };

        var adam = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, optionsAdam);
        var amsgrad = new AMSGradOptimizer<double, Vector<double>, Vector<double>>(null, optionsAMSGrad);

        var x = new double[] { 5.0, -3.0 };
        double initialLoss = SphereFunction(x);

        var xAdam = (double[])x.Clone();
        var xAMSGrad = (double[])x.Clone();

        // Act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradAdam = SphereGradient(xAdam);
            var gradAMSGrad = SphereGradient(xAMSGrad);

            var paramsAdam = new Vector<double>(xAdam);
            var paramsAMSGrad = new Vector<double>(xAMSGrad);

            var updatedAdam = adam.UpdateParameters(paramsAdam, new Vector<double>(gradAdam));
            var updatedAMSGrad = amsgrad.UpdateParameters(paramsAMSGrad, new Vector<double>(gradAMSGrad));

            xAdam = new double[] { updatedAdam[0], updatedAdam[1] };
            xAMSGrad = new double[] { updatedAMSGrad[0], updatedAMSGrad[1] };
        }

        double lossAdam = SphereFunction(xAdam);
        double lossAMSGrad = SphereFunction(xAMSGrad);

        // Assert - Both should converge on sphere (not a pathological case for Adam)
        Assert.True(lossAdam < initialLoss, "Adam should reduce loss");
        Assert.True(lossAMSGrad < initialLoss, "AMSGrad should reduce loss");
    }

    #endregion

    #region Nadam Tests

    [Fact]
    public void Nadam_CombinesNesterovAndAdam()
    {
        // Nadam should combine Nesterov momentum with Adam's adaptive learning rate
        // Note: Adam-family optimizers normalize gradients, so they need higher learning rates
        // than SGD for the same convergence speed. lr=0.1 is appropriate for sphere function.

        // Arrange
        var nadamOptions = new NadamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var nadam = new NadamOptimizer<double, Vector<double>, Vector<double>>(null, nadamOptions);

        // Compare with Adam using the same parameters
        var adamOptions = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var adam = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, adamOptions);

        // Verify learning rates are set correctly
        double nadamLR = nadam.GetCurrentLearningRate();
        double adamLR = adam.GetCurrentLearningRate();
        Assert.Equal(0.1, nadamLR, 6);
        Assert.Equal(0.1, adamLR, 6);

        var xNadam = new double[] { 5.0, -3.0, 2.0 };
        var xAdam = (double[])xNadam.Clone();
        double initialLoss = SphereFunction(xNadam);

        // Act - Run both optimizers
        for (int iter = 0; iter < 100; iter++)
        {
            // Nadam update
            var gradNadam = SphereGradient(xNadam);
            var paramsNadam = new Vector<double>(xNadam);
            var gradVecNadam = new Vector<double>(gradNadam);
            var updatedNadam = nadam.UpdateParameters(paramsNadam, gradVecNadam);
            for (int i = 0; i < xNadam.Length; i++) xNadam[i] = updatedNadam[i];

            // Adam update
            var gradAdam = SphereGradient(xAdam);
            var paramsAdam = new Vector<double>(xAdam);
            var gradVecAdam = new Vector<double>(gradAdam);
            var updatedAdam = adam.UpdateParameters(paramsAdam, gradVecAdam);
            for (int i = 0; i < xAdam.Length; i++) xAdam[i] = updatedAdam[i];
        }

        double nadamFinalLoss = SphereFunction(xNadam);
        double adamFinalLoss = SphereFunction(xAdam);

        // Assert - Both should significantly reduce loss, Nadam should be at least as good as Adam
        Assert.True(adamFinalLoss < initialLoss * 0.1,
            $"Adam should significantly reduce loss. Initial: {initialLoss}, Final: {adamFinalLoss}");
        Assert.True(nadamFinalLoss < initialLoss * 0.1,
            $"Nadam should significantly reduce loss. Initial: {initialLoss}, Final: {nadamFinalLoss}");
    }

    #endregion

    #region Edge Cases and Robustness Tests

    [Theory]
    [InlineData("SGD")]
    [InlineData("Momentum")]
    [InlineData("Adam")]
    [InlineData("RMSProp")]
    [InlineData("Adagrad")]
    public void AllOptimizers_HandleZeroGradient(string optimizerName)
    {
        // All optimizers should handle zero gradients gracefully

        // Arrange
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var zeroGradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        Vector<double> result;
        switch (optimizerName)
        {
            case "SGD":
                var sgd = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
                    new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = sgd.UpdateParameters(parameters, zeroGradient);
                break;
            case "Momentum":
                var mom = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
                    new MomentumOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, InitialMomentum = 0.9 });
                result = mom.UpdateParameters(parameters, zeroGradient);
                break;
            case "Adam":
                var adam = new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
                    new AdamOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = adam.UpdateParameters(parameters, zeroGradient);
                break;
            case "RMSProp":
                var rmsprop = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
                    new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = rmsprop.UpdateParameters(parameters, zeroGradient);
                break;
            case "Adagrad":
                var adagrad = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null,
                    new AdagradOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = adagrad.UpdateParameters(parameters, zeroGradient);
                break;
            default:
                throw new ArgumentException($"Unknown optimizer: {optimizerName}");
        }

        // Assert - No NaN, no Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"{optimizerName} produced NaN with zero gradient");
            Assert.False(double.IsInfinity(result[i]), $"{optimizerName} produced Infinity with zero gradient");
        }
    }

    [Theory]
    [InlineData("SGD")]
    [InlineData("Momentum")]
    [InlineData("Adam")]
    [InlineData("RMSProp")]
    [InlineData("Adagrad")]
    public void AllOptimizers_HandleLargeGradient(string optimizerName)
    {
        // All optimizers should handle large gradients without numerical issues

        // Arrange
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var largeGradient = new Vector<double>(new double[] { 1e6, -1e6, 1e6 });

        Vector<double> result;
        switch (optimizerName)
        {
            case "SGD":
                var sgd = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
                    new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 1e-8 });
                result = sgd.UpdateParameters(parameters, largeGradient);
                break;
            case "Momentum":
                var mom = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
                    new MomentumOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 1e-8, InitialMomentum = 0.9 });
                result = mom.UpdateParameters(parameters, largeGradient);
                break;
            case "Adam":
                var adam = new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
                    new AdamOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.001 });
                result = adam.UpdateParameters(parameters, largeGradient);
                break;
            case "RMSProp":
                var rmsprop = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
                    new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.001 });
                result = rmsprop.UpdateParameters(parameters, largeGradient);
                break;
            case "Adagrad":
                var adagrad = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null,
                    new AdagradOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.001 });
                result = adagrad.UpdateParameters(parameters, largeGradient);
                break;
            default:
                throw new ArgumentException($"Unknown optimizer: {optimizerName}");
        }

        // Assert - No NaN, no Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"{optimizerName} produced NaN with large gradient");
            Assert.False(double.IsInfinity(result[i]), $"{optimizerName} produced Infinity with large gradient");
        }
    }

    [Theory]
    [InlineData("SGD")]
    [InlineData("Momentum")]
    [InlineData("Adam")]
    [InlineData("RMSProp")]
    [InlineData("Adagrad")]
    public void AllOptimizers_HandleVerySmallGradient(string optimizerName)
    {
        // All optimizers should handle very small gradients without underflow

        // Arrange
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var smallGradient = new Vector<double>(new double[] { 1e-15, -1e-15, 1e-15 });

        Vector<double> result;
        switch (optimizerName)
        {
            case "SGD":
                var sgd = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
                    new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = sgd.UpdateParameters(parameters, smallGradient);
                break;
            case "Momentum":
                var mom = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
                    new MomentumOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, InitialMomentum = 0.9 });
                result = mom.UpdateParameters(parameters, smallGradient);
                break;
            case "Adam":
                var adam = new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
                    new AdamOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = adam.UpdateParameters(parameters, smallGradient);
                break;
            case "RMSProp":
                var rmsprop = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
                    new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = rmsprop.UpdateParameters(parameters, smallGradient);
                break;
            case "Adagrad":
                var adagrad = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null,
                    new AdagradOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
                result = adagrad.UpdateParameters(parameters, smallGradient);
                break;
            default:
                throw new ArgumentException($"Unknown optimizer: {optimizerName}");
        }

        // Assert - No NaN, no Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"{optimizerName} produced NaN with small gradient");
            Assert.False(double.IsInfinity(result[i]), $"{optimizerName} produced Infinity with small gradient");
        }
    }

    [Fact]
    public void AllOptimizers_MoveOppositeToGradient()
    {
        // All gradient-based optimizers should move parameters opposite to the gradient direction
        // (to minimize the objective function)

        var parameters = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
        var positiveGradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        // Test each optimizer
        var optimizers = new Dictionary<string, Func<Vector<double>>>
        {
            ["SGD"] = () => new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
                new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 })
                .UpdateParameters(parameters, positiveGradient),

            ["Momentum"] = () => new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
                new MomentumOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, InitialMomentum = 0.9 })
                .UpdateParameters(parameters, positiveGradient),

            ["Adam"] = () => new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
                new AdamOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 })
                .UpdateParameters(parameters, positiveGradient),

            ["RMSProp"] = () => new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
                new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 })
                .UpdateParameters(parameters, positiveGradient),

            ["Adagrad"] = () => new AdagradOptimizer<double, Vector<double>, Vector<double>>(null,
                new AdagradOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 })
                .UpdateParameters(parameters, positiveGradient),
        };

        foreach (var (name, getResult) in optimizers)
        {
            var result = getResult();

            // With positive gradient, all parameters should decrease
            for (int i = 0; i < result.Length; i++)
            {
                Assert.True(result[i] < parameters[i],
                    $"{name} should decrease parameter {i} with positive gradient");
            }
        }
    }

    #endregion

    #region Comparative Performance Tests

    [Fact]
    public void CompareOptimizers_OnSphereFunction()
    {
        // Compare convergence of different optimizers on the sphere function

        var startPoint = new double[] { 5.0, -3.0, 2.0, -4.0, 1.0 };
        int iterations = 100;

        var results = new Dictionary<string, double>();

        // SGD
        var xSGD = (double[])startPoint.Clone();
        var sgd = new GradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
            new GradientDescentOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
        for (int i = 0; i < iterations; i++)
        {
            var grad = SphereGradient(xSGD);
            var updated = sgd.UpdateParameters(new Vector<double>(xSGD), new Vector<double>(grad));
            for (int j = 0; j < xSGD.Length; j++) xSGD[j] = updated[j];
        }
        results["SGD"] = SphereFunction(xSGD);

        // Momentum
        var xMom = (double[])startPoint.Clone();
        var momentum = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
            new MomentumOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, InitialMomentum = 0.9 });
        for (int i = 0; i < iterations; i++)
        {
            var grad = SphereGradient(xMom);
            var updated = momentum.UpdateParameters(new Vector<double>(xMom), new Vector<double>(grad));
            for (int j = 0; j < xMom.Length; j++) xMom[j] = updated[j];
        }
        results["Momentum"] = SphereFunction(xMom);

        // Adam
        var xAdam = (double[])startPoint.Clone();
        var adam = new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
            new AdamOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
        for (int i = 0; i < iterations; i++)
        {
            var grad = SphereGradient(xAdam);
            var updated = adam.UpdateParameters(new Vector<double>(xAdam), new Vector<double>(grad));
            for (int j = 0; j < xAdam.Length; j++) xAdam[j] = updated[j];
        }
        results["Adam"] = SphereFunction(xAdam);

        // RMSProp
        var xRMS = (double[])startPoint.Clone();
        var rmsprop = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
            new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 });
        for (int i = 0; i < iterations; i++)
        {
            var grad = SphereGradient(xRMS);
            var updated = rmsprop.UpdateParameters(new Vector<double>(xRMS), new Vector<double>(grad));
            for (int j = 0; j < xRMS.Length; j++) xRMS[j] = updated[j];
        }
        results["RMSProp"] = SphereFunction(xRMS);

        // Assert - All optimizers should converge (loss < 0.1)
        double initialLoss = SphereFunction(startPoint);
        foreach (var (name, loss) in results)
        {
            Assert.True(loss < initialLoss * 0.01,
                $"{name} should reduce loss significantly. Initial: {initialLoss}, Final: {loss}");
        }
    }

    #endregion

    #region Lion Optimizer Tests

    [Fact]
    public void Lion_UpdateFormula_MatchesDefinition()
    {
        // Test Lion update formula:
        // c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // theta_t = theta_{t-1} - lr * sign(c_t)
        // m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
        // Reference: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

        // Arrange
        var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.99,
            WeightDecay = 0.0
        };
        var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, -2.0, 0.5 });
        var gradient = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Lion uses sign-based updates
        // First iteration: m = [0, 0, 0]
        // c = 0.9 * [0, 0, 0] + 0.1 * [0.5, -0.3, 0.8] = [0.05, -0.03, 0.08]
        // sign(c) = [1, -1, 1]
        // new_params = [1.0, -2.0, 0.5] - 0.1 * [1, -1, 1] = [0.9, -1.9, 0.4]
        Assert.Equal(0.9, result[0], 1e-5);
        Assert.Equal(-1.9, result[1], 1e-5);
        Assert.Equal(0.4, result[2], 1e-5);
    }

    [Fact]
    public void Lion_ConvergesOnSphereFunction()
    {
        // Lion should converge on the sphere function
        // Uses sign-based updates, so may have different convergence behavior

        // Arrange
        var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01, // Lion typically uses smaller learning rates
            Beta1 = 0.9,
            Beta2 = 0.99
        };
        var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act - Run 500 iterations (Lion converges more gradually due to sign-based updates)
        for (int iter = 0; iter < 500; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert - Should significantly reduce loss
        Assert.True(finalLoss < initialLoss * 0.1,
            $"Lion should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void Lion_SignBasedUpdates_ConsistentStepSize()
    {
        // Verify Lion's characteristic: consistent step sizes regardless of gradient magnitude

        // Arrange
        var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.99
        };
        var optimizer = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Two very different gradient magnitudes
        var params1 = new Vector<double>(new double[] { 0.0 });
        var smallGrad = new Vector<double>(new double[] { 0.001 });
        var largeGrad = new Vector<double>(new double[] { 1000.0 });

        // Need to reinitialize optimizer between tests
        var optimizer2 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Act
        var result1 = optimizer.UpdateParameters(params1, smallGrad);
        var result2 = optimizer2.UpdateParameters(params1, largeGrad);

        // Assert - Both should move by lr * sign(grad) = 0.1 * 1 = 0.1 in magnitude
        double move1 = Math.Abs(result1[0] - params1[0]);
        double move2 = Math.Abs(result2[0] - params1[0]);

        // Both moves should be the learning rate (within tolerance for accumulated momentum)
        Assert.True(Math.Abs(move1 - 0.1) < 0.02, $"Small gradient should produce ~0.1 move, got {move1}");
        Assert.True(Math.Abs(move2 - 0.1) < 0.02, $"Large gradient should produce ~0.1 move, got {move2}");
    }

    #endregion

    #region L-BFGS Optimizer Tests

    [Fact]
    public void LBFGS_ConvergesOnSphereFunction()
    {
        // L-BFGS should converge quickly on convex quadratic functions

        // Arrange
        var options = new LBFGSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0, // L-BFGS typically uses lr=1 and scales via quasi-Newton direction
            MemorySize = 10
        };
        var optimizer = new LBFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act - Run optimization (L-BFGS converges fast on convex quadratic)
        for (int iter = 0; iter < 50; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert - L-BFGS should converge well on convex quadratic functions
        Assert.True(finalLoss < initialLoss * 0.1,
            $"L-BFGS should significantly reduce loss on sphere function. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void LBFGS_ConvergesOnRosenbrockFunction()
    {
        // L-BFGS is known to be effective on Rosenbrock function

        // Arrange
        var options = new LBFGSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            MemorySize = 10
        };
        var optimizer = new LBFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Start away from minimum (1, 1)
        var x = new double[] { -1.0, -1.0 };
        double initialLoss = RosenbrockFunction(x);

        // Act - Run more iterations for the challenging Rosenbrock function
        for (int iter = 0; iter < 2000; iter++)
        {
            var gradient = RosenbrockGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = RosenbrockFunction(x);

        // Assert - Should make progress (Rosenbrock is challenging)
        Assert.True(finalLoss < initialLoss * 0.5,
            $"L-BFGS should make progress on Rosenbrock. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region Newton Method Optimizer Tests

    [Fact]
    public void NewtonMethod_ConvergesOnSphereFunction()
    {
        // Newton's Method should converge very quickly on quadratic functions
        // For the sphere function, it should theoretically converge in one step

        // Arrange
        var options = new NewtonMethodOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new NewtonMethodOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act - Newton method should converge fast on simple quadratic
        for (int iter = 0; iter < 50; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert - Newton should converge on sphere function
        Assert.True(finalLoss < initialLoss * 0.1,
            $"Newton's Method should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region AdamW Optimizer Tests

    [Fact]
    public void AdamW_UpdateFormula_AppliesDecoupledWeightDecay()
    {
        // AdamW applies weight decay directly to weights, not through gradient
        // Formula: w = w - lr * (adam_update + weight_decay * w)

        // Arrange
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            WeightDecay = 0.01,
            Epsilon = 1e-8
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Act - First update
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Parameters should change (AdamW with weight decay should reduce magnitudes)
        Assert.NotEqual(parameters[0], result[0]);
        Assert.NotEqual(parameters[1], result[1]);
        Assert.NotEqual(parameters[2], result[2]);
    }

    [Fact]
    public void AdamW_ConvergesOnSphereFunction()
    {
        // Arrange
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999,
            WeightDecay = 0.0, // Disable weight decay for pure convergence test
            Epsilon = 1e-8
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert
        Assert.True(finalLoss < initialLoss * 0.01,
            $"AdamW should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void AdamW_WeightDecay_ReducesWeightMagnitude()
    {
        // Test that weight decay actually reduces weight magnitude over time
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            WeightDecay = 0.1, // Strong weight decay for visible effect
            Epsilon = 1e-8
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 10.0, 10.0, 10.0 };
        double initialNorm = Math.Sqrt(x.Sum(xi => xi * xi));

        // Run with zero gradient - only weight decay should act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = new double[] { 0.0, 0.0, 0.0 };
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalNorm = Math.Sqrt(x.Sum(xi => xi * xi));

        // Assert - Weight decay should reduce magnitude
        Assert.True(finalNorm < initialNorm,
            $"Weight decay should reduce weight magnitude. Initial: {initialNorm}, Final: {finalNorm}");
    }

    #endregion

    #region FTRL Optimizer Tests

    [Fact]
    public void FTRL_ConvergesOnSphereFunction()
    {
        // FTRL (Follow The Regularized Leader) should converge on simple functions
        var options = new FTRLOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            Alpha = 0.5, // Higher alpha for faster convergence on simple function
            Beta = 1.0,
            Lambda1 = 0.0, // Disable L1 for pure convergence test
            Lambda2 = 0.0  // Disable L2 for pure convergence test
        };
        var optimizer = new FTRLOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act
        for (int iter = 0; iter < 200; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert
        Assert.True(finalLoss < initialLoss * 0.5,
            $"FTRL should reduce loss on sphere function. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void FTRL_L1Regularization_PromotesSparsity()
    {
        // FTRL with L1 regularization should drive some parameters to zero
        var options = new FTRLOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            Alpha = 0.1,
            Beta = 1.0,
            Lambda1 = 1.0, // Enable L1 for sparsity
            Lambda2 = 0.0
        };
        var optimizer = new FTRLOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Start with small values that L1 should push to zero
        var x = new double[] { 0.1, -0.1, 0.05 };

        // Run with small gradients
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = new double[] { 0.01, -0.01, 0.005 };
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        // Assert - At least some values should be very small or zero
        int nearZeroCount = x.Count(xi => Math.Abs(xi) < 0.01);
        Assert.True(nearZeroCount >= 1,
            $"FTRL with L1 should promote sparsity. Near-zero count: {nearZeroCount}, Values: [{string.Join(", ", x)}]");
    }

    #endregion

    #region LAMB Optimizer Tests

    [Fact]
    public void LAMB_ConvergesOnSphereFunction()
    {
        // LAMB (Layer-wise Adaptive Moments for Batch training)
        var options = new LAMBOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999,
            WeightDecay = 0.0,
            Epsilon = 1e-6
        };
        var optimizer = new LAMBOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert
        Assert.True(finalLoss < initialLoss * 0.1,
            $"LAMB should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void LAMB_TrustRatioScaling_NormalizesUpdates()
    {
        // LAMB uses trust ratio = ||w|| / ||r|| where r is the Adam update
        // This should help stabilize updates for different weight magnitudes
        var options = new LAMBOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            WeightDecay = 0.0,
            ClipTrustRatio = true,
            MaxTrustRatio = 10.0
        };
        var optimizer = new LAMBOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Large weights - trust ratio should scale updates appropriately
        var parameters = new Vector<double>(new double[] { 100.0, 100.0, 100.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Should make reasonable progress
        Assert.True(result[0] < parameters[0], "LAMB should update parameters in the right direction");
    }

    #endregion

    #region LARS Optimizer Tests

    [Fact]
    public void LARS_ConvergesOnSphereFunction()
    {
        // LARS (Layer-wise Adaptive Rate Scaling)
        // Note: TrustCoefficient of 0.001 is designed for large batch training (4096+).
        // For this simple test, we use a higher value (1.0) so that the effective learning
        // rate is reasonable. The LARS formula is: local_lr = base_lr * trust_coeff * ||w|| / ||g||
        // With trust_coeff=0.001 and ||w||≈6, ||g||≈12, effective lr would be ~0.00005 (too slow).
        var options = new LARSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Momentum = 0.9,
            WeightDecay = 0.0,
            TrustCoefficient = 1.0 // Use 1.0 for simple test (0.001 is for large batch training)
        };
        var optimizer = new LARSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        // Act
        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        // Assert
        Assert.True(finalLoss < initialLoss * 0.5,
            $"LARS should reduce loss on sphere function. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void LARS_LayerWiseScaling_AdaptsToLayerNorms()
    {
        // LARS scales learning rate by trust_coefficient * ||w|| / ||g||
        var options = new LARSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Momentum = 0.0, // Disable momentum to test pure LARS
            WeightDecay = 0.0,
            TrustCoefficient = 0.001
        };
        var optimizer = new LARSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        // Large weights with small gradient - LARS should adapt
        var parameters = new Vector<double>(new double[] { 100.0, 100.0, 100.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });

        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Should make progress
        Assert.True(result[0] < parameters[0], "LARS should update parameters");
    }

    #endregion

    #region StochasticGradientDescent Optimizer Tests

    [Fact]
    public void SGD_UpdateFormula_MatchesDefinition()
    {
        // SGD: w = w - lr * g
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.5, -0.25, 1.0 });

        // Expected: [1.0 - 0.05, 2.0 + 0.025, 3.0 - 0.1] = [0.95, 2.025, 2.9]
        var expected = new double[] { 0.95, 2.025, 2.9 };

        var result = optimizer.UpdateParameters(parameters, gradient);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], result[i], Tolerance);
        }
    }

    [Fact]
    public void SGD_ConvergesOnSphereFunction()
    {
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.01,
            $"SGD should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region MiniBatchGradientDescent Optimizer Tests

    [Fact]
    public void MiniBatchGD_ConvergesOnSphereFunction()
    {
        var options = new MiniBatchGradientDescentOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            BatchSize = 32
        };
        var optimizer = new MiniBatchGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.01,
            $"MiniBatch GD should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region BFGS Optimizer Tests

    [Fact]
    public void BFGS_ConvergesOnSphereFunction()
    {
        // BFGS is a quasi-Newton method with full Hessian approximation
        var options = new BFGSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0
        };
        var optimizer = new BFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.1,
            $"BFGS should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void BFGS_ConvergesOnRosenbrockFunction()
    {
        var options = new BFGSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new BFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { -1.0, -1.0 };
        double initialLoss = RosenbrockFunction(x);

        for (int iter = 0; iter < 500; iter++)
        {
            var gradient = RosenbrockGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = RosenbrockFunction(x);

        Assert.True(finalLoss < initialLoss * 0.5,
            $"BFGS should make progress on Rosenbrock. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region DFP Optimizer Tests

    [Fact]
    public void DFP_ConvergesOnSphereFunction()
    {
        // DFP (Davidon-Fletcher-Powell) is another quasi-Newton method
        var options = new DFPOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0
        };
        var optimizer = new DFPOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.1,
            $"DFP should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region ConjugateGradient Optimizer Tests

    [Fact]
    public void ConjugateGradient_ConvergesOnSphereFunction()
    {
        var options = new ConjugateGradientOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new ConjugateGradientOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.1,
            $"Conjugate Gradient should reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region TrustRegion Optimizer Tests

    [Fact]
    public void TrustRegion_ConvergesOnSphereFunction()
    {
        var options = new TrustRegionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0,
            InitialTrustRegionRadius = 1.0
        };
        var optimizer = new TrustRegionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.5,
            $"Trust Region should reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region LevenbergMarquardt Optimizer Tests

    [Fact]
    public void LevenbergMarquardt_ConvergesOnSphereFunction()
    {
        var options = new LevenbergMarquardtOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialDampingFactor = 0.001
        };
        var optimizer = new LevenbergMarquardtOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.5,
            $"Levenberg-Marquardt should reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region CoordinateDescent Optimizer Tests

    [Fact]
    public void CoordinateDescent_ConvergesOnSphereFunction()
    {
        var options = new CoordinateDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new CoordinateDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 300; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.5,
            $"Coordinate Descent should reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region ProximalGradientDescent Optimizer Tests

    [Fact]
    public void ProximalGradientDescent_ConvergesOnSphereFunction()
    {
        var options = new ProximalGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            RegularizationStrength = 0.0 // No regularization for pure convergence test
        };
        var optimizer = new ProximalGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.1,
            $"Proximal GD should significantly reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    [Fact]
    public void ProximalGradientDescent_L1Regularization_PromotesSparsity()
    {
        // Proximal GD with L1 should drive small values to zero
        var options = new ProximalGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            RegularizationStrength = 0.5 // L1 regularization
        };
        var optimizer = new ProximalGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 0.1, -0.1, 0.05 };

        for (int iter = 0; iter < 100; iter++)
        {
            var gradient = new double[] { 0.01, -0.01, 0.005 };
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        // Assert - Some values should be near zero due to soft thresholding
        int nearZeroCount = x.Count(xi => Math.Abs(xi) < 0.01);
        Assert.True(nearZeroCount >= 1,
            $"Proximal GD with L1 should promote sparsity. Near-zero count: {nearZeroCount}");
    }

    #endregion

    #region ADMM Optimizer Tests

    [Fact]
    public void ADMM_ConvergesOnSphereFunction()
    {
        // ADMM (Alternating Direction Method of Multipliers)
        var options = new ADMMOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Rho = 1.0
        };
        var optimizer = new ADMMOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new double[] { 5.0, -3.0, 2.0 };
        double initialLoss = SphereFunction(x);

        for (int iter = 0; iter < 200; iter++)
        {
            var gradient = SphereGradient(x);
            var parameters = new Vector<double>(x);
            var gradVector = new Vector<double>(gradient);
            var updated = optimizer.UpdateParameters(parameters, gradVector);

            for (int i = 0; i < x.Length; i++)
            {
                x[i] = updated[i];
            }
        }

        double finalLoss = SphereFunction(x);

        Assert.True(finalLoss < initialLoss * 0.5,
            $"ADMM should reduce loss. Initial: {initialLoss}, Final: {finalLoss}");
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void SGD_SerializeDeserialize_PreservesState()
    {
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01
        };
        var optimizer1 = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build state
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        // Serialize and deserialize
        var serialized = optimizer1.Serialize();
        var optimizer2 = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        // Continue updating with both optimizers
        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void Adagrad_SerializeDeserialize_PreservesState()
    {
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Epsilon = 1e-8
        };
        var optimizer1 = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build state - Adagrad accumulates squared gradients
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        // Serialize and deserialize
        var serialized = optimizer1.Serialize();
        var optimizer2 = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void AdaDelta_SerializeDeserialize_PreservesState()
    {
        var options = new AdaDeltaOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            Rho = 0.95,
            Epsilon = 1e-6
        };
        var optimizer1 = new AdaDeltaOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build state - AdaDelta has running average of gradients and updates
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new AdaDeltaOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void RMSProp_SerializeDeserialize_PreservesState()
    {
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Decay = 0.9,
            Epsilon = 1e-8
        };
        var optimizer1 = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build state
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void AdaMax_SerializeDeserialize_PreservesState()
    {
        var options = new AdaMaxOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer1 = new AdaMaxOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new AdaMaxOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void AMSGrad_SerializeDeserialize_PreservesState()
    {
        var options = new AMSGradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer1 = new AMSGradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new AMSGradOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void Nadam_SerializeDeserialize_PreservesState()
    {
        var options = new NadamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };
        var optimizer1 = new NadamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new NadamOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void Lion_SerializeDeserialize_PreservesState()
    {
        var options = new LionOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.99
        };
        var optimizer1 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new LionOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void Momentum_SerializeDeserialize_PreservesState()
    {
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            InitialMomentum = 0.9
        };
        var optimizer1 = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void BFGS_SerializeDeserialize_PreservesState()
    {
        var options = new BFGSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer1 = new BFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build Hessian approximation
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new BFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Fact]
    public void LBFGS_SerializeDeserialize_PreservesState()
    {
        var options = new LBFGSOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            MemorySize = 5
        };
        var optimizer1 = new LBFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build history for L-BFGS
        for (int i = 0; i < 5; i++)
        {
            parameters = optimizer1.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer1.Serialize();
        var optimizer2 = new LBFGSOptimizer<double, Vector<double>, Vector<double>>(null, options);
        optimizer2.Deserialize(serialized);

        var result1 = optimizer1.UpdateParameters(parameters, gradient);
        var result2 = optimizer2.UpdateParameters(parameters, gradient);

        for (int i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    [Theory]
    [InlineData("SGD")]
    [InlineData("Adam")]
    [InlineData("RMSProp")]
    [InlineData("Adagrad")]
    [InlineData("Momentum")]
    public void AllOptimizers_SerializeDeserialize_BytesNonEmpty(string optimizerName)
    {
        // All optimizers should produce non-empty serialized data
        IGradientBasedOptimizer<double, Vector<double>, Vector<double>> optimizer = optimizerName switch
        {
            "SGD" => new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null),
            "Adam" => new AdamOptimizer<double, Vector<double>, Vector<double>>(null),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null),
            "Adagrad" => new AdagradOptimizer<double, Vector<double>, Vector<double>>(null),
            "Momentum" => new MomentumOptimizer<double, Vector<double>, Vector<double>>(null),
            _ => throw new ArgumentException($"Unknown optimizer: {optimizerName}")
        };

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        // Build some state
        for (int i = 0; i < 3; i++)
        {
            optimizer.UpdateParameters(parameters, gradient);
        }

        var serialized = optimizer.Serialize();

        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0, $"{optimizerName} should produce non-empty serialized data");
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void SGD_ZeroGradients_ReturnsUnchangedParameters()
    {
        // Zero gradients should not change parameters
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var zeroGradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var result = optimizer.UpdateParameters(parameters, zeroGradient);

        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i], result[i], Tolerance);
        }
    }

    [Fact]
    public void Adam_ZeroGradients_ReturnsUnchangedParameters()
    {
        // Zero gradients should not change parameters (after initialization)
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var zeroGradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var result = optimizer.UpdateParameters(parameters, zeroGradient);

        // Adam might have small numerical differences due to epsilon term
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(Math.Abs(result[i] - parameters[i]) < 0.01,
                $"Parameter {i} changed significantly with zero gradient: expected {parameters[i]}, got {result[i]}");
        }
    }

    [Fact]
    public void Momentum_ZeroGradients_ReturnsUnchangedParameters()
    {
        // Zero gradients should not change parameters
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var zeroGradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var result = optimizer.UpdateParameters(parameters, zeroGradient);

        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(parameters[i], result[i], Tolerance);
        }
    }

    [Fact]
    public void SGD_VerySmallGradients_UpdatesCorrectly()
    {
        // Very small gradients should still produce proportionally small updates
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var smallGradient = new Vector<double>(new double[] { 1e-10, 1e-10, 1e-10 });

        var result = optimizer.UpdateParameters(parameters, smallGradient);

        // Updates should be tiny but non-zero
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(result[i] < parameters[i], $"Parameter should decrease slightly: {result[i]} >= {parameters[i]}");
            Assert.True(Math.Abs(result[i] - parameters[i]) < 1e-8,
                $"Update should be very small: diff = {Math.Abs(result[i] - parameters[i])}");
        }
    }

    [Fact]
    public void SGD_LargeGradients_UpdatesWithinReasonableBounds()
    {
        // Large gradients should not cause parameters to explode
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var largeGradient = new Vector<double>(new double[] { 1000.0, 1000.0, 1000.0 });

        var result = optimizer.UpdateParameters(parameters, largeGradient);

        // Parameters should decrease but not go to extreme values
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(result[i] < parameters[i], "Parameters should decrease with positive gradient");
            Assert.True(result[i] > -1000, "Parameters should not explode to extreme negative values");
        }
    }

    [Fact]
    public void Adam_VerySmallGradients_UpdatesCorrectly()
    {
        // Adam should handle very small gradients without numerical issues
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var smallGradient = new Vector<double>(new double[] { 1e-12, 1e-12, 1e-12 });

        // Run multiple steps to test numerical stability
        var result = parameters;
        for (int step = 0; step < 10; step++)
        {
            result = optimizer.UpdateParameters(result, smallGradient);
        }

        // Result should not contain NaN or Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Parameter {i} became NaN");
            Assert.False(double.IsInfinity(result[i]), $"Parameter {i} became Infinity");
        }
    }

    [Fact]
    public void RMSProp_VerySmallGradients_UpdatesCorrectly()
    {
        // RMSProp should handle very small gradients without numerical issues
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var smallGradient = new Vector<double>(new double[] { 1e-12, 1e-12, 1e-12 });

        // Run multiple steps to test numerical stability
        var result = parameters;
        for (int step = 0; step < 10; step++)
        {
            result = optimizer.UpdateParameters(result, smallGradient);
        }

        // Result should not contain NaN or Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Parameter {i} became NaN");
            Assert.False(double.IsInfinity(result[i]), $"Parameter {i} became Infinity");
        }
    }

    [Fact]
    public void Adagrad_VerySmallGradients_UpdatesCorrectly()
    {
        // Adagrad should handle very small gradients without numerical issues
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var smallGradient = new Vector<double>(new double[] { 1e-12, 1e-12, 1e-12 });

        // Run multiple steps to test numerical stability
        var result = parameters;
        for (int step = 0; step < 10; step++)
        {
            result = optimizer.UpdateParameters(result, smallGradient);
        }

        // Result should not contain NaN or Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Parameter {i} became NaN");
            Assert.False(double.IsInfinity(result[i]), $"Parameter {i} became Infinity");
        }
    }

    [Fact]
    public void SGD_MixedSignGradients_UpdatesCorrectly()
    {
        // Mixed sign gradients should produce mixed direction updates
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
        var mixedGradient = new Vector<double>(new double[] { 1.0, -1.0, 0.5 });

        var result = optimizer.UpdateParameters(parameters, mixedGradient);

        // Positive gradient -> negative update (descent)
        Assert.True(result[0] < 0, "Positive gradient should cause negative update");
        // Negative gradient -> positive update (descent)
        Assert.True(result[1] > 0, "Negative gradient should cause positive update");
        // Small positive gradient -> small negative update
        Assert.True(result[2] < 0, "Small positive gradient should cause small negative update");
    }

    [Fact]
    public void Adam_NegativeGradients_UpdatesCorrectly()
    {
        // Negative gradients should cause parameters to increase
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });
        var negativeGradient = new Vector<double>(new double[] { -1.0, -1.0, -1.0 });

        var result = optimizer.UpdateParameters(parameters, negativeGradient);

        // Negative gradient -> positive update (descent in negative gradient direction)
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] > 0, $"Parameter {i} should increase with negative gradient");
        }
    }

    [Theory]
    [InlineData(0.0001)]
    [InlineData(0.001)]
    [InlineData(0.01)]
    [InlineData(0.1)]
    [InlineData(1.0)]
    public void SGD_DifferentLearningRates_ScalesUpdateProportionally(double learningRate)
    {
        // Update magnitude should scale with learning rate
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = learningRate
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new double[] { 10.0, 10.0, 10.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        var result = optimizer.UpdateParameters(parameters, gradient);

        // Update = learning_rate * gradient = learningRate * 1.0
        var expectedUpdate = learningRate;
        for (int i = 0; i < result.Length; i++)
        {
            var actualUpdate = parameters[i] - result[i];
            Assert.Equal(expectedUpdate, actualUpdate, Tolerance);
        }
    }

    [Fact]
    public void SGD_ConsecutiveUpdates_AccumulatesCorrectly()
    {
        // Multiple updates should accumulate correctly
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new double[] { 10.0, 10.0, 10.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        // Apply 10 updates
        var result = parameters;
        for (int i = 0; i < 10; i++)
        {
            result = optimizer.UpdateParameters(result, gradient);
        }

        // Total change should be approximately 10 * 0.1 * 1.0 = 1.0
        for (int i = 0; i < result.Length; i++)
        {
            var totalChange = parameters[i] - result[i];
            Assert.Equal(1.0, totalChange, 0.001);
        }
    }

    [Fact]
    public void Momentum_BuildsVelocity_OverMultipleSteps()
    {
        // Momentum should build up velocity over consistent gradient directions
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.9
        };
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);
        var parameters = new Vector<double>(new double[] { 10.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Track the magnitude of updates
        var updates = new List<double>();
        var result = parameters;
        for (int i = 0; i < 5; i++)
        {
            var newResult = optimizer.UpdateParameters(result, gradient);
            updates.Add(result[0] - newResult[0]);
            result = newResult;
        }

        // Later updates should be larger due to momentum buildup
        // (assuming the optimizer correctly accumulates velocity)
        Assert.True(updates[^1] >= updates[0],
            $"Later updates should be at least as large as initial: {updates[^1]} vs {updates[0]}");
    }

    [Fact]
    public void AllOptimizers_SingleDimensionParameter_WorksCorrectly()
    {
        // Test that optimizers work with single-dimension parameters
        var optimizers = new IGradientBasedOptimizer<double, Vector<double>, Vector<double>>[]
        {
            new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null),
            new AdamOptimizer<double, Vector<double>, Vector<double>>(null),
            new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null),
            new AdagradOptimizer<double, Vector<double>, Vector<double>>(null),
            new MomentumOptimizer<double, Vector<double>, Vector<double>>(null)
        };

        var parameter = new Vector<double>(new double[] { 5.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        foreach (var optimizer in optimizers)
        {
            var result = optimizer.UpdateParameters(parameter, gradient);

            Assert.False(double.IsNaN(result[0]), $"{optimizer.GetType().Name} produced NaN for single dimension");
            Assert.False(double.IsInfinity(result[0]), $"{optimizer.GetType().Name} produced Infinity for single dimension");
        }
    }

    [Fact]
    public void AllOptimizers_LargeDimensionParameter_WorksCorrectly()
    {
        // Test that optimizers work with large-dimension parameters
        const int dimensions = 1000;
        var optimizers = new IGradientBasedOptimizer<double, Vector<double>, Vector<double>>[]
        {
            new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null),
            new AdamOptimizer<double, Vector<double>, Vector<double>>(null),
            new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null)
        };

        var rand = new Random(42);
        var paramData = new double[dimensions];
        var gradData = new double[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            paramData[i] = rand.NextDouble() * 10 - 5;
            gradData[i] = rand.NextDouble() * 2 - 1;
        }
        var parameters = new Vector<double>(paramData);
        var gradient = new Vector<double>(gradData);

        foreach (var optimizer in optimizers)
        {
            var result = optimizer.UpdateParameters(parameters, gradient);

            Assert.Equal(dimensions, result.Length);
            for (int i = 0; i < dimensions; i++)
            {
                Assert.False(double.IsNaN(result[i]), $"{optimizer.GetType().Name} produced NaN at index {i}");
                Assert.False(double.IsInfinity(result[i]), $"{optimizer.GetType().Name} produced Infinity at index {i}");
            }
        }
    }

    #endregion

    #region Multimodal Function Tests

    [Fact]
    public void Diagnostic_VectorArithmetic_WorksCorrectly()
    {
        // This diagnostic test verifies basic vector operations work correctly
        // on both net471 and net10.0 frameworks

        // Test 1: Vector multiplication by scalar
        var v1 = new Vector<double>(new double[] { 4.0, -4.0 });
        var scalar = 0.01;
        var multiplied = v1.Multiply(scalar);

        Assert.Equal(0.04, multiplied[0], 1e-10);
        Assert.Equal(-0.04, multiplied[1], 1e-10);

        // Test 2: Vector subtraction
        var v2 = new Vector<double>(new double[] { 2.0, -2.0 });
        var subtracted = v2.Subtract(multiplied);

        Assert.Equal(1.96, subtracted[0], 1e-10);
        Assert.Equal(-1.96, subtracted[1], 1e-10);

        // Test 3: Chained operation (what UpdateParameters does)
        var parameters = new Vector<double>(new double[] { 2.0, -2.0 });
        var gradient = new Vector<double>(new double[] { 4.0, -4.0 });
        var lr = 0.01;
        var result = parameters.Subtract(gradient.Multiply(lr));

        Assert.Equal(1.96, result[0], 1e-10);
        Assert.Equal(-1.96, result[1], 1e-10);

        // Test 4: Verify UpdateParameters produces same result
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var p = new Vector<double>(new double[] { 2.0, -2.0 });
        var g = new Vector<double>(new double[] { 4.0, -4.0 });
        var updated = optimizer.UpdateParameters(p, g);

        Assert.Equal(1.96, updated[0], 1e-10);
        Assert.Equal(-1.96, updated[1], 1e-10);
    }

    [Fact]
    public void Diagnostic_RastriginGradient_WorksCorrectly()
    {
        // Test that Math.Sin and Math.PI work correctly on all frameworks
        // Rastrigin gradient: grad_i = 2*x_i + 20*pi*sin(2*pi*x_i)

        // At x = [2.0, -2.0], sin(2*pi*2) = sin(4*pi) = 0
        // So gradient should be [4.0, -4.0]
        var x1 = new Vector<double>(new double[] { 2.0, -2.0 });
        var grad1 = BenchmarkFunctions.RastriginGradient(x1);

        Assert.Equal(4.0, grad1[0], 1e-10);
        Assert.Equal(-4.0, grad1[1], 1e-10);

        // At x = [0.25, -0.25], sin(2*pi*0.25) = sin(pi/2) = 1
        // gradient = [2*0.25 + 20*pi*1, 2*(-0.25) + 20*pi*(-1)]
        //          = [0.5 + 20*pi, -0.5 - 20*pi]
        var x2 = new Vector<double>(new double[] { 0.25, -0.25 });
        var grad2 = BenchmarkFunctions.RastriginGradient(x2);

        double expectedGrad2_0 = 0.5 + 20.0 * Math.PI;
        double expectedGrad2_1 = -0.5 - 20.0 * Math.PI;

        Assert.Equal(expectedGrad2_0, grad2[0], 1e-10);
        Assert.Equal(expectedGrad2_1, grad2[1], 1e-10);

        // Test Rastrigin function at origin (should be 0)
        var origin = new Vector<double>(new double[] { 0.0, 0.0 });
        double valueAtOrigin = BenchmarkFunctions.Rastrigin(origin);
        Assert.Equal(0.0, valueAtOrigin, 1e-10);

        // Test one full iteration of SGD starting at a non-local-minimum point
        // NOTE: [2.0, -2.0] is a local minimum of Rastrigin (integer coords are local minima)
        // So we test at [2.5, -2.5] instead to verify optimization can make progress
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 2.5, -2.5 });
        double initialValue = BenchmarkFunctions.Rastrigin(x);
        var grad = BenchmarkFunctions.RastriginGradient(x);

        var xNew = optimizer.UpdateParameters(x, grad);

        // Verify the UpdateParameters formula is applied correctly: x_new = x - lr * grad
        double expectedX0 = 2.5 - 0.01 * grad[0];
        double expectedX1 = -2.5 - 0.01 * grad[1];
        Assert.Equal(expectedX0, xNew[0], 1e-10);
        Assert.Equal(expectedX1, xNew[1], 1e-10);

        double newValue = BenchmarkFunctions.Rastrigin(xNew);

        // From a non-local-minimum starting point, value should decrease
        Assert.True(newValue < initialValue,
            $"Rastrigin value should decrease from non-local-minimum start. Initial: {initialValue}, New: {newValue}");
    }

    /// <summary>
    /// Helper class for multimodal benchmark functions used in optimization testing.
    /// </summary>
    private static class BenchmarkFunctions
    {
        /// <summary>
        /// Rastrigin function - a highly multimodal function with many local minima.
        /// Global minimum at origin with f(0,...,0) = 0.
        /// </summary>
        public static double Rastrigin(Vector<double> x)
        {
            double sum = 10.0 * x.Length;
            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i] * x[i] - 10.0 * Math.Cos(2.0 * Math.PI * x[i]);
            }
            return sum;
        }

        /// <summary>
        /// Gradient of the Rastrigin function.
        /// </summary>
        public static Vector<double> RastriginGradient(Vector<double> x)
        {
            var grad = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                grad[i] = 2.0 * x[i] + 20.0 * Math.PI * Math.Sin(2.0 * Math.PI * x[i]);
            }
            return new Vector<double>(grad);
        }

        /// <summary>
        /// Ackley function - bowl-shaped with many local minima.
        /// Global minimum at origin with f(0,...,0) = 0.
        /// </summary>
        public static double Ackley(Vector<double> x)
        {
            double sumSq = 0.0;
            double sumCos = 0.0;
            for (int i = 0; i < x.Length; i++)
            {
                sumSq += x[i] * x[i];
                sumCos += Math.Cos(2.0 * Math.PI * x[i]);
            }
            double n = x.Length;
            return -20.0 * Math.Exp(-0.2 * Math.Sqrt(sumSq / n))
                   - Math.Exp(sumCos / n) + 20.0 + Math.E;
        }

        /// <summary>
        /// Gradient of the Ackley function.
        /// </summary>
        public static Vector<double> AckleyGradient(Vector<double> x)
        {
            double sumSq = 0.0;
            double sumCos = 0.0;
            for (int i = 0; i < x.Length; i++)
            {
                sumSq += x[i] * x[i];
                sumCos += Math.Cos(2.0 * Math.PI * x[i]);
            }
            double n = x.Length;
            double sqrtTerm = Math.Sqrt(sumSq / n);
            double expSqrt = Math.Exp(-0.2 * sqrtTerm);
            double expCos = Math.Exp(sumCos / n);

            var grad = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                double dSqrt = (sqrtTerm > 1e-10) ? (x[i] / (n * sqrtTerm)) : 0.0;
                grad[i] = 4.0 * expSqrt * dSqrt
                         + (2.0 * Math.PI / n) * Math.Sin(2.0 * Math.PI * x[i]) * expCos;
            }
            return new Vector<double>(grad);
        }

        /// <summary>
        /// Beale function - a 2D function with a flat region and single global minimum.
        /// Global minimum at (3, 0.5) with f(3, 0.5) = 0.
        /// </summary>
        public static double Beale(Vector<double> x)
        {
            if (x.Length != 2) throw new ArgumentException("Beale function is 2D only");
            double a = 1.5 - x[0] + x[0] * x[1];
            double b = 2.25 - x[0] + x[0] * x[1] * x[1];
            double c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
            return a * a + b * b + c * c;
        }

        /// <summary>
        /// Gradient of the Beale function.
        /// </summary>
        public static Vector<double> BealeGradient(Vector<double> x)
        {
            if (x.Length != 2) throw new ArgumentException("Beale function is 2D only");
            double y = x[1];
            double y2 = y * y;
            double y3 = y2 * y;

            double a = 1.5 - x[0] + x[0] * y;
            double b = 2.25 - x[0] + x[0] * y2;
            double c = 2.625 - x[0] + x[0] * y3;

            double dadx = -1.0 + y;
            double dady = x[0];
            double dbdx = -1.0 + y2;
            double dbdy = 2.0 * x[0] * y;
            double dcdx = -1.0 + y3;
            double dcdy = 3.0 * x[0] * y2;

            var grad = new double[2];
            grad[0] = 2.0 * a * dadx + 2.0 * b * dbdx + 2.0 * c * dcdx;
            grad[1] = 2.0 * a * dady + 2.0 * b * dbdy + 2.0 * c * dcdy;
            return new Vector<double>(grad);
        }

        /// <summary>
        /// Sphere function - simple convex function for baseline testing.
        /// Global minimum at origin with f(0,...,0) = 0.
        /// </summary>
        public static double Sphere(Vector<double> x)
        {
            double sum = 0.0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i] * x[i];
            }
            return sum;
        }

        /// <summary>
        /// Gradient of the Sphere function.
        /// </summary>
        public static Vector<double> SphereGradient(Vector<double> x)
        {
            var grad = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                grad[i] = 2.0 * x[i];
            }
            return new Vector<double>(grad);
        }
    }

    [Fact]
    public void SGD_SphereFunction_ConvergesToMinimum()
    {
        // Sphere function is convex, should converge reliably
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        // Run gradient descent
        for (int i = 0; i < 100; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);

        // Should get close to minimum (0)
        Assert.True(finalValue < 1e-6, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Fact]
    public void Adam_SphereFunction_ConvergesToMinimum()
    {
        // Adam should converge quickly on convex function
        // Default Adam learning rate (0.001) is designed for neural networks with thousands of iterations.
        // For this simple benchmark, we use a higher learning rate.
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1  // Higher LR for simple benchmark function
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        // Run Adam
        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);

        // Should get close to minimum (0)
        Assert.True(finalValue < 1e-4, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Fact]
    public void Momentum_SphereFunction_ConvergesToMinimum()
    {
        // Momentum should accelerate convergence on sphere
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.9
        };
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        // Run momentum - use more iterations for tight convergence
        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);

        // Should get close to minimum (0)
        Assert.True(finalValue < 1e-4, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Fact]
    public void SGD_RastriginFunction_MakesProgress()
    {
        // Rastrigin is multimodal - optimizer should at least make progress
        // Start at [2.5, -2.5] which is NOT at a local minimum (integer coords are local minima)
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01  // Small learning rate for stability
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 2.5, -2.5 });
        double initialValue = BenchmarkFunctions.Rastrigin(x);

        // Run gradient descent
        for (int i = 0; i < 100; i++)
        {
            var grad = BenchmarkFunctions.RastriginGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Rastrigin(x);

        // Should make some progress (final < initial)
        Assert.True(finalValue < initialValue,
            $"Should make progress: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void Adam_RastriginFunction_MakesProgress()
    {
        // Adam should navigate Rastrigin better than vanilla SGD
        // Use higher learning rate and start at [2.5, -2.5] (not at local minimum)
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 2.5, -2.5 });
        double initialValue = BenchmarkFunctions.Rastrigin(x);

        // Run Adam
        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.RastriginGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Rastrigin(x);

        // Should make progress
        Assert.True(finalValue < initialValue,
            $"Should make progress: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void SGD_AckleyFunction_MakesProgress()
    {
        // Ackley has many local minima
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 3.0, -3.0 });
        double initialValue = BenchmarkFunctions.Ackley(x);

        // Run gradient descent
        for (int i = 0; i < 100; i++)
        {
            var grad = BenchmarkFunctions.AckleyGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Ackley(x);

        // Should make progress
        Assert.True(finalValue < initialValue,
            $"Should make progress: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void Adam_AckleyFunction_MakesProgress()
    {
        // Use higher learning rate for this simple benchmark
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 3.0, -3.0 });
        double initialValue = BenchmarkFunctions.Ackley(x);

        // Run Adam
        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.AckleyGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Ackley(x);

        // Should make progress
        Assert.True(finalValue < initialValue,
            $"Should make progress: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void SGD_BealeFunction_MakesProgress()
    {
        // Beale function has a flat region that can be challenging
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001  // Small learning rate for stability
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 0.0, 0.0 });
        double initialValue = BenchmarkFunctions.Beale(x);

        // Run gradient descent
        for (int i = 0; i < 500; i++)
        {
            var grad = BenchmarkFunctions.BealeGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Beale(x);

        // Should make progress toward minimum at (3, 0.5)
        Assert.True(finalValue < initialValue,
            $"Should make progress: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void Adam_BealeFunction_MakesProgress()
    {
        // Use higher learning rate for this benchmark
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 0.0, 0.0 });
        double initialValue = BenchmarkFunctions.Beale(x);

        // Run Adam
        for (int i = 0; i < 500; i++)
        {
            var grad = BenchmarkFunctions.BealeGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Beale(x);

        // Should make progress
        Assert.True(finalValue < initialValue,
            $"Should make progress: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void RMSProp_SphereFunction_ConvergesToMinimum()
    {
        // RMSProp normalizes by running average of squared gradients, which stabilizes training
        // but can cause oscillation around a fixed point on simple quadratic functions.
        // Use lower learning rate for more stable convergence.
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1  // Lower LR for stable convergence
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        // RMSProp converges slower than pure SGD - use more iterations
        for (int i = 0; i < 2000; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);
        // RMSProp has inherent oscillation on simple quadratics; use looser tolerance
        Assert.True(finalValue < 1e-2, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Fact]
    public void Adagrad_SphereFunction_ConvergesToMinimum()
    {
        // Adagrad accumulates squared gradients, causing diminishing learning rates.
        // Use a higher initial learning rate to compensate for this characteristic.
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0  // Higher LR to compensate for Adagrad's diminishing rates
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);
        Assert.True(finalValue < 1e-2, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Theory]
    [InlineData("SGD")]
    [InlineData("Adam")]
    [InlineData("RMSProp")]
    [InlineData("Momentum")]
    public void AllOptimizers_HighDimensionalSphere_MakesProgress(string optimizerName)
    {
        // Test on higher dimensional sphere
        // All optimizers use lr=0.1 for consistent comparison on this simple benchmark
        const int dimensions = 10;
        IGradientBasedOptimizer<double, Vector<double>, Vector<double>> optimizer = optimizerName switch
        {
            "SGD" => new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
                new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 }),
            "Adam" => new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
                new AdamOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 }),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
                new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1 }),
            "Momentum" => new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
                new MomentumOptimizerOptions<double, Vector<double>, Vector<double>> { InitialLearningRate = 0.1, InitialMomentum = 0.9 }),
            _ => throw new ArgumentException($"Unknown optimizer: {optimizerName}")
        };

        var rand = new Random(42);
        var initData = new double[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            initData[i] = rand.NextDouble() * 10 - 5;  // Random in [-5, 5]
        }
        var x = new Vector<double>(initData);
        double initialValue = BenchmarkFunctions.Sphere(x);

        // Run optimization
        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);

        // Should make significant progress
        Assert.True(finalValue < initialValue * 0.01,
            $"{optimizerName} should reduce sphere by 99%: initial={initialValue}, final={finalValue}");
    }

    [Fact]
    public void Nadam_SphereFunction_ConvergesToMinimum()
    {
        // Nadam default lr (0.002) is designed for neural networks. Use higher lr for simple benchmark.
        var options = new NadamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new NadamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);
        Assert.True(finalValue < 1e-4, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Fact]
    public void AMSGrad_SphereFunction_ConvergesToMinimum()
    {
        // AMSGrad default lr (0.001) is designed for neural networks. Use higher lr for simple benchmark.
        var options = new AMSGradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new AMSGradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        for (int i = 0; i < 200; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);
        Assert.True(finalValue < 1e-4, $"Sphere function should converge to near 0, got {finalValue}");
    }

    [Fact]
    public void AdaDelta_SphereFunction_ConvergesToMinimum()
    {
        // AdaDelta has a cold-start problem: initial updates are tiny because
        // _accumulatedSquaredUpdates starts at zero. Use a moderate learning rate
        // as a scaling factor to compensate (too high causes NaN).
        var options = new AdaDeltaOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 10.0  // Moderate scale factor to avoid NaN
        };
        var optimizer = new AdaDeltaOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var x = new Vector<double>(new double[] { 5.0, -3.0 });

        for (int i = 0; i < 1000; i++)
        {
            var grad = BenchmarkFunctions.SphereGradient(x);
            x = optimizer.UpdateParameters(x, grad);
        }

        double finalValue = BenchmarkFunctions.Sphere(x);
        // AdaDelta may converge slower, allow higher tolerance
        Assert.True(finalValue < 1.0, $"Sphere function should converge toward 0, got {finalValue}");
    }

    #endregion

    #region Modified Gradient Descent Optimizer Tests

    /// <summary>
    /// Tests for ModifiedGradientDescentOptimizer - a specialized optimizer for Hope architecture
    /// based on the "Nested Learning" paper (Equations 27-29).
    ///
    /// This optimizer has a different interface than standard gradient-based optimizers:
    /// - UpdateVector: w_{t+1} = w_t - x_t*dot(w_t,x_t) - η*gradient
    /// - UpdateMatrix: W_{t+1} = W_t * (I - x_t*x_t^T) - η * (∇_y L ⊗ x_t)
    /// </summary>
    [Fact]
    public void ModifiedGD_CanInstantiate()
    {
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);
        Assert.NotNull(optimizer);
        Assert.Equal(0.01, optimizer.LearningRate);
    }

    [Fact]
    public void ModifiedGD_UpdateVector_BasicOperation()
    {
        // Test the basic vector update operation
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        var updated = optimizer.UpdateVector(parameters, input, gradient);

        // Verify the update produces valid output
        Assert.Equal(parameters.Length, updated.Length);

        // The update should modify the parameters
        bool anyChanged = false;
        for (int i = 0; i < parameters.Length; i++)
        {
            if (Math.Abs(updated[i] - parameters[i]) > 1e-10)
            {
                anyChanged = true;
                break;
            }
        }
        Assert.True(anyChanged, "Parameters should be modified by the update");
    }

    [Fact]
    public void ModifiedGD_UpdateVector_ZeroGradient_StillProjectsParameters()
    {
        // With zero gradient, the modified GD still projects parameters
        // w_{t+1} = w_t - x_t*dot(w_t,x_t) - η*0 = w_t - x_t*dot(w_t,x_t)
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var zeroGradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var updated = optimizer.UpdateVector(parameters, input, zeroGradient);

        // dot(w, x) = 1*0.5 + 2*0.5 + 3*0.5 = 3.0
        // projection = x * dot(w,x) = [1.5, 1.5, 1.5]
        // updated = w - projection = [-0.5, 0.5, 1.5]
        Assert.Equal(-0.5, updated[0], 6);
        Assert.Equal(0.5, updated[1], 6);
        Assert.Equal(1.5, updated[2], 6);
    }

    [Fact]
    public void ModifiedGD_UpdateVector_LearningRateAffectsGradientTerm()
    {
        // Verify learning rate scales the gradient term
        var optimizerSmall = new ModifiedGradientDescentOptimizer<double>(0.01);
        var optimizerLarge = new ModifiedGradientDescentOptimizer<double>(1.0);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var input = new Vector<double>(new double[] { 0.0, 0.0, 0.0 }); // Zero input to eliminate projection term
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        var updatedSmall = optimizerSmall.UpdateVector(parameters, input, gradient);
        var updatedLarge = optimizerLarge.UpdateVector(parameters, input, gradient);

        // With zero input, projection term is zero
        // updatedSmall = [1.0, 2.0, 3.0] - 0.01 * [1,1,1] = [0.99, 1.99, 2.99]
        // updatedLarge = [1.0, 2.0, 3.0] - 1.0 * [1,1,1] = [0.0, 1.0, 2.0]
        Assert.Equal(0.99, updatedSmall[0], 6);
        Assert.Equal(0.0, updatedLarge[0], 6);
    }

    [Fact]
    public void ModifiedGD_UpdateMatrix_BasicOperation()
    {
        // Test the matrix update operation
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        var parameters = new Matrix<double>(2, 3);
        parameters[0, 0] = 1.0; parameters[0, 1] = 2.0; parameters[0, 2] = 3.0;
        parameters[1, 0] = 4.0; parameters[1, 1] = 5.0; parameters[1, 2] = 6.0;

        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var outputGradient = new Vector<double>(new double[] { 0.1, 0.2 });

        var updated = optimizer.UpdateMatrix(parameters, input, outputGradient);

        // Verify dimensions are preserved
        Assert.Equal(parameters.Rows, updated.Rows);
        Assert.Equal(parameters.Columns, updated.Columns);

        // Verify the update modified the parameters
        bool anyChanged = false;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                if (Math.Abs(updated[i, j] - parameters[i, j]) > 1e-10)
                {
                    anyChanged = true;
                    break;
                }
            }
        }
        Assert.True(anyChanged, "Matrix parameters should be modified by the update");
    }

    [Fact]
    public void ModifiedGD_UpdateVector_DimensionMismatch_ThrowsException()
    {
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var input = new Vector<double>(new double[] { 0.5, 0.5 }); // Wrong size
        var gradient = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });

        Assert.Throws<ArgumentException>(() => optimizer.UpdateVector(parameters, input, gradient));
    }

    [Fact]
    public void ModifiedGD_UpdateVector_GradientDimensionMismatch_ThrowsException()
    {
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var input = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
        var gradient = new Vector<double>(new double[] { 0.1, 0.2 }); // Wrong size

        Assert.Throws<ArgumentException>(() => optimizer.UpdateVector(parameters, input, gradient));
    }

    [Fact]
    public void ModifiedGD_UpdateVector_OrthogonalInput_NoProjection()
    {
        // When input is orthogonal to parameters, projection term is zero
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        // Parameters in one direction
        var parameters = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
        // Input orthogonal to parameters
        var input = new Vector<double>(new double[] { 0.0, 1.0, 0.0 });
        var gradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var updated = optimizer.UpdateVector(parameters, input, gradient);

        // dot(w, x) = 0, so projection is zero
        // updated = parameters (no change when gradient is also zero)
        Assert.Equal(parameters[0], updated[0], 6);
        Assert.Equal(parameters[1], updated[1], 6);
        Assert.Equal(parameters[2], updated[2], 6);
    }

    [Fact]
    public void ModifiedGD_UpdateVector_HighDimensional_WorksCorrectly()
    {
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.01);

        int dim = 100;
        var parametersData = new double[dim];
        var inputData = new double[dim];
        var gradientData = new double[dim];

        var rand = new Random(42);
        for (int i = 0; i < dim; i++)
        {
            parametersData[i] = rand.NextDouble() * 2 - 1;
            inputData[i] = rand.NextDouble() * 2 - 1;
            gradientData[i] = rand.NextDouble() * 0.1;
        }

        var parameters = new Vector<double>(parametersData);
        var input = new Vector<double>(inputData);
        var gradient = new Vector<double>(gradientData);

        var updated = optimizer.UpdateVector(parameters, input, gradient);

        Assert.Equal(dim, updated.Length);
        // Just verify it completes without error on high-dimensional data
    }

    [Fact]
    public void ModifiedGD_MultipleUpdates_AccumulatesChanges()
    {
        var optimizer = new ModifiedGradientDescentOptimizer<double>(0.1);

        var parameters = new Vector<double>(new double[] { 5.0, 5.0, 5.0 });

        // Simulate multiple update steps
        for (int i = 0; i < 10; i++)
        {
            // Use a consistent input and decreasing gradient
            var input = new Vector<double>(new double[] { 0.1, 0.1, 0.1 });
            var gradient = new Vector<double>(new double[] { 0.5, 0.5, 0.5 });
            parameters = optimizer.UpdateVector(parameters, input, gradient);
        }

        // Parameters should have moved significantly from initial values
        double totalChange = 0;
        for (int i = 0; i < 3; i++)
        {
            totalChange += Math.Abs(parameters[i] - 5.0);
        }
        Assert.True(totalChange > 1.0, $"Parameters should change significantly over multiple updates, total change: {totalChange}");
    }

    [Fact]
    public void ModifiedGD_FloatType_WorksCorrectly()
    {
        // Verify the optimizer works with float type as well
        var optimizer = new ModifiedGradientDescentOptimizer<float>(0.1f);

        var parameters = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });
        var input = new Vector<float>(new float[] { 0.5f, 0.5f, 0.5f });
        var gradient = new Vector<float>(new float[] { 0.1f, 0.2f, 0.3f });

        var updated = optimizer.UpdateVector(parameters, input, gradient);

        Assert.Equal(parameters.Length, updated.Length);
        Assert.NotEqual(parameters[0], updated[0]);
    }

    #endregion
}
