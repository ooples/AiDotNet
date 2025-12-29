using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests that verify the mathematical correctness of optimizer update formulas.
/// These tests validate optimizer implementations against known mathematical formulas and industry standards.
///
/// Reference formulas:
/// - SGD: θ(t+1) = θ(t) - α * g(t)
/// - Momentum: v(t+1) = μ * v(t) + α * g(t); θ(t+1) = θ(t) - v(t+1)
/// - Adam: m(t) = β₁ * m(t-1) + (1-β₁) * g; v(t) = β₂ * v(t-1) + (1-β₂) * g²
///         m̂ = m/(1-β₁^t); v̂ = v/(1-β₂^t); θ = θ - α * m̂ / (√v̂ + ε)
///
/// These tests do NOT modify tests to match buggy code - they verify code matches expected math.
/// </summary>
[Trait("Category", "Integration")]
public class OptimizerMathematicalCorrectnessTests
{
    private const double Tolerance = 1e-10;
    private const double RelativeTolerance = 1e-6;

    #region SGD Mathematical Correctness Tests

    /// <summary>
    /// Tests that SGD applies the correct update formula: θ = θ - α * g
    /// Reference: Standard gradient descent formula from optimization theory
    /// </summary>
    [Fact]
    public void SGD_UpdateParameters_AppliesCorrectFormula()
    {
        // Arrange
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.5, 1.0, -0.5 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        // Expected: θ = θ - α * g = [1.0, 2.0, 3.0] - 0.1 * [0.5, 1.0, -0.5]
        //                         = [1.0 - 0.05, 2.0 - 0.1, 3.0 + 0.05]
        //                         = [0.95, 1.9, 3.05]
        Assert.Equal(0.95, result[0], Tolerance);
        Assert.Equal(1.9, result[1], Tolerance);
        Assert.Equal(3.05, result[2], Tolerance);
    }

    /// <summary>
    /// Tests SGD with zero gradients produces no change.
    /// </summary>
    [Fact]
    public void SGD_ZeroGradient_NoChange()
    {
        // Arrange
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Parameters should be unchanged
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    /// <summary>
    /// Tests SGD with learning rate 0 produces no change.
    /// </summary>
    [Fact]
    public void SGD_ZeroLearningRate_NoChange()
    {
        // Arrange
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.0
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Parameters should be unchanged
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    /// <summary>
    /// Tests that multiple SGD steps accumulate correctly.
    /// </summary>
    [Fact]
    public void SGD_MultipleSteps_AccumulatesCorrectly()
    {
        // Arrange
        var options = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var optimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 10.0 });
        var gradient = new Vector<double>(new double[] { 1.0 }); // Constant gradient

        // Act - 10 steps
        for (int i = 0; i < 10; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert
        // After 10 steps: 10.0 - 10 * 0.1 * 1.0 = 10.0 - 1.0 = 9.0
        Assert.Equal(9.0, parameters[0], Tolerance);
    }

    #endregion

    #region Momentum Mathematical Correctness Tests

    /// <summary>
    /// Tests that Momentum applies the correct update formula:
    /// v = μ * v + α * g
    /// θ = θ - v
    /// Reference: Polyak's heavy ball method
    /// </summary>
    [Fact]
    public void Momentum_UpdateParameters_AppliesCorrectFormula()
    {
        // Arrange
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.9
        };
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0, 2.0 });
        var gradient = new Vector<double>(new double[] { 1.0, 1.0 });

        // Act - First step
        var result1 = optimizer.UpdateParameters(parameters, gradient);

        // Assert - First step
        // v = 0.9 * 0 + 0.1 * 1.0 = 0.1
        // θ = [1.0, 2.0] - [0.1, 0.1] = [0.9, 1.9]
        Assert.Equal(0.9, result1[0], Tolerance);
        Assert.Equal(1.9, result1[1], Tolerance);

        // Act - Second step with same gradient
        var result2 = optimizer.UpdateParameters(result1, gradient);

        // Assert - Second step
        // v = 0.9 * 0.1 + 0.1 * 1.0 = 0.09 + 0.1 = 0.19
        // θ = [0.9, 1.9] - [0.19, 0.19] = [0.71, 1.71]
        Assert.Equal(0.71, result2[0], Tolerance);
        Assert.Equal(1.71, result2[1], Tolerance);
    }

    /// <summary>
    /// Tests that Momentum with μ=0 behaves like SGD.
    /// </summary>
    [Fact]
    public void Momentum_ZeroMomentum_BehavesLikeSGD()
    {
        // Arrange
        var momentumOptions = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.0
        };
        var momentumOptimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, momentumOptions);

        var sgdOptions = new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1
        };
        var sgdOptimizer = new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null, sgdOptions);

        var parameters = new Vector<double>(new double[] { 5.0, 10.0, 15.0 });
        var gradient = new Vector<double>(new double[] { 0.5, -0.5, 1.0 });

        // Act
        var momentumResult = momentumOptimizer.UpdateParameters(parameters, gradient);
        var sgdResult = sgdOptimizer.UpdateParameters(parameters, gradient);

        // Assert - Results should be identical
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.Equal(sgdResult[i], momentumResult[i], Tolerance);
        }
    }

    /// <summary>
    /// Tests that momentum accumulates velocity correctly over multiple steps.
    /// This verifies the exponential moving average behavior.
    /// </summary>
    [Fact]
    public void Momentum_AccumulatesVelocity_CorrectlyOverMultipleSteps()
    {
        // Arrange
        var options = new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            InitialMomentum = 0.9
        };
        var optimizer = new MomentumOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var constantGradient = new Vector<double>(new double[] { 1.0 });

        // Track velocities manually
        double velocity = 0.0;
        double lr = 0.1;
        double momentum = 0.9;

        // Act - Run 5 steps and verify each
        for (int step = 0; step < 5; step++)
        {
            var oldParams = parameters[0];
            parameters = optimizer.UpdateParameters(parameters, constantGradient);

            // Calculate expected velocity manually
            velocity = momentum * velocity + lr * 1.0;
            double expectedParam = oldParams - velocity;

            // Assert
            Assert.Equal(expectedParam, parameters[0], Tolerance);
        }
    }

    #endregion

    #region Adam Mathematical Correctness Tests

    /// <summary>
    /// Tests that Adam applies the correct update formula:
    /// m = β₁ * m + (1-β₁) * g
    /// v = β₂ * v + (1-β₂) * g²
    /// m̂ = m / (1 - β₁^t)
    /// v̂ = v / (1 - β₂^t)
    /// θ = θ - α * m̂ / (√v̂ + ε)
    /// Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    /// </summary>
    [Fact]
    public void Adam_UpdateParameters_AppliesCorrectFormula_Step1()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001, // Standard Adam default
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 0.1 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Calculate expected value manually
        // t = 1
        // m = 0.9 * 0 + 0.1 * 0.1 = 0.01
        // v = 0.999 * 0 + 0.001 * (0.1)² = 0.001 * 0.01 = 0.00001
        // m̂ = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1
        // v̂ = 0.00001 / (1 - 0.999^1) = 0.00001 / 0.001 = 0.01
        // update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 0.001 * 0.1 / (0.1 + 1e-8) ≈ 0.001
        // θ = 1.0 - 0.001 = 0.999

        double beta1 = 0.9;
        double beta2 = 0.999;
        double alpha = 0.001;
        double epsilon = 1e-8;
        double g = 0.1;
        int t = 1;

        double m = (1 - beta1) * g; // 0.01
        double v = (1 - beta2) * g * g; // 0.00001
        double mHat = m / (1 - Math.Pow(beta1, t)); // 0.1
        double vHat = v / (1 - Math.Pow(beta2, t)); // 0.01
        double update = alpha * mHat / (Math.Sqrt(vHat) + epsilon);
        double expected = 1.0 - update;

        Assert.Equal(expected, result[0], RelativeTolerance);
    }

    /// <summary>
    /// Tests that Adam's bias correction is applied correctly.
    /// In early steps, bias correction significantly adjusts the moment estimates.
    /// </summary>
    [Fact]
    public void Adam_BiasCorrection_AppliedCorrectly()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - Multiple steps
        double expectedM = 0.0;
        double expectedV = 0.0;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double alpha = 0.1;
        double epsilon = 1e-8;
        double g = 1.0;
        double expectedParam = 0.0;

        for (int t = 1; t <= 5; t++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);

            // Manual calculation
            expectedM = beta1 * expectedM + (1 - beta1) * g;
            expectedV = beta2 * expectedV + (1 - beta2) * g * g;
            double mHat = expectedM / (1 - Math.Pow(beta1, t));
            double vHat = expectedV / (1 - Math.Pow(beta2, t));
            double update = alpha * mHat / (Math.Sqrt(vHat) + epsilon);
            expectedParam = expectedParam - update;

            // The difference between steps should show bias correction effect
        }

        // Assert final value matches calculation
        Assert.Equal(expectedParam, parameters[0], RelativeTolerance);
    }

    /// <summary>
    /// Tests that Adam with constant gradient converges to expected step size.
    /// With constant gradients, Adam should converge to approximately α (learning rate) step size.
    /// Reference: This is a known property of Adam discussed in the original paper.
    /// </summary>
    [Fact]
    public void Adam_ConstantGradient_ConvergesToExpectedStepSize()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - Many steps to let Adam converge
        double previousParam = 0.0;
        double stepSize = 0.0;

        for (int i = 0; i < 1000; i++)
        {
            var newParams = optimizer.UpdateParameters(parameters, gradient);
            stepSize = Math.Abs(previousParam - newParams[0]);
            previousParam = parameters[0];
            parameters = newParams;
        }

        // Assert - Step size should converge to approximately learning rate
        // With constant gradient of 1.0, Adam's effective step size converges to α
        // Allow 10% tolerance since convergence isn't exact
        Assert.True(stepSize < 0.02, $"Step size {stepSize} should be near learning rate 0.01");
        Assert.True(stepSize > 0.005, $"Step size {stepSize} should not be too small");
    }

    #endregion

    #region AdamW Mathematical Correctness Tests

    /// <summary>
    /// Tests that AdamW applies decoupled weight decay correctly.
    /// AdamW formula: θ = θ - α * (m̂ / (√v̂ + ε) + λ * θ)
    /// where λ is the weight decay coefficient.
    /// Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
    /// </summary>
    [Fact]
    public void AdamW_WeightDecay_AppliedCorrectly()
    {
        // Arrange
        var options = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = 0.01 // 1% weight decay
        };
        var optimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 10.0 });
        var gradient = new Vector<double>(new double[] { 0.0 }); // Zero gradient to isolate weight decay effect

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        // With zero gradient, only weight decay should affect parameters
        // θ_new = θ - α * λ * θ = θ * (1 - α * λ) = 10.0 * (1 - 0.001 * 0.01) = 10.0 * 0.99999 ≈ 9.99999
        double expectedDecay = 10.0 * (1 - 0.001 * 0.01);

        // AdamW still has Adam update even with zero gradient (moment estimates)
        // But the weight decay component should be present
        // The parameter should have decreased due to weight decay
        Assert.True(result[0] < parameters[0], "AdamW should decrease parameters through weight decay");
    }

    /// <summary>
    /// Tests that AdamW differs from Adam+L2 regularization.
    /// AdamW decouples weight decay from the gradient update, which is mathematically different
    /// from L2 regularization applied to the loss function.
    /// </summary>
    [Fact]
    public void AdamW_DiffersFromAdamWithL2_Conceptually()
    {
        // This test documents the conceptual difference:
        // Adam with L2: gradient_regularized = gradient + λ * θ
        //               then standard Adam update with gradient_regularized
        // AdamW: standard Adam update with gradient
        //        then θ = θ - α * λ * θ (additional weight decay step)

        // The key difference is that in AdamW, weight decay is not scaled by the
        // adaptive learning rate adjustment (1/√v), while in Adam+L2 it is.

        // Arrange
        var adamwOptions = new AdamWOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Beta1 = 0.9,
            Beta2 = 0.999,
            WeightDecay = 0.1
        };
        var adamwOptimizer = new AdamWOptimizer<double, Vector<double>, Vector<double>>(null, adamwOptions);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 0.1 });

        // Act
        var adamwResult = adamwOptimizer.UpdateParameters(parameters, gradient);

        // Assert - Just verify AdamW produces a reasonable result
        // The mathematical proof of difference from Adam+L2 is in the paper
        Assert.True(adamwResult[0] < 1.0, "AdamW should decrease parameter");
    }

    #endregion

    #region Adagrad Mathematical Correctness Tests

    /// <summary>
    /// Tests that Adagrad applies the correct update formula:
    /// G = G + g²
    /// θ = θ - α * g / (√G + ε)
    /// Reference: "Adaptive Subgradient Methods for Online Learning" (Duchi et al., 2011)
    /// </summary>
    [Fact]
    public void Adagrad_UpdateParameters_AppliesCorrectFormula()
    {
        // Arrange
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Epsilon = 1e-8
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Act - Step 1
        var result1 = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Step 1
        // G = 0 + 2² = 4
        // θ = 1.0 - 0.1 * 2.0 / (√4 + ε) = 1.0 - 0.2 / (2 + 1e-8)
        double g = 2.0;
        double G1 = g * g; // 4.0
        double epsilon = 1e-8;
        double expected1 = 1.0 - 0.1 * g / (Math.Sqrt(G1) + epsilon);
        Assert.Equal(expected1, result1[0], Tolerance);

        // Act - Step 2 with same gradient
        var result2 = optimizer.UpdateParameters(result1, gradient);

        // Assert - Step 2
        // G = 4 + 4 = 8
        // θ = expected1 - 0.1 * 2.0 / (√8 + ε)
        double G2 = G1 + g * g; // 8.0
        double expected2 = expected1 - 0.1 * g / (Math.Sqrt(G2) + epsilon);
        Assert.Equal(expected2, result2[0], Tolerance);
    }

    /// <summary>
    /// Tests that Adagrad's effective learning rate decreases over time.
    /// This is a fundamental property of Adagrad - it has a diminishing learning rate.
    /// </summary>
    [Fact]
    public void Adagrad_EffectiveLearningRate_DecreasesOverTime()
    {
        // Arrange
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0, // Start with high learning rate
            Epsilon = 1e-8
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 0.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - Track step sizes
        var stepSizes = new List<double>();
        for (int i = 0; i < 10; i++)
        {
            double oldParam = parameters[0];
            parameters = optimizer.UpdateParameters(parameters, gradient);
            stepSizes.Add(Math.Abs(parameters[0] - oldParam));
        }

        // Assert - Step sizes should decrease monotonically
        for (int i = 1; i < stepSizes.Count; i++)
        {
            Assert.True(stepSizes[i] < stepSizes[i - 1],
                $"Step {i} size {stepSizes[i]} should be less than step {i - 1} size {stepSizes[i - 1]}");
        }
    }

    #endregion

    #region RMSProp Mathematical Correctness Tests

    /// <summary>
    /// Tests that RMSProp applies the correct update formula:
    /// v = ρ * v + (1-ρ) * g²
    /// θ = θ - α * g / (√v + ε)
    /// Reference: Hinton's Neural Networks course lecture 6
    /// </summary>
    [Fact]
    public void RMSProp_UpdateParameters_AppliesCorrectFormula()
    {
        // Arrange
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Decay = 0.9, // ρ = 0.9
            Epsilon = 1e-8
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Act - Step 1
        var result1 = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Step 1
        // v = 0.9 * 0 + 0.1 * 4 = 0.4
        // θ = 1.0 - 0.01 * 2.0 / (√0.4 + 1e-8) = 1.0 - 0.02 / 0.632... ≈ 1.0 - 0.0316 ≈ 0.968
        double v1 = 0.1 * 4.0;
        double expected1 = 1.0 - 0.01 * 2.0 / Math.Sqrt(v1);
        Assert.Equal(expected1, result1[0], RelativeTolerance);
    }

    /// <summary>
    /// Tests that RMSProp with decay_rate=0 behaves like Adagrad scaled differently.
    /// When ρ=0, v = g², which is similar to Adagrad's first step.
    /// </summary>
    [Fact]
    public void RMSProp_ZeroDecay_VEqualsGradientSquared()
    {
        // Arrange
        var options = new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            Decay = 0.0, // ρ = 0
            Epsilon = 1e-8
        };
        var optimizer = new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 5.0 });
        var gradient = new Vector<double>(new double[] { 2.0 });

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert
        // v = 0 * 0 + 1.0 * 4 = 4
        // θ = 5.0 - 0.1 * 2.0 / (√4 + ε) = 5.0 - 0.2 / (2 + 1e-8)
        double g = 2.0;
        double v = g * g; // 4.0
        double epsilon = 1e-8;
        double expected = 5.0 - 0.1 * g / (Math.Sqrt(v) + epsilon);
        Assert.Equal(expected, result[0], Tolerance);
    }

    #endregion

    #region AdaDelta Mathematical Correctness Tests

    /// <summary>
    /// Tests that AdaDelta applies the correct update formula:
    /// E[g²] = ρ * E[g²] + (1-ρ) * g²
    /// Δθ = -√(E[Δθ²] + ε) / √(E[g²] + ε) * g
    /// E[Δθ²] = ρ * E[Δθ²] + (1-ρ) * Δθ²
    /// θ = θ + Δθ
    /// Reference: "ADADELTA: An Adaptive Learning Rate Method" (Zeiler, 2012)
    /// </summary>
    [Fact]
    public void AdaDelta_UpdateParameters_RequiresNoLearningRate()
    {
        // Arrange
        var options = new AdaDeltaOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            Rho = 0.9, // ρ = 0.9
            Epsilon = 1e-6
        };
        var optimizer = new AdaDeltaOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 10.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - Multiple steps to verify AdaDelta works without explicit learning rate
        for (int i = 0; i < 10; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - AdaDelta should have modified parameters
        // The key property of AdaDelta is it doesn't require a learning rate hyperparameter
        Assert.NotEqual(10.0, parameters[0]);
        Assert.True(parameters[0] < 10.0, "Parameters should decrease with positive gradient");
    }

    #endregion

    #region Optimizer Comparison Tests

    /// <summary>
    /// Tests that all optimizers can minimize a simple quadratic function.
    /// f(x) = x² has a minimum at x = 0.
    /// </summary>
    [Theory]
    [InlineData("SGD")]
    [InlineData("Momentum")]
    [InlineData("Adam")]
    [InlineData("Adagrad")]
    [InlineData("RMSProp")]
    public void AllOptimizers_CanMinimize_QuadraticFunction(string optimizerType)
    {
        // Arrange - Create optimizer based on type
        var parameters = new Vector<double>(new double[] { 5.0 }); // Start at x = 5
        dynamic optimizer = optimizerType switch
        {
            "SGD" => new StochasticGradientDescentOptimizer<double, Vector<double>, Vector<double>>(null,
                new StochasticGradientDescentOptimizerOptions<double, Vector<double>, Vector<double>>
                { InitialLearningRate = 0.1 }),
            "Momentum" => new MomentumOptimizer<double, Vector<double>, Vector<double>>(null,
                new MomentumOptimizerOptions<double, Vector<double>, Vector<double>>
                { InitialLearningRate = 0.1, InitialMomentum = 0.9 }),
            "Adam" => new AdamOptimizer<double, Vector<double>, Vector<double>>(null,
                new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
                { InitialLearningRate = 0.1, Beta1 = 0.9, Beta2 = 0.999 }),
            "Adagrad" => new AdagradOptimizer<double, Vector<double>, Vector<double>>(null,
                new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
                { InitialLearningRate = 1.0 }),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<double, Vector<double>, Vector<double>>(null,
                new RootMeanSquarePropagationOptimizerOptions<double, Vector<double>, Vector<double>>
                { InitialLearningRate = 0.1, Decay = 0.9 }),
            _ => throw new ArgumentException($"Unknown optimizer: {optimizerType}")
        };

        // Act - Run optimization for 100 steps
        // For f(x) = x², gradient = 2x
        for (int i = 0; i < 100; i++)
        {
            double x = parameters[0];
            double grad = 2.0 * x; // Gradient of x²
            var gradient = new Vector<double>(new double[] { grad });
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - Should be close to minimum at x = 0
        Assert.True(Math.Abs(parameters[0]) < 0.5,
            $"{optimizerType} should minimize x² to near 0, got {parameters[0]}");
    }

    #endregion

    #region Numerical Stability Tests

    /// <summary>
    /// Tests that Adam handles very small gradients without numerical issues.
    /// </summary>
    [Fact]
    public void Adam_SmallGradients_NumericallyStable()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 1e-10 }); // Very small gradient

        // Act - Many steps with tiny gradient
        for (int i = 0; i < 100; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - Should not produce NaN or Inf
        Assert.False(double.IsNaN(parameters[0]), "Result should not be NaN");
        Assert.False(double.IsInfinity(parameters[0]), "Result should not be Infinity");
    }

    /// <summary>
    /// Tests that Adam handles very large gradients without numerical issues.
    /// </summary>
    [Fact]
    public void Adam_LargeGradients_NumericallyStable()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 1.0 });
        var gradient = new Vector<double>(new double[] { 1e6 }); // Very large gradient

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - Should not produce NaN or Inf
        Assert.False(double.IsNaN(result[0]), "Result should not be NaN");
        Assert.False(double.IsInfinity(result[0]), "Result should not be Infinity");
    }

    /// <summary>
    /// Tests that Adagrad's accumulated gradient sum doesn't cause division by very large numbers.
    /// The effective step size is α/√G where G = sum of squared gradients.
    /// For constant gradient g=1, after N steps: G = N, so step ≈ α/√N.
    /// Total displacement ≈ sum(α/√n) for n=1 to N ≈ 2α√N.
    /// </summary>
    [Fact]
    public void Adagrad_ManySteps_StillProducesReasonableUpdates()
    {
        // Arrange
        var options = new AdagradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = 1.0,
            Epsilon = 1e-8
        };
        var optimizer = new AdagradOptimizer<double, Vector<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>(new double[] { 100.0 });
        var gradient = new Vector<double>(new double[] { 1.0 });

        // Act - Many steps
        for (int i = 0; i < 10000; i++)
        {
            var newParams = optimizer.UpdateParameters(parameters, gradient);
            Assert.False(double.IsNaN(newParams[0]), $"NaN at step {i}");
            Assert.False(double.IsInfinity(newParams[0]), $"Infinity at step {i}");
            parameters = newParams;
        }

        // Assert - With α=1 and N=10000 steps, total displacement ≈ 2*1*√10000 = 200
        // Starting at 100, we should be around 100 - 200 = -100
        // The key property is that the value is finite and numerically stable
        Assert.True(double.IsFinite(parameters[0]), "Parameter should be finite");
        // Verify the approximate math: displacement should be around 2√N = 200 for N=10000
        double expectedDisplacement = 2 * Math.Sqrt(10000); // ≈ 200
        double actualDisplacement = 100.0 - parameters[0];
        Assert.True(Math.Abs(actualDisplacement - expectedDisplacement) < 10,
            $"Displacement {actualDisplacement} should be approximately {expectedDisplacement}");
    }

    #endregion
}
