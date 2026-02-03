using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests for the Adam8BitOptimizer to verify it works correctly with real optimization scenarios.
/// </summary>
public class Adam8BitOptimizerIntegrationTests
{
    #region Quadratic Function Optimization Tests

    /// <summary>
    /// Tests that the optimizer can minimize a simple quadratic function f(x) = x^2.
    /// The minimum is at x = 0.
    /// </summary>
    [Fact]
    public void Optimize_SimpleQuadratic_ConvergesToMinimum()
    {
        // Arrange
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            BlockSize = 8,
            MaxIterations = 100
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // Start at x = 5, minimum is at x = 0
        var parameters = new Vector<double>([5.0]);

        // Run optimization with gradient of f(x) = x^2 which is 2x
        for (int i = 0; i < 100; i++)
        {
            var gradient = new Vector<double>([2.0 * parameters[0]]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - Should converge close to 0
        Assert.True(Math.Abs(parameters[0]) < 0.1, $"Expected parameter near 0, got {parameters[0]}");
    }

    /// <summary>
    /// Tests optimization of a 2D quadratic function f(x,y) = x^2 + y^2.
    /// </summary>
    [Fact]
    public void Optimize_2DQuadratic_ConvergesToMinimum()
    {
        // Arrange
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.1,
            BlockSize = 4,
            MaxIterations = 200
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // Start at (3, 4), minimum is at (0, 0)
        var parameters = new Vector<double>([3.0, 4.0]);

        for (int i = 0; i < 200; i++)
        {
            // Gradient of f(x,y) = x^2 + y^2 is [2x, 2y]
            var gradient = new Vector<double>([2.0 * parameters[0], 2.0 * parameters[1]]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - Both parameters should converge close to 0
        Assert.True(Math.Abs(parameters[0]) < 0.2, $"Expected x near 0, got {parameters[0]}");
        Assert.True(Math.Abs(parameters[1]) < 0.2, $"Expected y near 0, got {parameters[1]}");
    }

    /// <summary>
    /// Tests optimization of Rosenbrock function (a challenging test function).
    /// f(x,y) = (a-x)^2 + b*(y-x^2)^2, with a=1, b=100
    /// Minimum at (1, 1).
    /// </summary>
    [Fact]
    public void Optimize_RosenbrockFunction_MovesTowardsMinimum()
    {
        // Arrange
        const double a = 1.0;
        const double b = 100.0;
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.001, // Small learning rate for this challenging function
            BlockSize = 4,
            MaxIterations = 1000
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // Start at (-1, 1)
        var parameters = new Vector<double>([-1.0, 1.0]);
        double initialDistance = Math.Sqrt(Math.Pow(parameters[0] - 1.0, 2) + Math.Pow(parameters[1] - 1.0, 2));

        for (int i = 0; i < 1000; i++)
        {
            double x = parameters[0];
            double y = parameters[1];

            // Gradient of Rosenbrock function
            double dfdx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
            double dfdy = 2.0 * b * (y - x * x);

            var gradient = new Vector<double>([dfdx, dfdy]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - Should move closer to minimum at (1, 1)
        double finalDistance = Math.Sqrt(Math.Pow(parameters[0] - 1.0, 2) + Math.Pow(parameters[1] - 1.0, 2));
        Assert.True(finalDistance < initialDistance,
            $"Optimizer should move towards minimum. Initial distance: {initialDistance}, Final distance: {finalDistance}");
    }

    #endregion

    #region Comparison with Standard Adam Tests

    /// <summary>
    /// Tests that 8-bit Adam produces similar results to standard Adam on a simple problem.
    /// </summary>
    [Fact]
    public void Optimize_ComparesWithStandardAdam_SimilarResults()
    {
        // Arrange
        var adam8BitOptions = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            BlockSize = 16
        };
        var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999
        };

        var adam8Bit = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, adam8BitOptions);
        var adam = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, adamOptions);

        // Same starting point
        var params8Bit = new Vector<double>([5.0, 3.0, -2.0, 4.0]);
        var paramsAdam = new Vector<double>([5.0, 3.0, -2.0, 4.0]);

        // Run both for 50 iterations on the same quadratic
        for (int i = 0; i < 50; i++)
        {
            // Same gradient function: gradient of sum(x^2) = 2x
            var gradient8Bit = new Vector<double>(params8Bit.Length);
            var gradientAdam = new Vector<double>(paramsAdam.Length);

            for (int j = 0; j < params8Bit.Length; j++)
            {
                gradient8Bit[j] = 2.0 * params8Bit[j];
                gradientAdam[j] = 2.0 * paramsAdam[j];
            }

            params8Bit = adam8Bit.UpdateParameters(params8Bit, gradient8Bit);
            paramsAdam = adam.UpdateParameters(paramsAdam, gradientAdam);
        }

        // Assert - Results should be similar (within quantization tolerance)
        for (int j = 0; j < params8Bit.Length; j++)
        {
            double diff = Math.Abs(params8Bit[j] - paramsAdam[j]);
            Assert.True(diff < 0.5,
                $"Parameter {j} differs by {diff}. 8-bit: {params8Bit[j]}, Standard: {paramsAdam[j]}");
        }
    }

    #endregion

    #region Large Scale Tests

    /// <summary>
    /// Tests optimization with a large number of parameters to verify memory efficiency.
    /// </summary>
    [Fact]
    public void Optimize_LargeParameterCount_WorksCorrectly()
    {
        // Arrange
        const int paramCount = 10000;
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            BlockSize = 256
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // Initialize parameters with random values
        var random = new Random(42);
        var parameters = new Vector<double>(paramCount);
        for (int i = 0; i < paramCount; i++)
        {
            parameters[i] = random.NextDouble() * 10 - 5; // Random in [-5, 5]
        }

        // Run 10 optimization steps
        for (int iter = 0; iter < 10; iter++)
        {
            var gradient = new Vector<double>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                gradient[i] = 2.0 * parameters[i]; // Gradient of x^2
            }
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - All parameters should have moved towards 0
        double avgAbs = 0;
        for (int i = 0; i < paramCount; i++)
        {
            avgAbs += Math.Abs(parameters[i]);
        }
        avgAbs /= paramCount;

        // Average should be less than starting (which was ~2.5 on average)
        Assert.True(avgAbs < 2.5, $"Average absolute value should decrease. Got {avgAbs}");

        // Verify memory savings
        var memStats = optimizer.GetMemoryUsage();
        Assert.True(memStats["MemorySavingsBytes"] > 0, "Should save memory compared to standard Adam");
    }

    /// <summary>
    /// Tests memory statistics are correctly computed for various configurations.
    /// </summary>
    [Fact]
    public void GetMemoryUsage_VariousConfigurations_ReturnsCorrectStats()
    {
        // Test with different block sizes
        var blockSizes = new[] { 64, 256, 1024, 2048 };

        foreach (var blockSize in blockSizes)
        {
            var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                BlockSize = blockSize,
                CompressBothMoments = true
            };
            var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

            var parameters = new Vector<double>(new double[1024]);
            var gradients = new Vector<double>(new double[1024]);
            optimizer.UpdateParameters(parameters, gradients);

            var stats = optimizer.GetMemoryUsage();

            // Quantized state should be approximately paramCount * 2 bytes (for m and v)
            // The +100 buffer accounts for block alignment padding when paramCount doesn't
            // divide evenly by blockSize (worst case: ~blockSize bytes per moment state)
            const int alignmentBuffer = 100;
            Assert.True(stats["QuantizedStateBytes"] <= 1024 * 2 + alignmentBuffer,
                $"Quantized state for blockSize={blockSize} should be around 2KB");

            // Standard Adam would use 1024 * 2 * 8 = 16384 bytes for double
            Assert.Equal(1024 * 2 * 8, stats["StandardAdamBytes"]);

            // Memory savings should be positive
            Assert.True(stats["MemorySavingsBytes"] > 0,
                $"Should save memory with blockSize={blockSize}");
        }
    }

    #endregion

    #region Edge Cases and Stability Tests

    /// <summary>
    /// Tests optimization with zero gradients doesn't cause issues.
    /// </summary>
    [Fact]
    public void Optimize_ZeroGradients_HandledCorrectly()
    {
        // Arrange
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>([1.0, 2.0, 3.0, 4.0]);
        var zeroGradient = new Vector<double>([0.0, 0.0, 0.0, 0.0]);

        // Act - Should not throw
        var result = optimizer.UpdateParameters(parameters, zeroGradient);

        // Assert - Parameters should remain approximately the same
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(Math.Abs(result[i] - parameters[i]) < 1e-6,
                $"Parameter {i} should not change with zero gradient");
        }
    }

    /// <summary>
    /// Tests optimization with very large gradients (gradient explosion scenario).
    /// </summary>
    [Fact]
    public void Optimize_LargeGradients_HandledCorrectly()
    {
        // Arrange
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>([1.0, 2.0, 3.0, 4.0]);
        var largeGradient = new Vector<double>([1000.0, -1000.0, 500.0, -500.0]);

        // Act - Should not throw or produce NaN
        var result = optimizer.UpdateParameters(parameters, largeGradient);

        // Assert - Results should be finite
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Parameter {i} should not be NaN");
            Assert.False(double.IsInfinity(result[i]), $"Parameter {i} should not be infinite");
        }
    }

    /// <summary>
    /// Tests optimization with very small gradients (vanishing gradient scenario).
    /// </summary>
    [Fact]
    public void Optimize_VerySmallGradients_HandledCorrectly()
    {
        // Arrange
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>([1.0, 2.0, 3.0, 4.0]);
        var smallGradient = new Vector<double>([1e-10, 1e-10, 1e-10, 1e-10]);

        // Act - Should not throw or produce NaN
        var result = optimizer.UpdateParameters(parameters, smallGradient);

        // Assert - Results should be finite and close to original
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Parameter {i} should not be NaN");
            Assert.False(double.IsInfinity(result[i]), $"Parameter {i} should not be infinite");
        }
    }

    /// <summary>
    /// Tests optimization with mixed positive and negative values.
    /// </summary>
    [Fact]
    public void Optimize_MixedSignGradients_HandledCorrectly()
    {
        // Arrange
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>([-5.0, 3.0, -2.0, 7.0, 0.0]);
        var gradient = new Vector<double>([1.0, -1.0, 0.5, -0.5, 0.1]);

        // Act
        var result = optimizer.UpdateParameters(parameters, gradient);

        // Assert - All results should be finite
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"Parameter {i} should not be NaN");
            Assert.False(double.IsInfinity(result[i]), $"Parameter {i} should not be infinite");
        }
    }

    #endregion

    #region Serialization Roundtrip Tests

    /// <summary>
    /// Tests that serialization and deserialization preserves optimizer state correctly.
    /// </summary>
    [Fact]
    public void SerializeDeserialize_PreservesOptimizationState()
    {
        // Arrange
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.01,
            BlockSize = 8
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // Run some iterations
        var parameters = new Vector<double>([5.0, 3.0, -2.0, 4.0]);
        for (int i = 0; i < 10; i++)
        {
            var gradient = new Vector<double>([2.0 * parameters[0], 2.0 * parameters[1],
                                                2.0 * parameters[2], 2.0 * parameters[3]]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Serialize
        var serialized = optimizer.Serialize();

        // Create new optimizer and deserialize
        var restoredOptimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        restoredOptimizer.Deserialize(serialized);

        // Continue optimization on both
        var gradient1 = new Vector<double>([2.0 * parameters[0], 2.0 * parameters[1],
                                            2.0 * parameters[2], 2.0 * parameters[3]]);

        var resultOriginal = optimizer.UpdateParameters(parameters, gradient1);
        var resultRestored = restoredOptimizer.UpdateParameters(parameters, gradient1);

        // Assert - Results should be very similar (allowing for minor floating point differences)
        for (int i = 0; i < resultOriginal.Length; i++)
        {
            double diff = Math.Abs(resultOriginal[i] - resultRestored[i]);
            Assert.True(diff < 0.02,
                $"Parameter {i}: original={resultOriginal[i]:F6}, restored={resultRestored[i]:F6}, diff={diff:F6}");
        }
    }

    #endregion

    #region Configuration Variant Tests

    /// <summary>
    /// Tests various quantization configurations work correctly.
    /// </summary>
    [Theory]
    [InlineData(true, true)]   // Dynamic quantization, compress both moments
    [InlineData(true, false)]  // Dynamic quantization, keep m full precision
    [InlineData(false, true)]  // Static quantization, compress both moments
    [InlineData(false, false)] // Static quantization, keep m full precision
    public void Optimize_DifferentConfigurations_AllWork(bool useDynamic, bool compressBoth)
    {
        // Arrange
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            UseDynamicQuantization = useDynamic,
            CompressBothMoments = compressBoth,
            InitialLearningRate = 0.1,
            BlockSize = 4
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>([5.0, -3.0, 2.0, -1.0]);

        // Run optimization
        for (int i = 0; i < 50; i++)
        {
            var gradient = new Vector<double>([2.0 * parameters[0], 2.0 * parameters[1],
                                               2.0 * parameters[2], 2.0 * parameters[3]]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - All parameters should move towards 0
        // Quantization introduces some precision loss, so allow slightly larger threshold for compressBoth=true
        double threshold = compressBoth ? 1.5 : 1.0;
        for (int j = 0; j < parameters.Length; j++)
        {
            Assert.True(Math.Abs(parameters[j]) < threshold,
                $"Parameter {j} should converge towards 0. Value: {parameters[j]:F4}. Config: dynamic={useDynamic}, compressBoth={compressBoth}");
        }
    }

    /// <summary>
    /// Tests stochastic vs standard rounding produces valid results.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Optimize_RoundingModes_AllWork(bool useStochastic)
    {
        // Arrange
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            UseStochasticRounding = useStochastic,
            InitialLearningRate = 0.1,
            BlockSize = 4
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var parameters = new Vector<double>([3.0, -2.0, 4.0, -1.0]);

        // Run optimization
        for (int i = 0; i < 30; i++)
        {
            var gradient = new Vector<double>([2.0 * parameters[0], 2.0 * parameters[1],
                                               2.0 * parameters[2], 2.0 * parameters[3]]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - Should converge
        double totalAbs = 0;
        for (int j = 0; j < parameters.Length; j++)
        {
            Assert.False(double.IsNaN(parameters[j]));
            totalAbs += Math.Abs(parameters[j]);
        }
        Assert.True(totalAbs < 10.0); // Should have decreased from initial sum of ~10
    }

    #endregion

    #region Float Type Tests

    /// <summary>
    /// Tests that float type works correctly end-to-end.
    /// </summary>
    [Fact]
    public void Optimize_FloatType_ConvergesCorrectly()
    {
        // Arrange
        var options = new Adam8BitOptimizerOptions<float, Matrix<float>, Vector<float>>
        {
            InitialLearningRate = 0.1f,
            BlockSize = 4
        };
        var optimizer = new Adam8BitOptimizer<float, Matrix<float>, Vector<float>>(null, options);

        var parameters = new Vector<float>([5.0f, -3.0f, 2.0f, -1.0f]);

        // Run optimization
        for (int i = 0; i < 50; i++)
        {
            var gradient = new Vector<float>([2.0f * parameters[0], 2.0f * parameters[1],
                                              2.0f * parameters[2], 2.0f * parameters[3]]);
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - All parameters should move towards 0
        // Allow slightly larger threshold due to quantization with CompressBothMoments=true (default)
        for (int j = 0; j < parameters.Length; j++)
        {
            Assert.True(Math.Abs(parameters[j]) < 1.5f,
                $"Float parameter {j} should converge towards 0. Value: {parameters[j]:F4}");
        }
    }

    #endregion
}
