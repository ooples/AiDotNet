using AiDotNet.Initialization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Initialization;

/// <summary>
/// Deep integration tests for Initialization:
/// EagerInitializationStrategy (Xavier/Glorot normal weights, zero biases),
/// LazyInitializationStrategy (deferred flag, same Xavier normal + zero biases),
/// ZeroInitializationStrategy (all zeros),
/// InitializationStrategies factory, InitializationStrategyType enum,
/// mathematical correctness of Xavier/He weight distributions.
/// </summary>
public class InitializationDeepMathIntegrationTests
{
    // ============================
    // EagerInitializationStrategy: Properties
    // ============================

    [Fact]
    public void Eager_IsLazy_False()
    {
        var strategy = new EagerInitializationStrategy<double>();
        Assert.False(strategy.IsLazy);
    }

    [Fact]
    public void Eager_LoadFromExternal_False()
    {
        var strategy = new EagerInitializationStrategy<double>();
        Assert.False(strategy.LoadFromExternal);
    }

    // ============================
    // LazyInitializationStrategy: Properties
    // ============================

    [Fact]
    public void Lazy_IsLazy_True()
    {
        var strategy = new LazyInitializationStrategy<double>();
        Assert.True(strategy.IsLazy);
    }

    [Fact]
    public void Lazy_LoadFromExternal_False()
    {
        var strategy = new LazyInitializationStrategy<double>();
        Assert.False(strategy.LoadFromExternal);
    }

    // ============================
    // ZeroInitializationStrategy: Properties
    // ============================

    [Fact]
    public void Zero_IsLazy_False()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        Assert.False(strategy.IsLazy);
    }

    [Fact]
    public void Zero_LoadFromExternal_False()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        Assert.False(strategy.LoadFromExternal);
    }

    // ============================
    // ZeroInitializationStrategy: Weights and Biases
    // ============================

    [Fact]
    public void Zero_InitializeWeights_AllZeros()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 5, 3 });
        strategy.InitializeWeights(weights, 3, 5);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(0.0, weights[i]);
        }
    }

    [Fact]
    public void Zero_InitializeBiases_AllZeros()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        var biases = new Tensor<double>(new[] { 10 });
        strategy.InitializeBiases(biases);

        for (int i = 0; i < biases.Length; i++)
        {
            Assert.Equal(0.0, biases[i]);
        }
    }

    [Fact]
    public void Zero_LargeTensor_AllZeros()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 100, 50 });
        strategy.InitializeWeights(weights, 50, 100);

        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            sum += Math.Abs(weights[i]);
        }
        Assert.Equal(0.0, sum);
    }

    // ============================
    // EagerInitializationStrategy: Xavier Normal Distribution
    // ============================

    [Fact]
    public void Eager_InitializeWeights_NonZero()
    {
        var strategy = new EagerInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 10, 5 });
        strategy.InitializeWeights(weights, 5, 10);

        // At least some weights should be non-zero
        bool hasNonZero = false;
        for (int i = 0; i < weights.Length; i++)
        {
            if (Math.Abs(weights[i]) > 1e-10)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void Eager_InitializeWeights_XavierNormal_MeanNearZero()
    {
        // Xavier Normal: W ~ N(0, sqrt(2/(fan_in + fan_out)))
        var strategy = new EagerInitializationStrategy<double>();
        int fanIn = 100, fanOut = 100;
        var weights = new Tensor<double>(new[] { fanOut, fanIn });
        strategy.InitializeWeights(weights, fanIn, fanOut);

        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            sum += weights[i];
        }
        double mean = sum / weights.Length;

        // Mean should be near zero (within reasonable tolerance for 10000 samples)
        Assert.True(Math.Abs(mean) < 0.05, $"Mean {mean} should be near zero");
    }

    [Fact]
    public void Eager_InitializeWeights_XavierNormal_VarianceCorrect()
    {
        // Xavier Normal: variance = 2/(fan_in + fan_out)
        var strategy = new EagerInitializationStrategy<double>();
        int fanIn = 200, fanOut = 200;
        double expectedVariance = 2.0 / (fanIn + fanOut); // = 0.005

        var weights = new Tensor<double>(new[] { fanOut, fanIn });
        strategy.InitializeWeights(weights, fanIn, fanOut);

        double sum = 0, sumSq = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            double val = weights[i];
            sum += val;
            sumSq += val * val;
        }
        double mean = sum / weights.Length;
        double variance = sumSq / weights.Length - mean * mean;

        // Variance should be close to expected (within 50% tolerance for statistical sampling)
        Assert.True(variance > expectedVariance * 0.5 && variance < expectedVariance * 1.5,
            $"Variance {variance} should be near {expectedVariance}");
    }

    [Fact]
    public void Eager_InitializeWeights_XavierNormal_TruncatedBound()
    {
        // Xavier Normal uses truncated normal: clips at [-2*stddev, 2*stddev]
        var strategy = new EagerInitializationStrategy<double>();
        int fanIn = 100, fanOut = 100;
        double stddev = Math.Sqrt(2.0 / (fanIn + fanOut));
        double clipBound = 2.0 * stddev;

        var weights = new Tensor<double>(new[] { fanOut, fanIn });
        strategy.InitializeWeights(weights, fanIn, fanOut);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(Math.Abs(weights[i]) <= clipBound + 1e-10,
                $"Weight {weights[i]} exceeds truncation bound {clipBound}");
        }
    }

    [Fact]
    public void Eager_InitializeBiases_AllZeros()
    {
        var strategy = new EagerInitializationStrategy<double>();
        var biases = new Tensor<double>(new[] { 20 });
        strategy.InitializeBiases(biases);

        for (int i = 0; i < biases.Length; i++)
        {
            Assert.Equal(0.0, biases[i]);
        }
    }

    // ============================
    // LazyInitializationStrategy: Same Math as Eager
    // ============================

    [Fact]
    public void Lazy_InitializeWeights_NonZero()
    {
        var strategy = new LazyInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 10, 5 });
        strategy.InitializeWeights(weights, 5, 10);

        bool hasNonZero = false;
        for (int i = 0; i < weights.Length; i++)
        {
            if (Math.Abs(weights[i]) > 1e-10)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void Lazy_InitializeBiases_AllZeros()
    {
        var strategy = new LazyInitializationStrategy<double>();
        var biases = new Tensor<double>(new[] { 15 });
        strategy.InitializeBiases(biases);

        for (int i = 0; i < biases.Length; i++)
        {
            Assert.Equal(0.0, biases[i]);
        }
    }

    [Fact]
    public void Lazy_InitializeWeights_XavierNormal_TruncatedBound()
    {
        var strategy = new LazyInitializationStrategy<double>();
        int fanIn = 50, fanOut = 50;
        double stddev = Math.Sqrt(2.0 / (fanIn + fanOut));
        double clipBound = 2.0 * stddev;

        var weights = new Tensor<double>(new[] { fanOut, fanIn });
        strategy.InitializeWeights(weights, fanIn, fanOut);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.True(Math.Abs(weights[i]) <= clipBound + 1e-10);
        }
    }

    // ============================
    // Xavier Uniform vs Normal: Different Distributions
    // ============================

    [Fact]
    public void Eager_DifferentFanIn_DifferentScale()
    {
        var strategy = new EagerInitializationStrategy<double>();

        // Small fan-in should produce larger weights
        var smallFanWeights = new Tensor<double>(new[] { 10, 5 });
        strategy.InitializeWeights(smallFanWeights, 5, 10);

        // Large fan-in should produce smaller weights
        var largeFanWeights = new Tensor<double>(new[] { 10, 500 });
        strategy.InitializeWeights(largeFanWeights, 500, 10);

        double smallVar = CalculateVariance(smallFanWeights);
        double largeVar = CalculateVariance(largeFanWeights);

        // Variance for small fan should be larger than for large fan
        // Xavier: var = 2/(fan_in + fan_out)
        // Small: 2/(5+10) = 0.133, Large: 2/(500+10) = 0.0039
        Assert.True(smallVar > largeVar,
            $"Small fan variance ({smallVar}) should exceed large fan variance ({largeVar})");
    }

    // ============================
    // InitializationStrategies Factory
    // ============================

    [Fact]
    public void Factory_Lazy_ReturnsLazyStrategy()
    {
        var strategy = InitializationStrategies<double>.Lazy;
        Assert.NotNull(strategy);
        Assert.True(strategy.IsLazy);
        Assert.False(strategy.LoadFromExternal);
    }

    [Fact]
    public void Factory_Eager_ReturnsEagerStrategy()
    {
        var strategy = InitializationStrategies<double>.Eager;
        Assert.NotNull(strategy);
        Assert.False(strategy.IsLazy);
        Assert.False(strategy.LoadFromExternal);
    }

    [Fact]
    public void Factory_Zero_ReturnsZeroStrategy()
    {
        var strategy = InitializationStrategies<double>.Zero;
        Assert.NotNull(strategy);
        Assert.False(strategy.IsLazy);
        Assert.False(strategy.LoadFromExternal);
    }

    [Fact]
    public void Factory_Lazy_Singleton()
    {
        var s1 = InitializationStrategies<double>.Lazy;
        var s2 = InitializationStrategies<double>.Lazy;
        Assert.Same(s1, s2);
    }

    [Fact]
    public void Factory_Eager_Singleton()
    {
        var s1 = InitializationStrategies<double>.Eager;
        var s2 = InitializationStrategies<double>.Eager;
        Assert.Same(s1, s2);
    }

    [Fact]
    public void Factory_Zero_Singleton()
    {
        var s1 = InitializationStrategies<double>.Zero;
        var s2 = InitializationStrategies<double>.Zero;
        Assert.Same(s1, s2);
    }

    [Fact]
    public void Factory_FromFile_ReturnsFromFileStrategy()
    {
        var strategy = InitializationStrategies<double>.FromFile("dummy_path.json");
        Assert.NotNull(strategy);
        Assert.True(strategy.LoadFromExternal);
    }

    // ============================
    // InitializationStrategyType Enum
    // ============================

    [Fact]
    public void InitializationStrategyType_HasFourValues()
    {
        var values = (((InitializationStrategyType[])Enum.GetValues(typeof(InitializationStrategyType))));
        Assert.Equal(4, values.Length);
    }

    [Theory]
    [InlineData(InitializationStrategyType.Eager)]
    [InlineData(InitializationStrategyType.Lazy)]
    [InlineData(InitializationStrategyType.Zero)]
    [InlineData(InitializationStrategyType.FromFile)]
    public void InitializationStrategyType_AllValuesValid(InitializationStrategyType type)
    {
        Assert.True(Enum.IsDefined(typeof(InitializationStrategyType), type));
    }

    // ============================
    // Float type support
    // ============================

    [Fact]
    public void Eager_FloatType_WorksCorrectly()
    {
        var strategy = new EagerInitializationStrategy<float>();
        var weights = new Tensor<float>(new[] { 5, 3 });
        strategy.InitializeWeights(weights, 3, 5);

        bool hasNonZero = false;
        for (int i = 0; i < weights.Length; i++)
        {
            if (Math.Abs(weights[i]) > 1e-10f)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero);
    }

    [Fact]
    public void Zero_FloatType_AllZeros()
    {
        var strategy = new ZeroInitializationStrategy<float>();
        var weights = new Tensor<float>(new[] { 5, 3 });
        strategy.InitializeWeights(weights, 3, 5);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(0.0f, weights[i]);
        }
    }

    // ============================
    // Edge Cases
    // ============================

    [Fact]
    public void Eager_SingleElement_Initializes()
    {
        var strategy = new EagerInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 1 });
        strategy.InitializeWeights(weights, 1, 1);

        // Should initialize without error
        // Value constrained by Xavier: stddev = sqrt(2/2) = 1.0, clip at 2.0
        Assert.True(Math.Abs(weights[0]) <= 2.0 + 1e-10);
    }

    [Fact]
    public void Eager_LargeTensor_AllInitialized()
    {
        var strategy = new EagerInitializationStrategy<double>();
        int rows = 256, cols = 256;
        var weights = new Tensor<double>(new[] { rows, cols });
        strategy.InitializeWeights(weights, cols, rows);

        // Check mean is near zero
        double sum = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            sum += weights[i];
        }
        double mean = sum / weights.Length;
        Assert.True(Math.Abs(mean) < 0.02,
            $"Mean {mean} should be near zero for {rows * cols} weights");
    }

    // ============================
    // Helper Methods
    // ============================

    private static double CalculateVariance(Tensor<double> tensor)
    {
        double sum = 0, sumSq = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double val = tensor[i];
            sum += val;
            sumSq += val * val;
        }
        double mean = sum / tensor.Length;
        return sumSq / tensor.Length - mean * mean;
    }
}
