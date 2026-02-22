using AiDotNet.Initialization;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Initialization;

/// <summary>
/// Integration tests for initialization strategy classes.
/// </summary>
public class InitializationIntegrationTests
{
    #region EagerInitializationStrategy Tests

    [Fact]
    public void Eager_IsNotLazy()
    {
        var strategy = new EagerInitializationStrategy<double>();
        Assert.False(strategy.IsLazy);
        Assert.False(strategy.LoadFromExternal);
    }

    [Fact]
    public void Eager_InitializeWeights_ProducesNonZeroValues()
    {
        var strategy = new EagerInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 4, 3 });
        strategy.InitializeWeights(weights, 3, 4);

        // Xavier initialization should produce non-zero values
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
    public void Eager_InitializeBiases_ProducesZeros()
    {
        var strategy = new EagerInitializationStrategy<double>();
        var biases = new Tensor<double>(new[] { 5 });
        // Set some values first
        biases[0] = 99.0;
        strategy.InitializeBiases(biases);

        for (int i = 0; i < biases.Length; i++)
        {
            Assert.Equal(0.0, biases[i], 1e-10);
        }
    }

    #endregion

    #region LazyInitializationStrategy Tests

    [Fact]
    public void Lazy_IsLazy()
    {
        var strategy = new LazyInitializationStrategy<double>();
        Assert.True(strategy.IsLazy);
        Assert.False(strategy.LoadFromExternal);
    }

    [Fact]
    public void Lazy_InitializeWeights_StillProducesValues()
    {
        var strategy = new LazyInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 3, 3 });
        strategy.InitializeWeights(weights, 3, 3);

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

    #endregion

    #region ZeroInitializationStrategy Tests

    [Fact]
    public void Zero_IsNotLazy()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        Assert.False(strategy.IsLazy);
        Assert.False(strategy.LoadFromExternal);
    }

    [Fact]
    public void Zero_InitializeWeights_AllZero()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        var weights = new Tensor<double>(new[] { 4, 3 });

        // Set non-zero values first
        for (int i = 0; i < weights.Length; i++)
            weights[i] = 99.0;

        strategy.InitializeWeights(weights, 3, 4);

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(0.0, weights[i], 1e-10);
        }
    }

    [Fact]
    public void Zero_InitializeBiases_AllZero()
    {
        var strategy = new ZeroInitializationStrategy<double>();
        var biases = new Tensor<double>(new[] { 5 });
        biases[0] = 42.0;
        strategy.InitializeBiases(biases);

        for (int i = 0; i < biases.Length; i++)
        {
            Assert.Equal(0.0, biases[i], 1e-10);
        }
    }

    #endregion

    #region FromFileInitializationStrategy Tests

    [Fact]
    public void FromFile_EmptyPath_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new FromFileInitializationStrategy<double>(""));
        Assert.Throws<ArgumentException>(() => new FromFileInitializationStrategy<double>("   "));
    }

    [Fact]
    public void FromFile_IsNotLazy()
    {
        var strategy = new FromFileInitializationStrategy<double>("test.json");
        Assert.False(strategy.IsLazy);
        Assert.True(strategy.LoadFromExternal);
    }

    [Fact]
    public void FromFile_NonexistentFile_ThrowsOnInitialize()
    {
        var strategy = new FromFileInitializationStrategy<double>("nonexistent_weights.json");
        var weights = new Tensor<double>(new[] { 3, 3 });

        Assert.Throws<FileNotFoundException>(() => strategy.InitializeWeights(weights, 3, 3));
    }

    [Fact]
    public void FromFile_ValidJsonFile_LoadsWeights()
    {
        var tempFile = Path.GetTempFileName() + ".json";
        try
        {
            // Create a test weights file
            var data = new
            {
                weights = new Dictionary<string, double[]>
                {
                    ["weights_0"] = new double[] { 1.0, 2.0, 3.0, 4.0 }
                },
                biases = new Dictionary<string, double[]>
                {
                    ["biases_0"] = new double[] { 0.1, 0.2 }
                }
            };
            File.WriteAllText(tempFile, JsonConvert.SerializeObject(data));

            var strategy = new FromFileInitializationStrategy<double>(tempFile);
            var weights = new Tensor<double>(new[] { 4 });
            strategy.InitializeWeights(weights, 2, 2);

            Assert.Equal(1.0, weights[0], 1e-10);
            Assert.Equal(2.0, weights[1], 1e-10);
            Assert.Equal(3.0, weights[2], 1e-10);
            Assert.Equal(4.0, weights[3], 1e-10);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromFile_Reset_AllowsReinitialization()
    {
        var tempFile = Path.GetTempFileName() + ".json";
        try
        {
            var data = new
            {
                weights = new Dictionary<string, double[]>
                {
                    ["weights_0"] = new double[] { 1.0, 2.0 }
                },
                biases = new Dictionary<string, double[]>()
            };
            File.WriteAllText(tempFile, JsonConvert.SerializeObject(data));

            var strategy = new FromFileInitializationStrategy<double>(tempFile);
            var weights1 = new Tensor<double>(new[] { 2 });
            strategy.InitializeWeights(weights1, 1, 2);

            strategy.Reset();
            var weights2 = new Tensor<double>(new[] { 2 });
            strategy.InitializeWeights(weights2, 1, 2);

            Assert.Equal(1.0, weights2[0], 1e-10);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    #endregion

    #region InitializationStrategies Factory Tests

    [Fact]
    public void Strategies_Lazy_ReturnsLazyStrategy()
    {
        var strategy = InitializationStrategies<double>.Lazy;
        Assert.NotNull(strategy);
        Assert.True(strategy.IsLazy);
    }

    [Fact]
    public void Strategies_Eager_ReturnsEagerStrategy()
    {
        var strategy = InitializationStrategies<double>.Eager;
        Assert.NotNull(strategy);
        Assert.False(strategy.IsLazy);
    }

    [Fact]
    public void Strategies_Zero_ReturnsZeroStrategy()
    {
        var strategy = InitializationStrategies<double>.Zero;
        Assert.NotNull(strategy);
        Assert.False(strategy.IsLazy);
    }

    [Fact]
    public void Strategies_Singletons_AreSameInstance()
    {
        var lazy1 = InitializationStrategies<double>.Lazy;
        var lazy2 = InitializationStrategies<double>.Lazy;
        Assert.Same(lazy1, lazy2);

        var eager1 = InitializationStrategies<double>.Eager;
        var eager2 = InitializationStrategies<double>.Eager;
        Assert.Same(eager1, eager2);
    }

    [Fact]
    public void Strategies_FromFile_CreatesNewInstance()
    {
        var s1 = InitializationStrategies<double>.FromFile("a.json");
        var s2 = InitializationStrategies<double>.FromFile("b.json");
        Assert.NotSame(s1, s2);
    }

    #endregion
}
