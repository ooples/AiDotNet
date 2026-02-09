using AiDotNet.Agents;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Agents;

public class HyperparameterRegistryTests
{
    private readonly HyperparameterRegistry _registry = new();

    #region Tree Model Lookups

    [Theory]
    [InlineData("n_estimators", "NumberOfTrees")]
    [InlineData("num_trees", "NumberOfTrees")]
    [InlineData("ntrees", "NumberOfTrees")]
    [InlineData("number_of_trees", "NumberOfTrees")]
    [InlineData("max_depth", "MaxDepth")]
    [InlineData("maxdepth", "MaxDepth")]
    [InlineData("tree_depth", "MaxDepth")]
    [InlineData("min_samples_split", "MinSamplesSplit")]
    public void GetPropertyName_RandomForest_ReturnsCorrectMapping(string llmName, string expectedProperty)
    {
        var result = _registry.GetPropertyName(ModelType.RandomForest, llmName);

        Assert.Equal(expectedProperty, result);
    }

    [Theory]
    [InlineData("learning_rate", "LearningRate")]
    [InlineData("lr", "LearningRate")]
    [InlineData("eta", "LearningRate")]
    [InlineData("subsample", "SubsampleRatio")]
    [InlineData("subsample_ratio", "SubsampleRatio")]
    public void GetPropertyName_GradientBoosting_ReturnsCorrectMapping(string llmName, string expectedProperty)
    {
        var result = _registry.GetPropertyName(ModelType.GradientBoosting, llmName);

        Assert.Equal(expectedProperty, result);
    }

    [Fact]
    public void GetPropertyName_DecisionTree_HasMaxDepthAndMinSamples()
    {
        Assert.Equal("MaxDepth", _registry.GetPropertyName(ModelType.DecisionTree, "max_depth"));
        Assert.Equal("MinSamplesSplit", _registry.GetPropertyName(ModelType.DecisionTree, "min_samples_split"));
    }

    #endregion

    #region Neural Network Lookups

    [Theory]
    [InlineData("learning_rate", "LearningRate")]
    [InlineData("epochs", "Epochs")]
    [InlineData("num_epochs", "Epochs")]
    [InlineData("batch_size", "BatchSize")]
    [InlineData("batchsize", "BatchSize")]
    public void GetPropertyName_NeuralNetwork_ReturnsCorrectMapping(string llmName, string expectedProperty)
    {
        var result = _registry.GetPropertyName(ModelType.NeuralNetworkRegression, llmName);

        Assert.Equal(expectedProperty, result);
    }

    #endregion

    #region Linear Model Lookups

    [Theory]
    [InlineData("degree", "Degree")]
    [InlineData("polynomial_degree", "Degree")]
    public void GetPropertyName_PolynomialRegression_ReturnsCorrectMapping(string llmName, string expectedProperty)
    {
        var result = _registry.GetPropertyName(ModelType.PolynomialRegression, llmName);

        Assert.Equal(expectedProperty, result);
    }

    [Theory]
    [InlineData("alpha", "Alpha")]
    [InlineData("regularization", "Alpha")]
    [InlineData("lambda", "Alpha")]
    public void GetPropertyName_RidgeRegression_ReturnsCorrectMapping(string llmName, string expectedProperty)
    {
        var result = _registry.GetPropertyName(ModelType.RidgeRegression, llmName);

        Assert.Equal(expectedProperty, result);
    }

    #endregion

    #region Neighbor/Kernel Model Lookups

    [Theory]
    [InlineData("n_neighbors", "K")]
    [InlineData("k", "K")]
    [InlineData("num_neighbors", "K")]
    public void GetPropertyName_KNN_ReturnsCorrectMapping(string llmName, string expectedProperty)
    {
        var result = _registry.GetPropertyName(ModelType.KNearestNeighbors, llmName);

        Assert.Equal(expectedProperty, result);
    }

    [Fact]
    public void GetPropertyName_SVR_ReturnsCAndEpsilon()
    {
        Assert.Equal("C", _registry.GetPropertyName(ModelType.SupportVectorRegression, "C"));
        Assert.Equal("C", _registry.GetPropertyName(ModelType.SupportVectorRegression, "cost"));
        Assert.Equal("Epsilon", _registry.GetPropertyName(ModelType.SupportVectorRegression, "epsilon"));
        Assert.Equal("Epsilon", _registry.GetPropertyName(ModelType.SupportVectorRegression, "eps"));
    }

    #endregion

    #region Time Series Lookups

    [Fact]
    public void GetPropertyName_TimeSeries_ReturnsLagAndSeasonal()
    {
        Assert.Equal("LagOrder", _registry.GetPropertyName(ModelType.TimeSeriesRegression, "lag_order"));
        Assert.Equal("LagOrder", _registry.GetPropertyName(ModelType.TimeSeriesRegression, "lags"));
        Assert.Equal("SeasonalPeriod", _registry.GetPropertyName(ModelType.TimeSeriesRegression, "seasonal_period"));
        Assert.Equal("SeasonalPeriod", _registry.GetPropertyName(ModelType.TimeSeriesRegression, "seasonality"));
    }

    #endregion

    #region Shared Parameters

    [Theory]
    [InlineData(ModelType.RandomForest)]
    [InlineData(ModelType.NeuralNetworkRegression)]
    [InlineData(ModelType.KNearestNeighbors)]
    [InlineData(ModelType.PolynomialRegression)]
    public void GetPropertyName_SharedSeed_WorksAcrossModelTypes(ModelType modelType)
    {
        Assert.Equal("Seed", _registry.GetPropertyName(modelType, "seed"));
        Assert.Equal("Seed", _registry.GetPropertyName(modelType, "random_seed"));
        Assert.Equal("Seed", _registry.GetPropertyName(modelType, "random_state"));
    }

    [Fact]
    public void GetPropertyName_SharedUseIntercept_Works()
    {
        Assert.Equal("UseIntercept", _registry.GetPropertyName(ModelType.RidgeRegression, "use_intercept"));
        Assert.Equal("UseIntercept", _registry.GetPropertyName(ModelType.RidgeRegression, "fit_intercept"));
    }

    #endregion

    #region Unknown Parameters

    [Fact]
    public void GetPropertyName_UnknownParameter_ReturnsNull()
    {
        var result = _registry.GetPropertyName(ModelType.RandomForest, "nonexistent_param");

        Assert.Null(result);
    }

    [Fact]
    public void GetDefinition_UnknownParameter_ReturnsNull()
    {
        var result = _registry.GetDefinition(ModelType.RandomForest, "nonexistent_param");

        Assert.Null(result);
    }

    #endregion

    #region Alias Normalization

    [Theory]
    [InlineData("N_Estimators")]
    [InlineData("n_estimators")]
    [InlineData("N_ESTIMATORS")]
    [InlineData("nestimators")]
    public void GetPropertyName_CaseInsensitiveNormalized_ReturnsCorrect(string alias)
    {
        var result = _registry.GetPropertyName(ModelType.RandomForest, alias);

        Assert.Equal("NumberOfTrees", result);
    }

    #endregion

    #region Validation

    [Fact]
    public void Validate_InRange_ReturnsValid()
    {
        var result = _registry.Validate(ModelType.RandomForest, "n_estimators", 100);

        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    [Fact]
    public void Validate_BelowMinimum_ReturnsWarning()
    {
        var result = _registry.Validate(ModelType.RandomForest, "n_estimators", 0);

        Assert.True(result.IsValid);
        Assert.True(result.HasWarning);
        Assert.Contains("below", result.Warning!);
    }

    [Fact]
    public void Validate_AboveMaximum_ReturnsWarning()
    {
        var result = _registry.Validate(ModelType.RandomForest, "n_estimators", 50000);

        Assert.True(result.IsValid);
        Assert.True(result.HasWarning);
        Assert.Contains("above", result.Warning!);
    }

    [Fact]
    public void Validate_UnknownParameter_ReturnsValid()
    {
        var result = _registry.Validate(ModelType.RandomForest, "unknown_param", 42);

        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    [Fact]
    public void Validate_NonNumericValue_ReturnsValid()
    {
        var result = _registry.Validate(ModelType.RandomForest, "n_estimators", "not_a_number");

        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    #endregion

    #region Custom Registration

    [Fact]
    public void Register_CustomDefinition_IsAccessible()
    {
        _registry.Register(ModelType.RandomForest, new HyperparameterDefinition
        {
            PropertyName = "CustomParam",
            Aliases = new List<string> { "custom_param", "my_param" },
            ValueType = typeof(double),
            MinValue = 0,
            MaxValue = 100
        });

        Assert.Equal("CustomParam", _registry.GetPropertyName(ModelType.RandomForest, "custom_param"));
        Assert.Equal("CustomParam", _registry.GetPropertyName(ModelType.RandomForest, "my_param"));
    }

    [Fact]
    public void Register_CustomDefinition_ValidationWorks()
    {
        _registry.Register(ModelType.RandomForest, new HyperparameterDefinition
        {
            PropertyName = "TestParam",
            Aliases = new List<string> { "test_param" },
            ValueType = typeof(int),
            MinValue = 1,
            MaxValue = 10
        });

        var valid = _registry.Validate(ModelType.RandomForest, "test_param", 5);
        var tooLow = _registry.Validate(ModelType.RandomForest, "test_param", 0);
        var tooHigh = _registry.Validate(ModelType.RandomForest, "test_param", 20);

        Assert.False(valid.HasWarning);
        Assert.True(tooLow.HasWarning);
        Assert.True(tooHigh.HasWarning);
    }

    #endregion

    #region GetDefinition

    [Fact]
    public void GetDefinition_KnownParameter_ReturnsFullDefinition()
    {
        var def = _registry.GetDefinition(ModelType.RandomForest, "n_estimators");

        Assert.NotNull(def);
        Assert.Equal("NumberOfTrees", def.PropertyName);
        Assert.Equal(typeof(int), def.ValueType);
        Assert.Equal(1, def.MinValue);
        Assert.Equal(10000, def.MaxValue);
    }

    [Fact]
    public void GetDefinition_SharedParameter_ReturnsDefinition()
    {
        var def = _registry.GetDefinition(ModelType.RandomForest, "seed");

        Assert.NotNull(def);
        Assert.Equal("Seed", def.PropertyName);
    }

    #endregion
}
