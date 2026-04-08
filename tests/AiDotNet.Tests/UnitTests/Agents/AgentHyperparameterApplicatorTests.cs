using AiDotNet.Agents;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.Agents;

public class AgentHyperparameterApplicatorTests
{
    private readonly HyperparameterRegistry _registry = new();
    private readonly AgentHyperparameterApplicator<double> _applicator;

    public AgentHyperparameterApplicatorTests()
    {
        _applicator = new AgentHyperparameterApplicator<double>(_registry);
    }

    #region Test Helpers

    /// <summary>
    /// Simple options class for testing hyperparameter application.
    /// </summary>
    private class TestOptions : ModelOptions
    {
        public int NumberOfTrees { get; set; } = 100;
        public int MaxDepth { get; set; } = 10;
        public double LearningRate { get; set; } = 0.1;
        public double SubsampleRatio { get; set; } = 1.0;
        public bool UseIntercept { get; set; } = true;
        public string SomeStringParam { get; set; } = "default";
    }

    /// <summary>
    /// Simple model implementing IConfigurableModel for testing.
    /// </summary>
    private class TestModel : IConfigurableModel<double>
    {
        private readonly TestOptions _options;

        public TestModel(TestOptions options)
        {
            _options = options;
        }

        public ModelOptions GetOptions() => _options;
    }

    private static (TestModel model, TestOptions options) CreateTestModel()
    {
        var options = new TestOptions();
        var model = new TestModel(options);
        return (model, options);
    }

    #endregion

    #region Apply - Successful Application

    [Fact(Timeout = 60000)]
    public async Task Apply_KnownParameters_AppliesViaRegistry()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["n_estimators"] = 200,
            ["max_depth"] = 15
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(2, result.Applied.Count);
        Assert.Equal(200, options.NumberOfTrees);
        Assert.Equal(15, options.MaxDepth);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_DirectPropertyNames_AppliesWithoutRegistry()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["NumberOfTrees"] = 300,
            ["LearningRate"] = 0.05
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(300, options.NumberOfTrees);
        Assert.Equal(0.05, options.LearningRate);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_SharedParameter_AppliesSeed()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["random_seed"] = 42
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(42, options.Seed);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_TypeConversion_IntToDouble()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["LearningRate"] = 1  // int provided, double expected
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(1.0, options.LearningRate);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_TypeConversion_DoubleToInt()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["NumberOfTrees"] = 200.0  // double provided, int expected
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(200, options.NumberOfTrees);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_BoolParameter_SetsCorrectly()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["UseIntercept"] = false
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.False(options.UseIntercept);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_StringParameter_SetsCorrectly()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["SomeStringParam"] = "custom_value"
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal("custom_value", options.SomeStringParam);
    }

    #endregion

    #region Apply - Skipped Parameters

    [Fact(Timeout = 60000)]
    public async Task Apply_UnknownParameter_Skipped()
    {
        var (model, _) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["completely_unknown_param"] = 42
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.False(result.HasAppliedParameters);
        Assert.Single(result.Skipped);
        Assert.Equal(42, result.Skipped["completely_unknown_param"]);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_MixedKnownAndUnknown_ReportsBoth()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["n_estimators"] = 200,
            ["unknown_param"] = "value"
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.Single(result.Applied);
        Assert.Single(result.Skipped);
        Assert.Equal(200, options.NumberOfTrees);
    }

    #endregion

    #region Apply - Validation Warnings

    [Fact(Timeout = 60000)]
    public async Task Apply_OutOfRangeValue_AddsWarningButStillApplies()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["n_estimators"] = 50000  // Above max of 10000
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(50000, options.NumberOfTrees);
        Assert.Single(result.Warnings);
        Assert.Contains("above", result.Warnings[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task Apply_BelowRangeValue_AddsWarningButStillApplies()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["n_estimators"] = 0  // Below min of 1
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(0, options.NumberOfTrees);
        Assert.Single(result.Warnings);
        Assert.Contains("below", result.Warnings[0]);
    }

    #endregion

    #region Apply - Empty Inputs

    [Fact(Timeout = 60000)]
    public async Task Apply_EmptyHyperparameters_ReturnsNoResults()
    {
        var (model, _) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>();

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.False(result.HasAppliedParameters);
        Assert.Empty(result.Applied);
        Assert.Empty(result.Skipped);
        Assert.Empty(result.Failed);
    }

    #endregion

    #region ConvertValue

    [Theory]
    [InlineData(42, typeof(double), 42.0)]
    [InlineData(42, typeof(float), 42.0f)]
    [InlineData(42.5, typeof(int), 42)]
    [InlineData(42L, typeof(int), 42)]
    [InlineData(true, typeof(bool), true)]
    [InlineData("hello", typeof(string), "hello")]
    public void ConvertValue_ValidConversions_ReturnsCorrectType(object input, Type targetType, object expected)
    {
        var result = AgentHyperparameterApplicator<double>.ConvertValue(input, targetType);

        Assert.NotNull(result);
        Assert.IsType(expected.GetType(), result);
        Assert.Equal(expected, result);
    }

    [Fact(Timeout = 60000)]
    public async Task ConvertValue_NullInput_ReturnsNull()
    {
        var result = AgentHyperparameterApplicator<double>.ConvertValue(null!, typeof(int));

        Assert.Null(result);
    }

    [Fact(Timeout = 60000)]
    public async Task ConvertValue_NullableInt_ConvertsToUnderlyingType()
    {
        var result = AgentHyperparameterApplicator<double>.ConvertValue(42, typeof(int?));

        Assert.NotNull(result);
        Assert.Equal(42, result);
    }

    [Fact(Timeout = 60000)]
    public async Task ConvertValue_AlreadyCorrectType_ReturnsAsIs()
    {
        var result = AgentHyperparameterApplicator<double>.ConvertValue(42, typeof(int));

        Assert.Equal(42, result);
    }

    [Fact(Timeout = 60000)]
    public async Task ConvertValue_InvalidConversion_ReturnsNull()
    {
        var result = AgentHyperparameterApplicator<double>.ConvertValue("not_a_number", typeof(int));

        Assert.Null(result);
    }

    #endregion

    #region HyperparameterApplicationResult

    [Fact(Timeout = 60000)]
    public async Task Result_GetSummary_IncludesAllSections()
    {
        var (model, _) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["n_estimators"] = 200,
            ["unknown_param"] = "value"
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);
        var summary = result.GetSummary();

        Assert.Contains("Applied", summary);
        Assert.Contains("Skipped", summary);
        Assert.Contains("n_estimators", summary);
        Assert.Contains("unknown_param", summary);
    }

    [Fact(Timeout = 60000)]
    public async Task Result_GetSummary_EmptyResult_ShowsNoParameters()
    {
        var result = new AiDotNet.Models.HyperparameterApplicationResult();
        var summary = result.GetSummary();

        Assert.Contains("No hyperparameters were processed", summary);
    }

    #endregion

    #region Case-Insensitive Property Matching

    [Fact(Timeout = 60000)]
    public async Task Apply_CaseInsensitivePropertyName_Works()
    {
        var (model, options) = CreateTestModel();
        var hyperparams = new Dictionary<string, object>
        {
            ["numberoftrees"] = 250  // lowercase version of NumberOfTrees
        };

        var result = _applicator.Apply(model, typeof(AiDotNet.Regression.RandomForestRegression<>), hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(250, options.NumberOfTrees);
    }

    #endregion
}
