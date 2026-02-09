using AiDotNet.Agents;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Agents;

/// <summary>
/// Integration tests for the full hyperparameter auto-apply pipeline:
/// LLM response text -> HyperparameterResponseParser -> HyperparameterRegistry -> AgentHyperparameterApplicator -> real model options.
///
/// These tests use real model classes (DecisionTreeRegression, GradientBoostingRegression, etc.)
/// to verify that hyperparameters are actually applied to the correct options properties end-to-end.
/// </summary>
public class HyperparameterAutoApplyIntegrationTests
{
    #region Full Pipeline: Parse -> Registry -> Apply to Real Models

    [Fact]
    public void FullPipeline_DecisionTree_JsonResponse_AppliesMaxDepthAndMinSamples()
    {
        // Arrange: Simulate an LLM response with JSON hyperparameters
        var llmResponse = @"Based on your dataset analysis, I recommend:

```json
{
    ""max_depth"": 15,
    ""min_samples_split"": 5,
    ""seed"": 42
}
```

These values should prevent overfitting while capturing the key patterns.";

        var options = new DecisionTreeOptions();
        var model = new DecisionTreeRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        // Act: Parse LLM response and apply to model
        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.DecisionTree, hyperparams);

        // Assert: Verify hyperparameters were applied to the real options object
        Assert.True(result.HasAppliedParameters);
        Assert.Equal(15, options.MaxDepth);
        Assert.Equal(5, options.MinSamplesSplit);
        Assert.Equal(42, options.Seed);
        Assert.Empty(result.Failed);
    }

    [Fact]
    public void FullPipeline_GradientBoosting_MarkdownResponse_AppliesTreeAndBoostingParams()
    {
        // Arrange: Simulate an LLM response with markdown bold hyperparameters
        var llmResponse = @"For your gradient boosting model, I recommend:

- **n_estimators:** 200
- **learning_rate:** 0.05
- **max_depth:** 8
- **subsample:** 0.8";

        var options = new GradientBoostingRegressionOptions();
        var model = new GradientBoostingRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        // Act
        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.GradientBoosting, hyperparams);

        // Assert: All 4 parameters should be applied
        Assert.True(result.HasAppliedParameters);
        Assert.Equal(200, options.NumberOfTrees);
        Assert.Equal(0.05, options.LearningRate);
        Assert.Equal(8, options.MaxDepth);
        Assert.Equal(0.8, options.SubsampleRatio);
        Assert.Empty(result.Failed);
        Assert.Empty(result.Skipped);
    }

    [Fact]
    public void FullPipeline_RandomForest_ColonResponse_AppliesParameters()
    {
        // Arrange: Simulate an LLM response with colon-separated hyperparameters
        var llmResponse = @"Recommended settings for your random forest:
n_estimators: 500
max_depth: 12
min_samples_split: 10";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        // Act
        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        // Assert
        Assert.True(result.HasAppliedParameters);
        Assert.Equal(500, options.NumberOfTrees);
        Assert.Equal(12, options.MaxDepth);
        Assert.Equal(10, options.MinSamplesSplit);
    }

    [Fact]
    public void FullPipeline_KNN_AppliesKParameter()
    {
        // Arrange
        var llmResponse = @"```json
{""n_neighbors"": 7, ""random_seed"": 123}
```";

        var options = new KNearestNeighborsOptions();
        var model = new KNearestNeighborsRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        // Act
        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.KNearestNeighbors, hyperparams);

        // Assert
        Assert.True(result.HasAppliedParameters);
        Assert.Equal(7, options.K);
        Assert.Equal(123, options.Seed);
    }

    [Fact]
    public void FullPipeline_SVR_AppliesCAndEpsilon()
    {
        // Arrange
        var llmResponse = @"```json
{""C"": 10.0, ""epsilon"": 0.05}
```";

        var options = new SupportVectorRegressionOptions();
        var model = new SupportVectorRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        // Act
        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.SupportVectorRegression, hyperparams);

        // Assert
        Assert.True(result.HasAppliedParameters);
        Assert.Equal(10.0, options.C);
        Assert.Equal(0.05, options.Epsilon);
    }

    #endregion

    #region IConfigurableModel: Verify GetOptions Returns Live Options

    [Fact]
    public void GetOptions_DecisionTree_ReturnsMutableOptions()
    {
        // Arrange
        var options = new DecisionTreeOptions { MaxDepth = 5 };
        var model = new DecisionTreeRegression<double>(options);

        // Act: Modify options via GetOptions
        var configModel = (IConfigurableModel<double>)model;
        var returnedOptions = configModel.GetOptions();
        var dtOptions = returnedOptions as DecisionTreeOptions;

        // Assert: GetOptions returns the same instance
        Assert.NotNull(dtOptions);
        Assert.Equal(5, dtOptions.MaxDepth);

        // Mutate via returned options
        dtOptions.MaxDepth = 20;
        Assert.Equal(20, options.MaxDepth); // Original options should reflect the change
    }

    [Fact]
    public void GetOptions_GradientBoosting_WithExplicitOptions_ReturnsDerivedType()
    {
        // Arrange: Pass explicit options so base and derived both have the same object
        var options = new GradientBoostingRegressionOptions
        {
            NumberOfTrees = 100,
            LearningRate = 0.1,
            MaxDepth = 10
        };
        var model = new GradientBoostingRegression<double>(options);

        // Act
        var configModel = (IConfigurableModel<double>)model;
        var returnedOptions = configModel.GetOptions();

        // Assert: Should return GradientBoostingRegressionOptions, not just DecisionTreeOptions
        Assert.IsType<GradientBoostingRegressionOptions>(returnedOptions);
        var gbOptions = (GradientBoostingRegressionOptions)returnedOptions;
        Assert.Equal(100, gbOptions.NumberOfTrees);
        Assert.Equal(0.1, gbOptions.LearningRate);
    }

    [Fact]
    public void GetOptions_RandomForest_WithExplicitOptions_ReturnsDerivedType()
    {
        var options = new RandomForestRegressionOptions
        {
            NumberOfTrees = 200,
            MaxDepth = 15
        };
        var model = new RandomForestRegression<double>(options);

        var configModel = (IConfigurableModel<double>)model;
        var returnedOptions = configModel.GetOptions();

        Assert.IsType<RandomForestRegressionOptions>(returnedOptions);
        var rfOptions = (RandomForestRegressionOptions)returnedOptions;
        Assert.Equal(200, rfOptions.NumberOfTrees);
    }

    [Fact]
    public void GetOptions_KNN_WithExplicitOptions_ReturnsDerivedType()
    {
        var options = new KNearestNeighborsOptions { K = 3 };
        var model = new KNearestNeighborsRegression<double>(options);

        var configModel = (IConfigurableModel<double>)model;
        var returnedOptions = configModel.GetOptions();

        Assert.IsType<KNearestNeighborsOptions>(returnedOptions);
        Assert.Equal(3, ((KNearestNeighborsOptions)returnedOptions).K);
    }

    #endregion

    #region GetOptions Bug Detection: Null Options Constructor

    [Fact]
    public void GetOptions_GradientBoosting_WithNullOptions_ReturnsModelOptions()
    {
        // Verify GetOptions() returns a valid ModelOptions when constructed with null options.
        // The base class creates default options in this case.
        var model = new GradientBoostingRegression<double>(null);
        var configModel = (IConfigurableModel<double>)model;
        var returnedOptions = configModel.GetOptions();

        Assert.NotNull(returnedOptions);
        Assert.IsAssignableFrom<ModelOptions>(returnedOptions);

        // Seed should be settable on the returned options regardless of concrete type
        returnedOptions.Seed = 42;
        Assert.Equal(42, returnedOptions.Seed);
    }

    [Fact]
    public void GetOptions_KNN_WithNullOptions_ReturnsModelOptions()
    {
        // Verify GetOptions() returns a valid ModelOptions when constructed with null options.
        var model = new KNearestNeighborsRegression<double>(null);
        var configModel = (IConfigurableModel<double>)model;
        var returnedOptions = configModel.GetOptions();

        Assert.NotNull(returnedOptions);
        Assert.IsAssignableFrom<ModelOptions>(returnedOptions);

        // Seed should be settable on the returned options regardless of concrete type
        returnedOptions.Seed = 123;
        Assert.Equal(123, returnedOptions.Seed);
    }

    #endregion

    #region Validation Warnings Through Full Pipeline

    [Fact]
    public void FullPipeline_OutOfRangeValues_GeneratesWarningsButStillApplies()
    {
        var llmResponse = @"```json
{""n_estimators"": 50000, ""max_depth"": 200}
```";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        // Values should still be applied (warnings, not errors)
        Assert.True(result.HasAppliedParameters);
        Assert.Equal(50000, options.NumberOfTrees);
        Assert.Equal(200, options.MaxDepth);

        // But warnings should be generated
        Assert.True(result.Warnings.Count >= 2);
        Assert.Contains(result.Warnings, w => w.Contains("above"));
    }

    #endregion

    #region Mixed Known and Unknown Parameters Through Pipeline

    [Fact]
    public void FullPipeline_MixedParameters_ReportsAppliedAndSkipped()
    {
        var llmResponse = @"```json
{
    ""n_estimators"": 300,
    ""max_depth"": 10,
    ""completely_made_up_param"": 42,
    ""another_unknown"": ""value""
}
```";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        // Known params applied
        Assert.Equal(2, result.Applied.Count);
        Assert.Equal(300, options.NumberOfTrees);
        Assert.Equal(10, options.MaxDepth);

        // Unknown params skipped
        Assert.Equal(2, result.Skipped.Count);
        Assert.Contains("completely_made_up_param", result.Skipped.Keys);
        Assert.Contains("another_unknown", result.Skipped.Keys);
    }

    #endregion

    #region Type Conversion Through Pipeline

    [Fact]
    public void FullPipeline_StringValuesFromParser_ConvertedToCorrectTypes()
    {
        // Colon-separated parsing extracts values as typed via InferTypedValue
        var llmResponse = @"Settings:
max_depth: 8
n_estimators: 150";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);

        // Verify parser extracted correct types
        Assert.IsType<int>(hyperparams["max_depth"]);
        Assert.IsType<int>(hyperparams["n_estimators"]);

        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(8, options.MaxDepth);
        Assert.Equal(150, options.NumberOfTrees);
    }

    [Fact]
    public void FullPipeline_DoubleValuesForIntProperties_ConvertedCorrectly()
    {
        // JSON parser may return doubles for integer values when they have decimal points
        var llmResponse = @"```json
{""max_depth"": 8.0, ""n_estimators"": 200.0}
```";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(8, options.MaxDepth);
        Assert.Equal(200, options.NumberOfTrees);
    }

    [Fact]
    public void FullPipeline_IntValuesForDoubleProperties_ConvertedCorrectly()
    {
        // LLM might say learning_rate: 1 (int) when the property expects double
        var llmResponse = @"```json
{""learning_rate"": 1, ""subsample"": 1}
```";

        var options = new GradientBoostingRegressionOptions();
        var model = new GradientBoostingRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.GradientBoosting, hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(1.0, options.LearningRate);
        Assert.Equal(1.0, options.SubsampleRatio);
    }

    #endregion

    #region Seed Application Across Model Families

    [Fact]
    public void FullPipeline_SeedApplied_AcrossModelFamilies()
    {
        var llmResponse = @"```json
{""random_seed"": 42}
```";

        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var hyperparams = parser.Parse(llmResponse);

        // Test on DecisionTree
        var dtOptions = new DecisionTreeOptions();
        var dtModel = new DecisionTreeRegression<double>(dtOptions);
        var dtApplicator = new AgentHyperparameterApplicator<double>(registry);
        var dtResult = dtApplicator.Apply(dtModel, ModelType.DecisionTree, hyperparams);
        Assert.Equal(42, dtOptions.Seed);
        Assert.True(dtResult.HasAppliedParameters);

        // Test on GradientBoosting
        var gbOptions = new GradientBoostingRegressionOptions();
        var gbModel = new GradientBoostingRegression<double>(gbOptions);
        var gbApplicator = new AgentHyperparameterApplicator<double>(registry);
        var gbResult = gbApplicator.Apply(gbModel, ModelType.GradientBoosting, hyperparams);
        Assert.Equal(42, gbOptions.Seed);
        Assert.True(gbResult.HasAppliedParameters);

        // Test on KNN
        var knnOptions = new KNearestNeighborsOptions();
        var knnModel = new KNearestNeighborsRegression<double>(knnOptions);
        var knnApplicator = new AgentHyperparameterApplicator<double>(registry);
        var knnResult = knnApplicator.Apply(knnModel, ModelType.KNearestNeighbors, hyperparams);
        Assert.Equal(42, knnOptions.Seed);
        Assert.True(knnResult.HasAppliedParameters);
    }

    #endregion

    #region Result Summary Reporting

    [Fact]
    public void FullPipeline_ResultSummary_ContainsAllSections()
    {
        var llmResponse = @"```json
{
    ""n_estimators"": 200,
    ""max_depth"": 99999,
    ""unknown_param"": ""value""
}
```";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);
        var summary = result.GetSummary();

        // Summary should mention applied params
        Assert.Contains("Applied", summary);
        Assert.Contains("n_estimators", summary);

        // Summary should mention skipped params
        Assert.Contains("Skipped", summary);
        Assert.Contains("unknown_param", summary);

        // Summary should mention warnings for out-of-range values
        Assert.Contains("Warnings", summary);
    }

    #endregion

    #region Registry-to-Options Property Name Verification

    [Fact]
    public void Registry_RandomForest_AllRegisteredAliases_MapToRealProperties()
    {
        // Verify that every alias registered for RandomForest actually maps to a real property
        // on RandomForestRegressionOptions (or its parent DecisionTreeOptions)
        var registry = new HyperparameterRegistry();
        var optionsType = typeof(RandomForestRegressionOptions);

        var aliases = new Dictionary<string, string>
        {
            ["n_estimators"] = "NumberOfTrees",
            ["num_trees"] = "NumberOfTrees",
            ["max_depth"] = "MaxDepth",
            ["maxdepth"] = "MaxDepth",
            ["min_samples_split"] = "MinSamplesSplit"
        };

        foreach (var (alias, expectedProp) in aliases)
        {
            var propertyName = registry.GetPropertyName(ModelType.RandomForest, alias);
            Assert.NotNull(propertyName);
            Assert.Equal(expectedProp, propertyName);

            // Verify the property actually exists on the options class
            var prop = optionsType.GetProperty(propertyName);
            Assert.NotNull(prop);
            Assert.True(prop.CanWrite, $"Property {propertyName} must be writable");
        }
    }

    [Fact]
    public void Registry_GradientBoosting_AllRegisteredAliases_MapToRealProperties()
    {
        var registry = new HyperparameterRegistry();
        var optionsType = typeof(GradientBoostingRegressionOptions);

        var aliases = new Dictionary<string, string>
        {
            ["learning_rate"] = "LearningRate",
            ["lr"] = "LearningRate",
            ["eta"] = "LearningRate",
            ["n_estimators"] = "NumberOfTrees",
            ["subsample"] = "SubsampleRatio",
            ["max_depth"] = "MaxDepth"
        };

        foreach (var (alias, expectedProp) in aliases)
        {
            var propertyName = registry.GetPropertyName(ModelType.GradientBoosting, alias);
            Assert.NotNull(propertyName);
            Assert.Equal(expectedProp, propertyName);

            var prop = optionsType.GetProperty(propertyName);
            Assert.NotNull(prop);
            Assert.True(prop.CanWrite, $"Property {propertyName} must be writable");
        }
    }

    [Fact]
    public void Registry_KNN_AllRegisteredAliases_MapToRealProperties()
    {
        var registry = new HyperparameterRegistry();
        var optionsType = typeof(KNearestNeighborsOptions);

        var aliases = new Dictionary<string, string>
        {
            ["n_neighbors"] = "K",
            ["k"] = "K",
            ["num_neighbors"] = "K"
        };

        foreach (var (alias, expectedProp) in aliases)
        {
            var propertyName = registry.GetPropertyName(ModelType.KNearestNeighbors, alias);
            Assert.NotNull(propertyName);

            var prop = optionsType.GetProperty(propertyName);
            Assert.NotNull(prop);
            Assert.True(prop.CanWrite, $"Property {propertyName} must be writable");
        }
    }

    [Fact]
    public void Registry_SVR_AllRegisteredAliases_MapToRealProperties()
    {
        var registry = new HyperparameterRegistry();
        var optionsType = typeof(SupportVectorRegressionOptions);

        var aliases = new Dictionary<string, string>
        {
            ["C"] = "C",
            ["cost"] = "C",
            ["epsilon"] = "Epsilon",
            ["eps"] = "Epsilon"
        };

        foreach (var (alias, expectedProp) in aliases)
        {
            var propertyName = registry.GetPropertyName(ModelType.SupportVectorRegression, alias);
            Assert.NotNull(propertyName);

            var prop = optionsType.GetProperty(propertyName);
            Assert.NotNull(prop);
            Assert.True(prop.CanWrite, $"Property {propertyName} must be writable");
        }
    }

    #endregion

    #region Realistic LLM Response Formats

    [Fact]
    public void FullPipeline_VerboseLLMResponse_ExtractsParametersCorrectly()
    {
        // Simulate a realistic verbose LLM response with surrounding prose
        var llmResponse = @"# Hyperparameter Recommendations

Based on my analysis of your dataset with 10,000 samples and 25 features, I've determined
that a Gradient Boosting model would benefit from the following configuration:

## Recommended Settings

```json
{
    ""n_estimators"": 500,
    ""learning_rate"": 0.03,
    ""max_depth"": 6,
    ""subsample"": 0.7,
    ""min_samples_split"": 10
}
```

### Reasoning

1. **n_estimators = 500**: With a low learning rate, more trees are needed
2. **learning_rate = 0.03**: Conservative learning rate for better generalization
3. **max_depth = 6**: Moderate depth to balance complexity and overfitting
4. **subsample = 0.7**: Stochastic gradient boosting for better generalization
5. **min_samples_split = 10**: Prevents overfitting on small subgroups

These settings represent a good starting point. Consider monitoring validation
performance and adjusting if needed.";

        var options = new GradientBoostingRegressionOptions();
        var model = new GradientBoostingRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.GradientBoosting, hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(500, options.NumberOfTrees);
        Assert.Equal(0.03, options.LearningRate);
        Assert.Equal(6, options.MaxDepth);
        Assert.Equal(0.7, options.SubsampleRatio);
        Assert.Equal(10, options.MinSamplesSplit);
        Assert.Empty(result.Failed);
    }

    [Fact]
    public void FullPipeline_MarkdownOnlyResponse_ExtractsParametersCorrectly()
    {
        var llmResponse = @"Here are my tuning recommendations:

For your Random Forest model:
- **n_estimators:** 250 (enough trees for stable predictions)
- **max_depth:** 20 (deep enough for your complex data)
- **min_samples_split:** 8 (prevents overfit on small subgroups)

I chose these values because your dataset has 50,000 rows and 30 features.";

        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(250, options.NumberOfTrees);
        Assert.Equal(20, options.MaxDepth);
        Assert.Equal(8, options.MinSamplesSplit);
    }

    #endregion

    #region AgentAssistanceOptions Integration

    [Fact]
    public void AgentAssistanceOptions_EnableAutoApply_DefaultsToFalse()
    {
        var options = new AgentAssistanceOptions();
        Assert.False(options.EnableAutoApplyHyperparameters);
    }

    [Fact]
    public void AgentAssistanceOptions_ComprehensivePreset_EnablesAutoApply()
    {
        var options = AgentAssistanceOptions.Comprehensive;
        Assert.True(options.EnableAutoApplyHyperparameters);
    }

    [Fact]
    public void AgentAssistanceOptions_Clone_PreservesAutoApplySetting()
    {
        var original = new AgentAssistanceOptions { EnableAutoApplyHyperparameters = true };
        var clone = original.Clone();

        Assert.True(clone.EnableAutoApplyHyperparameters);
    }

    #endregion

    #region AgentRecommendation Integration

    [Fact]
    public void AgentRecommendation_StoresHyperparameterResult()
    {
        var recommendation = new AgentRecommendation<double, double[], double>
        {
            SuggestedHyperparameters = new Dictionary<string, object>
            {
                ["n_estimators"] = 200,
                ["max_depth"] = 10
            }
        };

        // Simulate what ApplyAgentRecommendationsCore does
        var options = new RandomForestRegressionOptions();
        var model = new RandomForestRegression<double>(options);
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var result = applicator.Apply(model, ModelType.RandomForest, recommendation.SuggestedHyperparameters);
        recommendation.HyperparameterApplicationResult = result;

        // Verify the result is stored on the recommendation
        Assert.NotNull(recommendation.HyperparameterApplicationResult);
        Assert.True(recommendation.HyperparameterApplicationResult.HasAppliedParameters);
        Assert.Equal(200, options.NumberOfTrees);
    }

    #endregion

    #region Edge Cases: Empty and No-Op Scenarios

    [Fact]
    public void FullPipeline_EmptyLLMResponse_NoChangesToModel()
    {
        var options = new RandomForestRegressionOptions { NumberOfTrees = 100, MaxDepth = 10 };
        var model = new RandomForestRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse("");
        var result = applicator.Apply(model, ModelType.RandomForest, hyperparams);

        // Model should remain unchanged
        Assert.False(result.HasAppliedParameters);
        Assert.Equal(100, options.NumberOfTrees);
        Assert.Equal(10, options.MaxDepth);
    }

    [Fact]
    public void FullPipeline_ProseOnlyResponse_NoChangesToModel()
    {
        var llmResponse = "The model looks good as configured. I don't recommend any changes to the hyperparameters at this time.";

        var options = new GradientBoostingRegressionOptions { LearningRate = 0.1 };
        var model = new GradientBoostingRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.GradientBoosting, hyperparams);

        Assert.False(result.HasAppliedParameters);
        Assert.Equal(0.1, options.LearningRate);
    }

    #endregion

    #region Scientific Notation and Special Values

    [Fact]
    public void FullPipeline_ScientificNotation_ParsedCorrectly()
    {
        var llmResponse = @"```json
{""learning_rate"": 1e-3}
```";

        var options = new GradientBoostingRegressionOptions();
        var model = new GradientBoostingRegression<double>(options);
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();
        var applicator = new AgentHyperparameterApplicator<double>(registry);

        var hyperparams = parser.Parse(llmResponse);
        var result = applicator.Apply(model, ModelType.GradientBoosting, hyperparams);

        Assert.True(result.HasAppliedParameters);
        Assert.Equal(0.001, options.LearningRate);
    }

    #endregion
}
