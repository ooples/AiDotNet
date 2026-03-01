using AiDotNet.Agents;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Agents;

/// <summary>
/// Deep integration tests for HyperparameterDefinition (NormalizeName, alias matching),
/// HyperparameterRegistry (parameter lookup, validation ranges, default registrations),
/// HyperparameterResponseParser (JSON, markdown, colon-separated parsing, type inference),
/// and HyperparameterValidationResult (factory methods).
/// </summary>
public class AgentsDeepMathIntegrationTests
{
    // ============================
    // HyperparameterDefinition: NormalizeName
    // ============================

    [Theory]
    [InlineData("learning_rate", "learningrate")]
    [InlineData("LEARNING_RATE", "learningrate")]
    [InlineData("LearningRate", "learningrate")]
    [InlineData("learning-rate", "learningrate")]
    [InlineData("learning rate", "learningrate")]
    [InlineData("n_estimators", "nestimators")]
    [InlineData("max_depth", "maxdepth")]
    [InlineData("MaxDepth", "maxdepth")]
    [InlineData("MIN_SAMPLES_SPLIT", "minsamplessplit")]
    public void NormalizeName_VariousFormats_ReturnsLowercaseNoSeparators(string input, string expected)
    {
        Assert.Equal(expected, HyperparameterDefinition.NormalizeName(input));
    }

    [Fact]
    public void NormalizeName_EmptyString_ReturnsEmpty()
    {
        Assert.Equal("", HyperparameterDefinition.NormalizeName(""));
    }

    [Fact]
    public void NormalizeName_AlreadyNormalized_Unchanged()
    {
        var name = "learningrate";
        Assert.Equal(name, HyperparameterDefinition.NormalizeName(name));
    }

    // ============================
    // HyperparameterDefinition: Alias Matching
    // ============================

    [Fact]
    public void MatchesAlias_AfterBuild_MatchesPropertyName()
    {
        var def = new HyperparameterDefinition
        {
            PropertyName = "MaxDepth",
            Aliases = new List<string> { "max_depth", "depth" }
        };
        def.BuildNormalizedAliases();

        // Property name itself is included as an alias
        Assert.True(def.MatchesAlias("maxdepth"));
    }

    [Fact]
    public void MatchesAlias_AfterBuild_MatchesAliases()
    {
        var def = new HyperparameterDefinition
        {
            PropertyName = "MaxDepth",
            Aliases = new List<string> { "max_depth", "tree_depth", "depth" }
        };
        def.BuildNormalizedAliases();

        Assert.True(def.MatchesAlias("maxdepth"));
        Assert.True(def.MatchesAlias("treedepth"));
        Assert.True(def.MatchesAlias("depth"));
    }

    [Fact]
    public void MatchesAlias_CaseInsensitive()
    {
        var def = new HyperparameterDefinition
        {
            PropertyName = "MaxDepth",
            Aliases = new List<string> { "max_depth" }
        };
        def.BuildNormalizedAliases();

        Assert.True(def.MatchesAlias("MAXDEPTH"));
        Assert.True(def.MatchesAlias("MaxDepth"));
        Assert.True(def.MatchesAlias("maxdepth"));
    }

    [Fact]
    public void MatchesAlias_UnknownAlias_ReturnsFalse()
    {
        var def = new HyperparameterDefinition
        {
            PropertyName = "MaxDepth",
            Aliases = new List<string> { "max_depth" }
        };
        def.BuildNormalizedAliases();

        Assert.False(def.MatchesAlias("learningrate"));
    }

    // ============================
    // HyperparameterRegistry: Default Registrations
    // ============================

    [Fact]
    public void Registry_TreeModels_HaveMaxDepth()
    {
        var registry = new HyperparameterRegistry();

        var treeModels = new[] {
            ModelType.RandomForest, ModelType.GradientBoosting, ModelType.DecisionTree,
            ModelType.ConditionalInferenceTree, ModelType.ExtremelyRandomizedTrees,
            ModelType.HistGradientBoosting, ModelType.AdaBoostR2
        };

        foreach (var model in treeModels)
        {
            var propName = registry.GetPropertyName(model, "max_depth");
            Assert.Equal("MaxDepth", propName);
        }
    }

    [Fact]
    public void Registry_TreeModels_MaxDepth_Range1To100()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.RandomForest, "max_depth");

        Assert.NotNull(def);
        Assert.Equal(1.0, def.MinValue);
        Assert.Equal(100.0, def.MaxValue);
    }

    [Fact]
    public void Registry_EnsembleModels_HaveNumberOfTrees()
    {
        var registry = new HyperparameterRegistry();

        var ensembleModels = new[] {
            ModelType.RandomForest, ModelType.GradientBoosting,
            ModelType.ExtremelyRandomizedTrees, ModelType.HistGradientBoosting,
            ModelType.AdaBoostR2
        };

        foreach (var model in ensembleModels)
        {
            var propName = registry.GetPropertyName(model, "n_estimators");
            Assert.Equal("NumberOfTrees", propName);
        }
    }

    [Fact]
    public void Registry_NumberOfTrees_Range1To10000()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.RandomForest, "n_estimators");

        Assert.NotNull(def);
        Assert.Equal(1.0, def.MinValue);
        Assert.Equal(10000.0, def.MaxValue);
    }

    [Fact]
    public void Registry_NEstimators_AllAliasesWork()
    {
        var registry = new HyperparameterRegistry();
        var aliases = new[] { "n_estimators", "num_trees", "ntrees", "number_of_trees", "n_trees" };

        foreach (var alias in aliases)
        {
            var propName = registry.GetPropertyName(ModelType.RandomForest, alias);
            Assert.Equal("NumberOfTrees", propName);
        }
    }

    [Fact]
    public void Registry_GradientBoosting_HasLearningRate()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.GradientBoosting, "learning_rate");

        Assert.NotNull(def);
        Assert.Equal("LearningRate", def.PropertyName);
        Assert.Equal(0.0001, def.MinValue);
        Assert.Equal(1.0, def.MaxValue);
    }

    [Fact]
    public void Registry_GradientBoosting_LearningRateAliases()
    {
        var registry = new HyperparameterRegistry();
        var aliases = new[] { "learning_rate", "lr", "eta", "shrinkage" };

        foreach (var alias in aliases)
        {
            Assert.Equal("LearningRate", registry.GetPropertyName(ModelType.GradientBoosting, alias));
        }
    }

    [Fact]
    public void Registry_GradientBoosting_SubsampleRatio()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.GradientBoosting, "subsample");

        Assert.NotNull(def);
        Assert.Equal("SubsampleRatio", def.PropertyName);
        Assert.Equal(0.1, def.MinValue);
        Assert.Equal(1.0, def.MaxValue);
    }

    [Fact]
    public void Registry_NeuralNetworks_HaveLearningRateEpochsBatchSize()
    {
        var registry = new HyperparameterRegistry();

        var nnModels = new[] { ModelType.NeuralNetworkRegression, ModelType.MultilayerPerceptronRegression };

        foreach (var model in nnModels)
        {
            Assert.Equal("LearningRate", registry.GetPropertyName(model, "learning_rate"));
            Assert.Equal("Epochs", registry.GetPropertyName(model, "epochs"));
            Assert.Equal("BatchSize", registry.GetPropertyName(model, "batch_size"));
        }
    }

    [Fact]
    public void Registry_NeuralNetwork_LearningRate_Range()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.NeuralNetworkRegression, "lr");

        Assert.NotNull(def);
        Assert.Equal(1e-5, def.MinValue);
        Assert.Equal(1.0, def.MaxValue);
    }

    [Fact]
    public void Registry_NeuralNetwork_Epochs_Range()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.NeuralNetworkRegression, "num_epochs");

        Assert.NotNull(def);
        Assert.Equal(1.0, def.MinValue);
        Assert.Equal(100000.0, def.MaxValue);
    }

    [Fact]
    public void Registry_PolynomialRegression_Degree()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.PolynomialRegression, "degree");

        Assert.NotNull(def);
        Assert.Equal("Degree", def.PropertyName);
        Assert.Equal(1.0, def.MinValue);
        Assert.Equal(20.0, def.MaxValue);
    }

    [Fact]
    public void Registry_RegularizedModels_Alpha()
    {
        var registry = new HyperparameterRegistry();

        var models = new[] { ModelType.RidgeRegression, ModelType.LassoRegression, ModelType.ElasticNetRegression };

        foreach (var model in models)
        {
            var def = registry.GetDefinition(model, "alpha");
            Assert.NotNull(def);
            Assert.Equal("Alpha", def.PropertyName);
            Assert.Equal(0.0001, def.MinValue);
            Assert.Equal(1000.0, def.MaxValue);
        }
    }

    [Fact]
    public void Registry_Alpha_AllAliases()
    {
        var registry = new HyperparameterRegistry();
        var aliases = new[] { "alpha", "regularization", "lambda", "reg_strength", "penalty" };

        foreach (var alias in aliases)
        {
            Assert.Equal("Alpha", registry.GetPropertyName(ModelType.RidgeRegression, alias));
        }
    }

    [Fact]
    public void Registry_KNN_K()
    {
        var registry = new HyperparameterRegistry();
        var def = registry.GetDefinition(ModelType.KNearestNeighbors, "n_neighbors");

        Assert.NotNull(def);
        Assert.Equal("K", def.PropertyName);
        Assert.Equal(1.0, def.MinValue);
        Assert.Equal(1000.0, def.MaxValue);
    }

    [Fact]
    public void Registry_SVR_C_And_Epsilon()
    {
        var registry = new HyperparameterRegistry();

        var cDef = registry.GetDefinition(ModelType.SupportVectorRegression, "cost");
        Assert.NotNull(cDef);
        Assert.Equal("C", cDef.PropertyName);
        Assert.Equal(0.001, cDef.MinValue);
        Assert.Equal(10000.0, cDef.MaxValue);

        var epsDef = registry.GetDefinition(ModelType.SupportVectorRegression, "epsilon");
        Assert.NotNull(epsDef);
        Assert.Equal("Epsilon", epsDef.PropertyName);
        Assert.Equal(0.0, epsDef.MinValue);
        Assert.Equal(10.0, epsDef.MaxValue);
    }

    [Fact]
    public void Registry_TimeSeries_LagOrder_And_SeasonalPeriod()
    {
        var registry = new HyperparameterRegistry();

        var lagDef = registry.GetDefinition(ModelType.TimeSeriesRegression, "lag_order");
        Assert.NotNull(lagDef);
        Assert.Equal("LagOrder", lagDef.PropertyName);
        Assert.Equal(1.0, lagDef.MinValue);
        Assert.Equal(100.0, lagDef.MaxValue);

        var seasonDef = registry.GetDefinition(ModelType.TimeSeriesRegression, "seasonal_period");
        Assert.NotNull(seasonDef);
        Assert.Equal("SeasonalPeriod", seasonDef.PropertyName);
        Assert.Equal(2.0, seasonDef.MinValue);
        Assert.Equal(365.0, seasonDef.MaxValue);
    }

    [Fact]
    public void Registry_SharedParameters_Seed()
    {
        var registry = new HyperparameterRegistry();

        // Shared parameters should work for any model type
        var propName = registry.GetPropertyName(ModelType.RandomForest, "random_seed");
        Assert.Equal("Seed", propName);

        propName = registry.GetPropertyName(ModelType.NeuralNetworkRegression, "seed");
        Assert.Equal("Seed", propName);
    }

    [Fact]
    public void Registry_SharedParameters_UseIntercept()
    {
        var registry = new HyperparameterRegistry();

        var propName = registry.GetPropertyName(ModelType.RidgeRegression, "fit_intercept");
        Assert.Equal("UseIntercept", propName);
    }

    [Fact]
    public void Registry_UnknownParam_ReturnsNull()
    {
        var registry = new HyperparameterRegistry();

        var propName = registry.GetPropertyName(ModelType.RandomForest, "completely_unknown_param");
        Assert.Null(propName);
    }

    // ============================
    // HyperparameterRegistry: Validation
    // ============================

    [Fact]
    public void Validate_WithinRange_Valid()
    {
        var registry = new HyperparameterRegistry();

        var result = registry.Validate(ModelType.RandomForest, "max_depth", 10);
        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    [Fact]
    public void Validate_BelowMinimum_Warning()
    {
        var registry = new HyperparameterRegistry();

        // max_depth minimum is 1, so 0 should warn
        var result = registry.Validate(ModelType.RandomForest, "max_depth", 0);
        Assert.True(result.IsValid);
        Assert.True(result.HasWarning);
        Assert.Contains("below the typical minimum", result.Warning);
    }

    [Fact]
    public void Validate_AboveMaximum_Warning()
    {
        var registry = new HyperparameterRegistry();

        // max_depth maximum is 100, so 200 should warn
        var result = registry.Validate(ModelType.RandomForest, "max_depth", 200);
        Assert.True(result.IsValid);
        Assert.True(result.HasWarning);
        Assert.Contains("above the typical maximum", result.Warning);
    }

    [Fact]
    public void Validate_UnknownParam_Valid()
    {
        var registry = new HyperparameterRegistry();

        // Unknown parameters are always valid (no constraints to check)
        var result = registry.Validate(ModelType.RandomForest, "unknown_param", 999);
        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    [Fact]
    public void Validate_NonNumericValue_Valid()
    {
        var registry = new HyperparameterRegistry();

        // Non-convertible values skip range validation
        var result = registry.Validate(ModelType.RandomForest, "max_depth", "not_a_number");
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_AtExactMinimum_Valid()
    {
        var registry = new HyperparameterRegistry();

        // max_depth min is 1, so 1 should be valid (inclusive)
        var result = registry.Validate(ModelType.RandomForest, "max_depth", 1);
        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    [Fact]
    public void Validate_AtExactMaximum_Valid()
    {
        var registry = new HyperparameterRegistry();

        // max_depth max is 100, so 100 should be valid (inclusive)
        var result = registry.Validate(ModelType.RandomForest, "max_depth", 100);
        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
    }

    // ============================
    // HyperparameterValidationResult: Factory Methods
    // ============================

    [Fact]
    public void ValidationResult_Valid_NoWarning()
    {
        var result = HyperparameterValidationResult.Valid();

        Assert.True(result.IsValid);
        Assert.False(result.HasWarning);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void ValidationResult_WithWarning_StillValid()
    {
        var result = HyperparameterValidationResult.WithWarning("test warning");

        Assert.True(result.IsValid);
        Assert.True(result.HasWarning);
        Assert.Equal("test warning", result.Warning);
    }

    [Fact]
    public void ValidationResult_Invalid_NotValid()
    {
        var result = HyperparameterValidationResult.Invalid("test reason");

        Assert.False(result.IsValid);
        Assert.True(result.HasWarning);
        Assert.Equal("test reason", result.Warning);
    }

    // ============================
    // HyperparameterResponseParser: JSON Parsing
    // ============================

    [Fact]
    public void Parse_JsonBlock_ExtractsParameters()
    {
        var parser = new HyperparameterResponseParser();

        var response = @"Here are the parameters:
```json
{""learning_rate"": 0.01, ""n_estimators"": 100, ""max_depth"": 5}
```";

        var result = parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(0.01, result["learning_rate"]);
        Assert.Equal(100, result["n_estimators"]);
        Assert.Equal(5, result["max_depth"]);
    }

    [Fact]
    public void TryParseJson_JsonBlockWithCodeFence_ParsesCorrectly()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"```json
{""lr"": 0.001, ""epochs"": 50}
```";

        var result = parser.TryParseJson(text);

        Assert.Equal(2, result.Count);
        Assert.Equal(0.001, result["lr"]);
        Assert.Equal(50, result["epochs"]);
    }

    [Fact]
    public void TryParseJson_RawJsonObject_ParsesCorrectly()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"I recommend {""alpha"": 0.5, ""degree"": 3}";

        var result = parser.TryParseJson(text);

        Assert.Equal(2, result.Count);
        Assert.Equal(0.5, result["alpha"]);
        Assert.Equal(3, result["degree"]);
    }

    [Fact]
    public void TryParseJson_NoJson_ReturnsEmpty()
    {
        var parser = new HyperparameterResponseParser();

        var result = parser.TryParseJson("No JSON here, just text");

        Assert.Empty(result);
    }

    [Fact]
    public void TryParseJson_BooleanValues_Parsed()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"```json
{""fit_intercept"": true, ""verbose"": false}
```";

        var result = parser.TryParseJson(text);

        Assert.Equal(true, result["fit_intercept"]);
        Assert.Equal(false, result["verbose"]);
    }

    // ============================
    // HyperparameterResponseParser: Markdown Bold Parsing
    // ============================

    [Fact]
    public void TryParseMarkdownBold_BoldPatterns_Extracted()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"Recommended parameters:
**learning_rate:** 0.01
**n_estimators:** 200
**max_depth:** 8";

        var result = parser.TryParseMarkdownBold(text);

        Assert.Equal(3, result.Count);
        Assert.Equal(0.01, result["learning_rate"]);
        Assert.Equal(200, result["n_estimators"]);
        Assert.Equal(8, result["max_depth"]);
    }

    [Fact]
    public void TryParseMarkdownBold_NoMarkdown_ReturnsEmpty()
    {
        var parser = new HyperparameterResponseParser();

        var result = parser.TryParseMarkdownBold("Just plain text");

        Assert.Empty(result);
    }

    // ============================
    // HyperparameterResponseParser: Colon-Separated Parsing
    // ============================

    [Fact]
    public void TryParseColonSeparated_ColonFormat_Extracted()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"learning_rate: 0.001
n_estimators: 500
max_depth: 10";

        var result = parser.TryParseColonSeparated(text);

        Assert.Equal(3, result.Count);
        Assert.Equal(0.001, result["learning_rate"]);
        Assert.Equal(500, result["n_estimators"]);
        Assert.Equal(10, result["max_depth"]);
    }

    [Fact]
    public void TryParseColonSeparated_EqualsFormat_Extracted()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"alpha = 0.5
degree = 3";

        var result = parser.TryParseColonSeparated(text);

        Assert.Equal(2, result.Count);
        Assert.Equal(0.5, result["alpha"]);
        Assert.Equal(3, result["degree"]);
    }

    [Fact]
    public void TryParseColonSeparated_SkipsNonParameterNames()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"step: 1
note: something
model: RandomForest
learning_rate: 0.01";

        var result = parser.TryParseColonSeparated(text);

        // "step", "note", "model" are filtered out as non-parameter names
        Assert.DoesNotContain("step", result.Keys);
        Assert.DoesNotContain("note", result.Keys);
        Assert.DoesNotContain("model", result.Keys);
        Assert.Contains("learning_rate", result.Keys);
    }

    // ============================
    // HyperparameterResponseParser: Type Inference
    // ============================

    [Fact]
    public void InferTypedValue_Integer_ReturnsInt()
    {
        var result = HyperparameterResponseParser.InferTypedValue("42");

        Assert.IsType<int>(result);
        Assert.Equal(42, result);
    }

    [Fact]
    public void InferTypedValue_Double_ReturnsDouble()
    {
        var result = HyperparameterResponseParser.InferTypedValue("0.001");

        Assert.IsType<double>(result);
        Assert.Equal(0.001, result);
    }

    [Fact]
    public void InferTypedValue_Boolean_ReturnsBool()
    {
        Assert.Equal(true, HyperparameterResponseParser.InferTypedValue("true"));
        Assert.Equal(false, HyperparameterResponseParser.InferTypedValue("false"));
    }

    [Fact]
    public void InferTypedValue_String_ReturnsString()
    {
        var result = HyperparameterResponseParser.InferTypedValue("relu");

        Assert.IsType<string>(result);
        Assert.Equal("relu", result);
    }

    [Fact]
    public void InferTypedValue_QuotedString_StripsQuotes()
    {
        var result = HyperparameterResponseParser.InferTypedValue("\"sigmoid\"");
        Assert.Equal("sigmoid", result);
    }

    [Fact]
    public void InferTypedValue_NullOrEmpty_ReturnsNull()
    {
        Assert.Null(HyperparameterResponseParser.InferTypedValue(""));
        Assert.Null(HyperparameterResponseParser.InferTypedValue("  "));
    }

    // ============================
    // HyperparameterResponseParser: Parse Strategy Priority
    // ============================

    [Fact]
    public void Parse_EmptyInput_ReturnsEmpty()
    {
        var parser = new HyperparameterResponseParser();

        Assert.Empty(parser.Parse(""));
        Assert.Empty(parser.Parse("   "));
    }

    [Fact]
    public void Parse_JsonTakesPriority_OverMarkdown()
    {
        var parser = new HyperparameterResponseParser();

        var text = @"**learning_rate:** 0.1
```json
{""learning_rate"": 0.001}
```";

        var result = parser.Parse(text);

        // JSON should take priority
        Assert.Equal(0.001, result["learning_rate"]);
    }

    [Fact]
    public void Parse_MarkdownTakesPriority_OverColon()
    {
        var parser = new HyperparameterResponseParser();

        // No JSON, so markdown bold should be checked first
        var text = @"**learning_rate:** 0.01
Some text learning_rate: 0.1";

        var result = parser.Parse(text);

        // Markdown bold should take priority over colon-separated
        Assert.Equal(0.01, result["learning_rate"]);
    }

    // ============================
    // HyperparameterRegistry: Custom Registration
    // ============================

    [Fact]
    public void Register_CustomParameter_Resolvable()
    {
        var registry = new HyperparameterRegistry();

        registry.Register(ModelType.RandomForest, new HyperparameterDefinition
        {
            PropertyName = "CustomParam",
            Aliases = new List<string> { "custom_param", "my_param" },
            ValueType = typeof(double),
            MinValue = 0.0,
            MaxValue = 1.0
        });

        Assert.Equal("CustomParam", registry.GetPropertyName(ModelType.RandomForest, "custom_param"));
        Assert.Equal("CustomParam", registry.GetPropertyName(ModelType.RandomForest, "my_param"));
    }

    [Fact]
    public void Register_ModelSpecific_TakesPriorityOverShared()
    {
        var registry = new HyperparameterRegistry();

        // "seed" is registered as a shared parameter
        // Model-specific lookup should find the shared one
        var propName = registry.GetPropertyName(ModelType.RandomForest, "seed");
        Assert.Equal("Seed", propName);
    }

    // ============================
    // Cross-Component Integration: Registry + Parser + Validation
    // ============================

    [Fact]
    public void EndToEnd_ParseThenLookupThenValidate()
    {
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();

        var llmResponse = @"```json
{""n_estimators"": 200, ""max_depth"": 15, ""learning_rate"": 0.01}
```";

        // Step 1: Parse LLM response
        var parsed = parser.Parse(llmResponse);
        Assert.Equal(3, parsed.Count);

        // Step 2: Resolve property names
        Assert.Equal("NumberOfTrees", registry.GetPropertyName(ModelType.GradientBoosting, "n_estimators"));
        Assert.Equal("MaxDepth", registry.GetPropertyName(ModelType.GradientBoosting, "max_depth"));
        Assert.Equal("LearningRate", registry.GetPropertyName(ModelType.GradientBoosting, "learning_rate"));

        // Step 3: Validate
        foreach (var kvp in parsed)
        {
            var result = registry.Validate(ModelType.GradientBoosting, kvp.Key, kvp.Value);
            Assert.True(result.IsValid);
            Assert.False(result.HasWarning);
        }
    }

    [Fact]
    public void EndToEnd_ParseThenValidate_OutOfRange()
    {
        var parser = new HyperparameterResponseParser();
        var registry = new HyperparameterRegistry();

        var llmResponse = @"```json
{""max_depth"": 999, ""learning_rate"": 5.0}
```";

        var parsed = parser.Parse(llmResponse);

        // max_depth 999 > max 100 → warning
        var depthResult = registry.Validate(ModelType.GradientBoosting, "max_depth", parsed["max_depth"]);
        Assert.True(depthResult.HasWarning);

        // learning_rate 5.0 > max 1.0 → warning
        var lrResult = registry.Validate(ModelType.GradientBoosting, "learning_rate", parsed["learning_rate"]);
        Assert.True(lrResult.HasWarning);
    }
}
