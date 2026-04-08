using AiDotNet.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agents;

public class HyperparameterResponseParserTests
{
    private readonly HyperparameterResponseParser _parser = new();

    #region Parse - JSON Extraction

    [Fact]
    public void Parse_WithJsonCodeBlock_ExtractsParameters()
    {
        var response = @"Based on your data, I recommend these hyperparameters:

```json
{
    ""n_estimators"": 200,
    ""max_depth"": 10,
    ""learning_rate"": 0.01
}
```

These values should work well for your dataset.";

        var result = _parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(200, result["n_estimators"]);
        Assert.Equal(10, result["max_depth"]);
        Assert.Equal(0.01, result["learning_rate"]);
    }

    [Fact]
    public void Parse_WithRawJsonObject_ExtractsParameters()
    {
        var response = @"I recommend using {""batch_size"": 32, ""epochs"": 100, ""lr"": 0.001} for training.";

        var result = _parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(32, result["batch_size"]);
        Assert.Equal(100, result["epochs"]);
        Assert.Equal(0.001, result["lr"]);
    }

    [Fact]
    public void Parse_WithJsonCodeBlockNoLanguageTag_ExtractsParameters()
    {
        var response = @"Here are my recommendations:

```
{
    ""k"": 5,
    ""metric"": ""euclidean""
}
```";

        var result = _parser.Parse(response);

        Assert.Equal(2, result.Count);
        Assert.Equal(5, result["k"]);
        Assert.Equal("euclidean", result["metric"]);
    }

    [Fact]
    public void Parse_WithJsonBooleans_ExtractsCorrectTypes()
    {
        var response = @"```json
{""use_intercept"": true, ""normalize"": false, ""alpha"": 0.5}
```";

        var result = _parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(true, result["use_intercept"]);
        Assert.Equal(false, result["normalize"]);
        Assert.Equal(0.5, result["alpha"]);
    }

    #endregion

    #region Parse - Markdown Bold Pattern

    [Fact]
    public void Parse_WithMarkdownBold_ExtractsParameters()
    {
        var response = @"I recommend the following hyperparameters:

- **n_estimators:** 200
- **max_depth:** 15
- **learning_rate:** 0.05
- **subsample:** 0.8";

        var result = _parser.Parse(response);

        Assert.Equal(4, result.Count);
        Assert.Equal(200, result["n_estimators"]);
        Assert.Equal(15, result["max_depth"]);
        Assert.Equal(0.05, result["learning_rate"]);
        Assert.Equal(0.8, result["subsample"]);
    }

    [Fact]
    public void Parse_WithMarkdownBoldAndTrailingPunctuation_TrimsCorrectly()
    {
        var response = @"**n_estimators:** 200,
**learning_rate:** 0.01.
**max_depth:** 5;";

        var result = _parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(200, result["n_estimators"]);
        Assert.Equal(0.01, result["learning_rate"]);
        Assert.Equal(5, result["max_depth"]);
    }

    #endregion

    #region Parse - Colon/Equals Separated

    [Fact]
    public void Parse_WithColonSeparated_ExtractsParameters()
    {
        var response = @"Recommended hyperparameters:
n_estimators: 150
max_depth: 8
learning_rate: 0.03";

        var result = _parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(150, result["n_estimators"]);
        Assert.Equal(8, result["max_depth"]);
        Assert.Equal(0.03, result["learning_rate"]);
    }

    [Fact]
    public void Parse_WithEqualsSeparated_ExtractsParameters()
    {
        var response = @"Set these parameters:
batch_size = 64
epochs = 50";

        var result = _parser.Parse(response);

        Assert.Equal(2, result.Count);
        Assert.Equal(64, result["batch_size"]);
        Assert.Equal(50, result["epochs"]);
    }

    [Fact]
    public void Parse_WithBulletPoints_ExtractsParameters()
    {
        var response = @"Try these settings:
- n_neighbors: 7
- metric: manhattan
* weights: distance";

        var result = _parser.Parse(response);

        Assert.Equal(3, result.Count);
        Assert.Equal(7, result["n_neighbors"]);
        Assert.Equal("manhattan", result["metric"]);
        Assert.Equal("distance", result["weights"]);
    }

    [Fact]
    public void Parse_SkipsCommonNonParameters()
    {
        var response = @"step: 1
model: RandomForest
n_estimators: 200
note: This is a recommendation
example: try this";

        var result = _parser.Parse(response);

        Assert.Single(result);
        Assert.Equal(200, result["n_estimators"]);
    }

    #endregion

    #region Parse - Edge Cases

    [Fact]
    public void Parse_WithEmptyInput_ReturnsEmptyDictionary()
    {
        var result = _parser.Parse("");

        Assert.Empty(result);
    }

    [Fact]
    public void Parse_WithNullInput_ReturnsEmptyDictionary()
    {
        var result = _parser.Parse(null!);

        Assert.Empty(result);
    }

    [Fact]
    public void Parse_WithWhitespaceOnly_ReturnsEmptyDictionary()
    {
        var result = _parser.Parse("   \n  \t  ");

        Assert.Empty(result);
    }

    [Fact]
    public void Parse_WithNoHyperparameters_ReturnsEmptyDictionary()
    {
        var response = "The model looks good. No changes needed.";

        var result = _parser.Parse(response);

        Assert.Empty(result);
    }

    [Fact]
    public void Parse_WithMalformedJson_FallsBackToOtherStrategies()
    {
        var response = @"```json
{invalid json here}
```

**n_estimators:** 100
**max_depth:** 5";

        var result = _parser.Parse(response);

        Assert.Equal(2, result.Count);
        Assert.Equal(100, result["n_estimators"]);
        Assert.Equal(5, result["max_depth"]);
    }

    #endregion

    #region Type Inference

    [Fact]
    public void InferTypedValue_WithIntegerString_ReturnsInt()
    {
        var result = HyperparameterResponseParser.InferTypedValue("42");

        Assert.IsType<int>(result);
        Assert.Equal(42, result);
    }

    [Fact]
    public void InferTypedValue_WithDoubleString_ReturnsDouble()
    {
        var result = HyperparameterResponseParser.InferTypedValue("3.14");

        Assert.IsType<double>(result);
        Assert.Equal(3.14, result);
    }

    [Fact]
    public void InferTypedValue_WithBoolString_ReturnsBool()
    {
        var resultTrue = HyperparameterResponseParser.InferTypedValue("true");
        var resultFalse = HyperparameterResponseParser.InferTypedValue("false");

        Assert.IsType<bool>(resultTrue);
        Assert.Equal(true, resultTrue);
        Assert.IsType<bool>(resultFalse);
        Assert.Equal(false, resultFalse);
    }

    [Fact]
    public void InferTypedValue_WithNonNumericString_ReturnsString()
    {
        var result = HyperparameterResponseParser.InferTypedValue("euclidean");

        Assert.IsType<string>(result);
        Assert.Equal("euclidean", result);
    }

    [Fact]
    public void InferTypedValue_WithQuotedString_RemovesQuotes()
    {
        var result = HyperparameterResponseParser.InferTypedValue("\"manhattan\"");

        Assert.IsType<string>(result);
        Assert.Equal("manhattan", result);
    }

    [Fact]
    public void InferTypedValue_WithEmptyString_ReturnsNull()
    {
        var result = HyperparameterResponseParser.InferTypedValue("");

        Assert.Null(result);
    }

    [Fact]
    public void InferTypedValue_WithScientificNotation_ReturnsDouble()
    {
        var result = HyperparameterResponseParser.InferTypedValue("1e-5");

        Assert.IsType<double>(result);
        Assert.Equal(1e-5, result);
    }

    #endregion

    #region Strategy Priority

    [Fact]
    public void Parse_JsonTakesPriorityOverMarkdown()
    {
        var response = @"**n_estimators:** 999

```json
{""n_estimators"": 200}
```";

        var result = _parser.Parse(response);

        Assert.Equal(200, result["n_estimators"]);
    }

    [Fact]
    public void Parse_MarkdownTakesPriorityOverColon()
    {
        var response = @"n_estimators: 999
**n_estimators:** 200";

        var result = _parser.Parse(response);

        Assert.Equal(200, result["n_estimators"]);
    }

    #endregion
}
