using AiDotNet.Tools;
using Newtonsoft.Json.Linq;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.Tools;

/// <summary>
/// Unit tests for the ToolBase class helper methods.
/// Tests the robustness of TryGet* methods against invalid input types.
/// </summary>
public class ToolBaseTests
{
    // Use ModelSelectionTool as a concrete implementation to test ToolBase methods
    private readonly ModelSelectionTool _tool = new();

    #region PR #756 Bug Fix Tests - Type Conversion Robustness

    [Fact(Timeout = 60000)]
    public async Task Execute_InvalidIntType_UsesDefaultInsteadOfThrowing()
    {
        // Arrange - n_samples should be int but we pass a string
        var input = """{"problem_type": "regression", "n_samples": "not_a_number"}""";

        // Act
        var result = _tool.Execute(input);

        // Assert - Should not throw, should use default value and complete successfully
        Assert.DoesNotContain("Error", result);
        Assert.Contains("MODEL SELECTION RECOMMENDATION", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_InvalidDoubleType_UsesDefaultInsteadOfThrowing()
    {
        // Arrange - Use RegularizationTool which uses TryGetDouble for training_score
        var regularizationTool = new RegularizationTool();
        var input = """{"model_type": "NeuralNetwork", "training_score": "invalid", "validation_score": 0.8}""";

        // Act
        var result = regularizationTool.Execute(input);

        // Assert - Should not throw, should use default value
        Assert.DoesNotContain("Error", result);
        Assert.Contains("REGULARIZATION RECOMMENDATIONS", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_InvalidBoolType_UsesDefaultInsteadOfThrowing()
    {
        // Arrange - is_linear should be bool but we pass a string
        var input = """{"problem_type": "regression", "is_linear": "not_a_bool"}""";

        // Act
        var result = _tool.Execute(input);

        // Assert - Should not throw, should use default value
        Assert.DoesNotContain("Error", result);
        Assert.Contains("MODEL SELECTION RECOMMENDATION", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_MissingProperties_UsesDefaults()
    {
        // Arrange - Only provide minimal JSON
        var input = """{"problem_type": "classification"}""";

        // Act
        var result = _tool.Execute(input);

        // Assert - Should use defaults for all missing properties
        Assert.DoesNotContain("Error", result);
        Assert.Contains("MODEL SELECTION RECOMMENDATION", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_EmptyInput_ReturnsError()
    {
        // Act
        var result = _tool.Execute("");

        // Assert
        Assert.Contains("Error", result);
        Assert.Contains("empty", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_InvalidJson_ReturnsJsonError()
    {
        // Act
        var result = _tool.Execute("not valid json");

        // Assert
        Assert.Contains("Error", result);
        Assert.Contains("JSON", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_NullPropertyValues_UsesDefaults()
    {
        // Arrange - Explicit null values
        var input = """{"problem_type": null, "n_samples": null, "is_linear": null}""";

        // Act
        var result = _tool.Execute(input);

        // Assert - Should use defaults
        Assert.DoesNotContain("Error", result);
        Assert.Contains("MODEL SELECTION RECOMMENDATION", result);
    }

    #endregion
}
