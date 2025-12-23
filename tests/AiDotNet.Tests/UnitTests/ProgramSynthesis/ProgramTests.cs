using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

/// <summary>
/// Unit tests for Program class.
/// </summary>
public class ProgramTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesInstance()
    {
        // Arrange
        const string sourceCode = "def add(a, b):\n    return a + b";
        const ProgramLanguage language = ProgramLanguage.Python;

        // Act
        var program = new Program<double>(sourceCode, language, isValid: true, fitnessScore: 1.0, complexity: 2);

        // Assert
        Assert.NotNull(program);
        Assert.Equal(sourceCode, program.SourceCode);
        Assert.Equal(language, program.Language);
        Assert.True(program.IsValid);
        Assert.Equal(1.0, program.FitnessScore);
        Assert.Equal(2, program.Complexity);
    }

    [Fact]
    public void Constructor_DefaultConstructor_CreatesEmptyProgram()
    {
        // Act
        var program = new Program<double>();

        // Assert
        Assert.NotNull(program);
        Assert.Empty(program.SourceCode);
        Assert.Equal(ProgramLanguage.Generic, program.Language);
        Assert.False(program.IsValid);
        Assert.Equal(0.0, program.FitnessScore);
        Assert.Equal(0, program.Complexity);
    }

    [Fact]
    public void Properties_SettersAndGetters_WorkCorrectly()
    {
        // Arrange
        var program = new Program<double>();

        // Act
        program.SourceCode = "print('Hello, World!')";
        program.Language = ProgramLanguage.Python;
        program.IsValid = true;
        program.FitnessScore = 0.95;
        program.Complexity = 1;
        program.ErrorMessage = null;
        program.ExecutionTimeMs = 5.5;

        // Assert
        Assert.Equal("print('Hello, World!')", program.SourceCode);
        Assert.Equal(ProgramLanguage.Python, program.Language);
        Assert.True(program.IsValid);
        Assert.Equal(0.95, program.FitnessScore);
        Assert.Equal(1, program.Complexity);
        Assert.Null(program.ErrorMessage);
        Assert.Equal(5.5, program.ExecutionTimeMs);
    }

    [Fact]
    public void ToString_ReturnsFormattedString()
    {
        // Arrange
        var program = new Program<double>(
            "x = 5",
            ProgramLanguage.Python,
            isValid: true,
            fitnessScore: 0.75,
            complexity: 1);

        // Act
        var result = program.ToString();

        // Assert
        Assert.Contains("[Python]", result);
        Assert.Contains("Valid: True", result);
        Assert.Contains("Fitness: 0.75", result);
        Assert.Contains("Complexity: 1", result);
        Assert.Contains("x = 5", result);
    }

    [Fact]
    public void ErrorMessage_WhenSet_StoresCorrectly()
    {
        // Arrange
        var program = new Program<double>("invalid code", ProgramLanguage.Python, false);

        // Act
        program.ErrorMessage = "Syntax error on line 1";

        // Assert
        Assert.Equal("Syntax error on line 1", program.ErrorMessage);
    }
}
