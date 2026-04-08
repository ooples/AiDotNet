using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

/// <summary>
/// Unit tests for ProgramInput class.
/// </summary>
public class ProgramInputTests
{
    [Fact]
    public void Constructor_WithParameters_CreatesInstance()
    {
        // Arrange
        const string description = "Create a function that sorts a list";
        var examples = new List<ProgramInputOutputExample>
        {
            new() { Input = "[3, 1, 2]", ExpectedOutput = "[1, 2, 3]" },
            new() { Input = "[5, 4]", ExpectedOutput = "[4, 5]" }
        };
        var constraints = new List<string> { "Must use O(n log n) algorithm" };

        // Act
        var input = new ProgramInput<double>(
            description,
            ProgramLanguage.Python,
            examples,
            constraints);

        // Assert
        Assert.NotNull(input);
        Assert.Equal(description, input.Description);
        Assert.Equal(ProgramLanguage.Python, input.TargetLanguage);
        Assert.Equal(2, input.Examples?.Count);
        Assert.Single(input.Constraints ?? new List<string>());
    }

    [Fact]
    public void Constructor_DefaultConstructor_CreatesEmptyInstance()
    {
        // Act
        var input = new ProgramInput<double>();

        // Assert
        Assert.NotNull(input);
        Assert.Equal(ProgramLanguage.Generic, input.TargetLanguage);
        Assert.Null(input.Description);
        Assert.Null(input.Examples);
    }

    [Fact]
    public void AddExample_AddsExampleCorrectly()
    {
        // Arrange
        var input = new ProgramInput<double>();

        // Act
        input.AddExample("[1, 2, 3]", "6");
        input.AddExample("[4, 5]", "9");

        // Assert
        Assert.NotNull(input.Examples);
        Assert.Equal(2, input.Examples.Count);
        Assert.Equal("[1, 2, 3]", input.Examples[0].Input);
        Assert.Equal("6", input.Examples[0].ExpectedOutput);
        Assert.Equal("[4, 5]", input.Examples[1].Input);
        Assert.Equal("9", input.Examples[1].ExpectedOutput);
    }

    [Fact]
    public void AddTestCase_AddsTestCaseCorrectly()
    {
        // Arrange
        var input = new ProgramInput<double>();

        // Act
        input.AddTestCase("[10]", "10");
        input.AddTestCase("[1, 1, 1]", "3");

        // Assert
        Assert.NotNull(input.TestCases);
        Assert.Equal(2, input.TestCases.Count);
        Assert.Equal("[10]", input.TestCases[0].Input);
        Assert.Equal("10", input.TestCases[0].ExpectedOutput);
        Assert.Equal("[1, 1, 1]", input.TestCases[1].Input);
        Assert.Equal("3", input.TestCases[1].ExpectedOutput);
    }

    [Fact]
    public void AddConstraint_AddsConstraintCorrectly()
    {
        // Arrange
        var input = new ProgramInput<double>();

        // Act
        input.AddConstraint("Must be fast");
        input.AddConstraint("Should be readable");

        // Assert
        Assert.NotNull(input.Constraints);
        Assert.Equal(2, input.Constraints.Count);
        Assert.Contains("Must be fast", input.Constraints);
        Assert.Contains("Should be readable", input.Constraints);
    }

    [Fact]
    public void Properties_SettersAndGetters_WorkCorrectly()
    {
        // Arrange
        var input = new ProgramInput<double>();

        // Act
        input.Description = "Generate a sorting function";
        input.TargetLanguage = ProgramLanguage.Java;
        input.FormalSpecification = "∀x∀y: x < y ⇒ sorted[x] ≤ sorted[y]";
        input.MaxComplexity = 50;
        input.TimeoutMs = 5000;

        // Assert
        Assert.Equal("Generate a sorting function", input.Description);
        Assert.Equal(ProgramLanguage.Java, input.TargetLanguage);
        Assert.Equal("∀x∀y: x < y ⇒ sorted[x] ≤ sorted[y]", input.FormalSpecification);
        Assert.Equal(50, input.MaxComplexity);
        Assert.Equal(5000, input.TimeoutMs);
    }

    [Fact]
    public void AddExample_MultipleTimesSeparately_MaintainsOrder()
    {
        // Arrange
        var input = new ProgramInput<double>();

        // Act
        input.AddExample("input1", "output1");
        input.AddExample("input2", "output2");
        input.AddExample("input3", "output3");

        // Assert
        Assert.Equal(3, input.Examples?.Count);
        Assert.Equal("input1", input.Examples?[0].Input);
        Assert.Equal("output2", input.Examples?[1].ExpectedOutput);
        Assert.Equal("input3", input.Examples?[2].Input);
    }
}
