using AiDotNet.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Tools;

/// <summary>
/// Unit tests for the CalculatorTool class.
/// </summary>
public class CalculatorToolTests
{
    [Fact]
    public void Name_ReturnsCalculator()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var name = calculator.Name;

        // Assert
        Assert.Equal("Calculator", name);
    }

    [Fact]
    public void Description_ReturnsNonEmptyString()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var description = calculator.Description;

        // Assert
        Assert.False(string.IsNullOrWhiteSpace(description));
        Assert.Contains("mathematical", description, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("2 + 2", "4")]
    [InlineData("5 - 3", "2")]
    [InlineData("10 * 5", "50")]
    [InlineData("100 / 4", "25")]
    public void Execute_BasicArithmetic_ReturnsCorrectResult(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("(10 + 5) * 2", "30")]
    [InlineData("100 / (2 + 3)", "20")]
    [InlineData("((10 - 2) * 4) + 5", "37")]
    [InlineData("(5 * (3 + 2)) - 10", "15")]
    public void Execute_ComplexExpressions_ReturnsCorrectResult(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("3.14 * 2", "6.28")]
    [InlineData("10.5 + 5.5", "16")]
    [InlineData("7.5 / 2.5", "3")]
    public void Execute_DecimalNumbers_ReturnsCorrectResult(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }

    [Fact]
    public void Execute_EmptyInput_ReturnsError()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute("");

        // Assert
        Assert.Contains("Error", result);
        Assert.Contains("empty", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Execute_WhitespaceInput_ReturnsError()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute("   ");

        // Assert
        Assert.Contains("Error", result);
    }

    [Theory]
    [InlineData("2 +")]
    [InlineData("* 5")]
    [InlineData("2 + + 3")]
    [InlineData("(5 + 3")]
    public void Execute_InvalidExpression_ReturnsError(string expression)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Contains("Error", result);
    }

    [Theory]
    [InlineData("10 / 0")]
    [InlineData("5 * (10 / 0)")]
    public void Execute_DivisionByZero_ReturnsError(string expression)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Contains("Error", result);
    }

    [Theory]
    [InlineData(" 5 + 3 ", "8")]
    [InlineData("  10  *  2  ", "20")]
    public void Execute_ExpressionWithExtraWhitespace_ReturnsCorrectResult(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("0 + 0", "0")]
    [InlineData("0 * 100", "0")]
    [InlineData("5 - 5", "0")]
    public void Execute_ResultIsZero_ReturnsZero(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("-5 + 10", "5")]
    [InlineData("-10 * 2", "-20")]
    [InlineData("15 + (-5)", "10")]
    public void Execute_NegativeNumbers_ReturnsCorrectResult(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }

    [Fact]
    public void Execute_LargeNumbers_HandlesCorrectly()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute("1000000 * 1000");

        // Assert
        Assert.Equal("1000000000", result);
    }

    [Theory]
    [InlineData("2 + 3 * 4", "14")] // Multiplication before addition
    [InlineData("10 - 2 * 3", "4")] // Multiplication before subtraction
    [InlineData("20 / 4 + 3", "8")] // Division before addition
    public void Execute_OrderOfOperations_ReturnsCorrectResult(string expression, string expected)
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(expression);

        // Assert
        Assert.Equal(expected, result);
    }
}
