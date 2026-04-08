using AiDotNet.Tools;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Tools;

/// <summary>
/// Deep integration tests for Tools:
/// CalculatorTool (expression evaluation, arithmetic, parentheses, division by zero,
/// decimal formatting, error handling),
/// testing mathematical correctness of the DataTable.Compute evaluation engine.
/// </summary>
public class ToolsDeepMathIntegrationTests
{
    // ============================
    // CalculatorTool: Basic Arithmetic
    // ============================

    [Theory]
    [InlineData("2 + 3", "5")]
    [InlineData("10 - 4", "6")]
    [InlineData("6 * 7", "42")]
    [InlineData("100 / 4", "25")]
    public void Calculator_BasicArithmetic_CorrectResults(string expression, string expected)
    {
        var calc = new CalculatorTool();
        Assert.Equal(expected, calc.Execute(expression));
    }

    [Theory]
    [InlineData("2 + 3 * 4", "14")]      // Operator precedence: 3*4=12, 2+12=14
    [InlineData("10 - 2 * 3", "4")]       // 10 - 6 = 4
    [InlineData("20 / 4 + 1", "6")]       // 5 + 1 = 6
    public void Calculator_OperatorPrecedence_CorrectOrder(string expression, string expected)
    {
        var calc = new CalculatorTool();
        Assert.Equal(expected, calc.Execute(expression));
    }

    // ============================
    // CalculatorTool: Parentheses
    // ============================

    [Theory]
    [InlineData("(2 + 3) * 4", "20")]
    [InlineData("(10 - 2) * (3 + 1)", "32")]
    [InlineData("((2 + 3) * (4 - 1))", "15")]
    [InlineData("100 / (5 * 4)", "5")]
    public void Calculator_Parentheses_CorrectGrouping(string expression, string expected)
    {
        var calc = new CalculatorTool();
        Assert.Equal(expected, calc.Execute(expression));
    }

    // ============================
    // CalculatorTool: Decimal Numbers
    // ============================

    [Fact]
    public void Calculator_DecimalInput_ReturnsDecimal()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("3.5 + 1.5");
        Assert.Equal("5", result); // 5.0 formatted as whole number
    }

    [Fact]
    public void Calculator_DecimalResult_NotWholeNumber()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("10 / 3");

        // Should return a decimal representation
        Assert.True(double.TryParse(result, System.Globalization.NumberStyles.Any,
            System.Globalization.CultureInfo.InvariantCulture, out var value));
        Assert.Equal(10.0 / 3.0, value, 0.0001);
    }

    // ============================
    // CalculatorTool: Division by Zero
    // ============================

    [Fact]
    public void Calculator_DivisionByZero_ReturnsError()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("1 / 0");

        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
    }

    // ============================
    // CalculatorTool: Edge Cases
    // ============================

    [Fact]
    public void Calculator_EmptyInput_ReturnsError()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("");
        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Calculator_WhitespaceInput_ReturnsError()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("   ");
        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Calculator_InvalidExpression_ReturnsError()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("hello world");
        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Calculator_UnbalancedParentheses_ReturnsError()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("(2 + 3");
        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
    }

    // ============================
    // CalculatorTool: Properties
    // ============================

    [Fact]
    public void Calculator_Name_IsCalculator()
    {
        var calc = new CalculatorTool();
        Assert.Equal("Calculator", calc.Name);
    }

    [Fact]
    public void Calculator_Description_NotEmpty()
    {
        var calc = new CalculatorTool();
        Assert.False(string.IsNullOrWhiteSpace(calc.Description));
    }

    // ============================
    // CalculatorTool: Large Numbers
    // ============================

    [Fact]
    public void Calculator_LargeNumbers_HandlesCorrectly()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("1000000 * 1000000");

        Assert.True(long.TryParse(result, out var value));
        Assert.Equal(1_000_000_000_000L, value);
    }

    [Fact]
    public void Calculator_NegativeNumbers_HandlesCorrectly()
    {
        var calc = new CalculatorTool();
        var result = calc.Execute("-5 + 3");

        Assert.True(double.TryParse(result, System.Globalization.NumberStyles.Any,
            System.Globalization.CultureInfo.InvariantCulture, out var value));
        Assert.Equal(-2.0, value, 0.0001);
    }

    // ============================
    // CalculatorTool: Complex Expressions
    // ============================

    [Fact]
    public void Calculator_ComplexExpression_Correct()
    {
        var calc = new CalculatorTool();
        // (2 + 3) * (4 - 1) + 10 / 2 = 5 * 3 + 5 = 15 + 5 = 20
        var result = calc.Execute("(2 + 3) * (4 - 1) + 10 / 2");
        Assert.Equal("20", result);
    }

    // ============================
    // CalculatorTool: Whole Number Formatting
    // ============================

    [Theory]
    [InlineData("4.0 + 0", "4")]    // Whole number result formatted without decimals
    [InlineData("2.5 + 2.5", "5")]  // 5.0 -> "5"
    [InlineData("10.0 / 2", "5")]   // 5.0 -> "5"
    public void Calculator_WholeNumberResults_NoDecimals(string expression, string expected)
    {
        var calc = new CalculatorTool();
        Assert.Equal(expected, calc.Execute(expression));
    }
}
