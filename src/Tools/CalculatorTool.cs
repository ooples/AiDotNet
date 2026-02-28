using System.Data;
using System.Globalization;
using AiDotNet.Interfaces;

namespace AiDotNet.Tools;

/// <summary>
/// A tool that performs mathematical calculations using expression evaluation.
/// This tool can evaluate arithmetic expressions including basic operations, parentheses, and common functions.
/// </summary>
/// <remarks>
/// For Beginners:
/// The CalculatorTool is like a smart calculator that can understand and evaluate mathematical expressions
/// written as text. Instead of pressing buttons, you give it a string like "2 + 2" or "(10 * 5) + 3" and
/// it calculates the result.
///
/// This tool uses .NET's DataTable.Compute() method, which can evaluate mathematical expressions safely.
/// It supports:
/// - Basic arithmetic: +, -, *, /
/// - Parentheses for grouping: (2 + 3) * 4
/// - Decimal numbers: 3.14 * 2
///
/// Example usage:
/// <code>
/// var calculator = new CalculatorTool();
/// string result1 = calculator.Execute("5 + 3");        // Returns "8"
/// string result2 = calculator.Execute("(10 - 2) * 4"); // Returns "32"
/// string result3 = calculator.Execute("100 / 4");      // Returns "25"
/// </code>
///
/// Note: For more advanced mathematical functions (sin, cos, sqrt, etc.), you would need to extend
/// this implementation or use a more sophisticated expression parser.
/// </remarks>
public class CalculatorTool : ITool
{
    /// <inheritdoc/>
    public string Name => "Calculator";

    /// <inheritdoc/>
    public string Description =>
        "Performs mathematical calculations. " +
        "Input should be a valid mathematical expression using operators +, -, *, /, and parentheses. " +
        "Examples: '2 + 2', '(10 * 5) - 3', '100 / 4'. " +
        "Returns the calculated result as a number.";

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Input expression cannot be empty.";
        }

        try
        {
            // Clean up the input to handle common mathematical notations
            string processedInput = PreprocessExpression(input);

            // Use DataTable.Compute to evaluate the expression with invariant culture
            using (var dataTable = new DataTable { Locale = CultureInfo.InvariantCulture })
            {
                var result = dataTable.Compute(processedInput, string.Empty);

                // Convert the result to a string
                if (result == null || result == DBNull.Value)
                {
                    return "Error: Calculation produced no result.";
                }

                // Convert result to double for consistent formatting
                double doubleResult;
                if (result is double d)
                {
                    doubleResult = d;
                }
                else if (result is decimal dec)
                {
                    doubleResult = (double)dec;
                }
                else if (result is int i)
                {
                    doubleResult = i;
                }
                else if (result is long l)
                {
                    doubleResult = l;
                }
                else
                {
                    // Try parsing the string representation as a fallback
                    string resultStr = result.ToString() ?? string.Empty;
                    if (!double.TryParse(resultStr, NumberStyles.Any, CultureInfo.InvariantCulture, out doubleResult))
                    {
                        return $"Error: Could not interpret result '{resultStr}'.";
                    }
                }

                // Check for infinity (e.g., division by zero)
                if (double.IsInfinity(doubleResult))
                {
                    return "Error: Division by zero is not allowed.";
                }

                // Check for NaN
                if (double.IsNaN(doubleResult))
                {
                    return "Error: Calculation produced an undefined result.";
                }

                // Remove unnecessary decimal places for whole numbers
                const double epsilon = 1e-9;
                if (Math.Abs(doubleResult - Math.Floor(doubleResult)) < epsilon)
                {
                    return ((long)doubleResult).ToString(CultureInfo.InvariantCulture);
                }

                return doubleResult.ToString("G", CultureInfo.InvariantCulture);
            }
        }
        catch (SyntaxErrorException ex)
        {
            return $"Error: Invalid mathematical expression. {ex.Message}";
        }
        catch (EvaluateException ex)
        {
            return $"Error: Could not evaluate expression. {ex.Message}";
        }
        catch (DivideByZeroException)
        {
            return "Error: Division by zero is not allowed.";
        }
        catch (Exception ex) when (ex is not SyntaxErrorException and not EvaluateException and not DivideByZeroException)
        {
            // Final safety net for truly unexpected exceptions (e.g., OutOfMemoryException, StackOverflowException, etc.)
            // This is intentionally generic to gracefully handle edge cases not covered by specific handlers
            return $"Error: An unexpected error occurred during calculation. {ex.Message}";
        }
    }

    /// <summary>
    /// Preprocesses the mathematical expression to handle common notations and improve compatibility.
    /// </summary>
    /// <param name="expression">The raw mathematical expression.</param>
    /// <returns>A processed expression ready for DataTable.Compute.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method cleans up the input expression to make it compatible with DataTable.Compute.
    /// It handles common variations in how people write math expressions.
    ///
    /// For example:
    /// - Removes extra whitespace
    /// - Handles common function names that might be in the input
    ///
    /// This preprocessing makes the calculator more user-friendly and robust.
    /// </remarks>
    private string PreprocessExpression(string expression)
    {
        // Trim whitespace
        expression = expression.Trim();

        // Convert large integer literals to decimal format to avoid int overflow in DataTable.Compute.
        // DataTable.Compute performs integer arithmetic for integer operands, which can overflow
        // for numbers larger than int.MaxValue (~2.1 billion). By appending ".0" to large numbers,
        // we force decimal/double arithmetic instead.
        expression = System.Text.RegularExpressions.Regex.Replace(
            expression,
            @"\b(\d{7,})\b",
            "$1.0",
            System.Text.RegularExpressions.RegexOptions.None,
            TimeSpan.FromSeconds(1));

        return expression;
    }
}
