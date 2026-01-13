using System.Data;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Verifies mathematical calculations in reasoning steps using actual computation.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The calculator verifier checks mathematical claims by actually
/// performing the calculations. Instead of trusting what the AI says, we verify it with real math.
///
/// **Example:**
/// Step claims: "15% of 240 is 36"
///
/// Verifier:
/// 1. Extracts the calculation: 15% of 240
/// 2. Computes: 0.15 × 240 = 36
/// 3. Compares with claimed answer: 36
/// 4. Result: ✓ PASSED - Calculation is correct
///
/// **Why it's important:**
/// - LLMs sometimes make calculation errors
/// - Especially common with multi-step math
/// - External verification catches these errors
/// - Critical for mathematical reasoning tasks
///
/// **Supports:**
/// - Basic arithmetic (+, -, *, /, ^)
/// - Percentages
/// - Parentheses and order of operations
/// - Multi-step expressions
/// </para>
/// </remarks>
internal class CalculatorVerifier<T> : IExternalToolVerifier<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="CalculatorVerifier{T}"/> class.
    /// </summary>
    public CalculatorVerifier()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <inheritdoc/>
    public string ToolName => "Calculator";

    /// <inheritdoc/>
    public bool CanVerify(ReasoningStep<T> step)
    {
        if (step == null || string.IsNullOrWhiteSpace(step.Content))
            return false;

        string content = step.Content.ToLowerInvariant();

        // Check for mathematical content
        return content.Contains("calculate") ||
               content.Contains("=") ||
               content.Contains("×") ||
               content.Contains("÷") ||
               RegexHelper.IsMatch(content, @"\d+\s*[\+\-\*/]\s*\d+", RegexOptions.None) || // arithmetic expressions
               RegexHelper.IsMatch(content, @"\d+%", RegexOptions.None) || // percentages
               RegexHelper.IsMatch(content, @"(?:sum|product|quotient|difference)", RegexOptions.None);
    }

    /// <inheritdoc/>
    public async Task<VerificationResult<T>> VerifyStepAsync(
        ReasoningStep<T> step,
        CancellationToken cancellationToken = default)
    {
        if (step == null)
            throw new ArgumentNullException(nameof(step));

        var result = new VerificationResult<T>
        {
            ToolUsed = ToolName
        };

        // Extract mathematical expressions and claimed results
        var expressions = ExtractMathematicalExpressions(step.Content);

        if (expressions.Count == 0)
        {
            result.Passed = true;
            result.Confidence = _numOps.FromDouble(0.5);
            result.Explanation = "No verifiable calculations found in this step";
            return result;
        }

        // Verify each expression
        var verifications = new List<(string expr, string expected, string actual, bool matches)>();

        foreach (var (expression, expectedResult) in expressions)
        {
            try
            {
                string actualResult = EvaluateExpression(expression);
                bool matches = CompareResults(expectedResult, actualResult);

                verifications.Add((expression, expectedResult, actualResult, matches));
            }
            catch (Exception ex)
            {
                verifications.Add((expression, expectedResult, $"Error: {ex.Message}", false));
            }
        }

        // Determine overall pass/fail
        int totalChecks = verifications.Count;
        int passedChecks = verifications.Count(v => v.matches);

        result.Passed = passedChecks == totalChecks;
        result.Confidence = _numOps.FromDouble((double)passedChecks / totalChecks);

        // Build explanation
        var explanationParts = new List<string>();
        foreach (var (expr, expected, actual, matches) in verifications)
        {
            string status = matches ? "✓" : "✗";
            explanationParts.Add($"{status} {expr} = {actual} (claimed: {expected})");
        }

        result.Explanation = string.Join("\n", explanationParts);
        result.ActualResult = string.Join("; ", verifications.Select(v => v.actual));
        result.ExpectedResult = string.Join("; ", verifications.Select(v => v.expected));

        return await Task.FromResult(result);
    }

    /// <summary>
    /// Extracts mathematical expressions and their claimed results from text.
    /// </summary>
    private List<(string expression, string result)> ExtractMathematicalExpressions(string text)
    {
        var expressions = new List<(string, string)>();

        // Pattern 1: "X = Y" style
        var equalsMatches = RegexHelper.Matches(text, @"([0-9\.\+\-\*/\(\)\^%\s×÷]+)\s*=\s*([0-9\.]+)", RegexOptions.None);
        foreach (Match match in equalsMatches)
        {
            string expr = match.Groups[1].Value.Trim();
            string result = match.Groups[2].Value.Trim();

            // Clean up expression
            expr = expr.Replace("×", "*").Replace("÷", "/");

            expressions.Add((expr, result));
        }

        // Pattern 2: "X% of Y"
        var percentMatches = RegexHelper.Matches(text, @"([0-9\.]+)%\s+of\s+([0-9\.]+)", RegexOptions.None);
        foreach (Match match in percentMatches)
        {
            double percent = double.Parse(match.Groups[1].Value);
            double number = double.Parse(match.Groups[2].Value);
            string expr = $"{percent / 100} * {number}";

            // Try to find the claimed result nearby
            var resultMatch = RegexHelper.Match(text, $@"{RegexHelper.Escape(match.Value)}[^\d]*([0-9\.]+)", RegexOptions.None);
            if (resultMatch.Success)
            {
                string result = resultMatch.Groups[1].Value;
                expressions.Add((expr, result));
            }
        }

        return expressions;
    }

    /// <summary>
    /// Evaluates a mathematical expression.
    /// </summary>
    private string EvaluateExpression(string expression)
    {
        try
        {
            // Clean up the expression
            expression = expression.Trim()
                .Replace("^", "**")  // Power operator (not natively supported by DataTable)
                .Replace(",", "");   // Remove comma separators

            // Handle power operator manually if present
            if (expression.Contains("**"))
            {
                expression = EvaluatePowerOperations(expression);
            }

            // Use DataTable to evaluate (supports +, -, *, /, ())
            var table = new DataTable();
            var result = table.Compute(expression, null);

            // Format result
            double value = Convert.ToDouble(result);

            // Round to reasonable precision
            if (Math.Abs(value - Math.Round(value)) < 0.0001)
            {
                return Math.Round(value).ToString();
            }
            else
            {
                return Math.Round(value, 4).ToString();
            }
        }
        catch
        {
            throw new InvalidOperationException($"Could not evaluate expression: {expression}");
        }
    }

    /// <summary>
    /// Evaluates power operations (e.g., 2**3 = 8) before passing to DataTable.
    /// </summary>
    private string EvaluatePowerOperations(string expression)
    {
        while (expression.Contains("**"))
        {
            var match = RegexHelper.Match(expression, @"([0-9\.]+)\s*\*\*\s*([0-9\.]+)", RegexOptions.None);
            if (!match.Success) break;

            double baseNum = double.Parse(match.Groups[1].Value);
            double exponent = double.Parse(match.Groups[2].Value);
            double result = Math.Pow(baseNum, exponent);

            expression = expression.Replace(match.Value, result.ToString());
        }

        return expression;
    }

    /// <summary>
    /// Compares expected and actual results with tolerance for floating point.
    /// </summary>
    private bool CompareResults(string expected, string actual)
    {
        if (!double.TryParse(expected, out double expectedValue))
            return false;

        if (!double.TryParse(actual, out double actualValue))
            return false;

        // Allow small floating point differences
        const double tolerance = 0.01;
        return Math.Abs(expectedValue - actualValue) <= tolerance;
    }
}



