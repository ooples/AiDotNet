using AiDotNet.Autodiff.Testing;

namespace AiDotNet.JitCompiler.Testing;

/// <summary>
/// Extension methods for gradient verification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These extension methods provide convenient ways to print
/// gradient verification results to the console for debugging purposes.
/// </para>
/// </remarks>
public static class GradientVerificationExtensions
{
    /// <summary>
    /// Runs gradient comparison and prints results to console.
    /// </summary>
    /// <typeparam name="T">The numeric type used in verification.</typeparam>
    /// <param name="result">The comparison result to print.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method prints a summary of the gradient verification
    /// including overall pass/fail status and any specific errors found.
    /// </para>
    /// </remarks>
    public static void RunAndPrint<T>(this NumericalGradient<T>.ComparisonResult result)
    {
        Console.WriteLine($"Gradient Verification: {(result.Passed ? "PASSED" : "FAILED")}");
        Console.WriteLine($"Max Relative Error: {result.MaxRelativeError:E4}");
        Console.WriteLine($"Average Relative Error: {result.AverageRelativeError:E4}");
        Console.WriteLine($"Failed/Total: {result.FailedElements}/{result.TotalElementsChecked}");
        Console.WriteLine();

        if (result.Errors.Count > 0)
        {
            Console.WriteLine($"Errors ({result.Errors.Count}):");
            foreach (var error in result.Errors)
            {
                Console.WriteLine($"  {error}");
            }
        }
    }
}
