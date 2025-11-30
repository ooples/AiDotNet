namespace AiDotNet.JitCompiler.Testing;

/// <summary>
/// Extension methods for gradient verification.
/// </summary>
public static class GradientVerificationExtensions
{
    /// <summary>
    /// Runs gradient verification and prints results to console.
    /// </summary>
    public static void RunAndPrint<T>(this GradientVerification<T>.VerificationResult result)
    {
        Console.WriteLine(result.ToString());
        Console.WriteLine();

        foreach (var error in result.Errors)
        {
            Console.WriteLine(error);
        }
    }
}
