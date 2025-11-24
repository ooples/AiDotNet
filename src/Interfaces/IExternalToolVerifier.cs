using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for verifying reasoning steps using external tools (calculators, code execution, etc.).
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of just trusting what the AI says, an external tool verifier
/// actually runs calculations or code to confirm the results are correct.
///
/// Examples:
/// - **Calculator Verifier**: If the AI says "15 × 240 = 3600", actually run it through a calculator
/// - **Code Execution Verifier**: If the AI writes code that should output "Hello", actually run it
/// - **Math Proof Verifier**: Use symbolic math systems to verify algebraic manipulations
///
/// This is like showing your work and then checking it with a calculator - it catches mistakes
/// that look plausible but are actually wrong.
///
/// Highly recommended for:
/// - Mathematical calculations
/// - Code reasoning
/// - Scientific computations
/// - Any task where accuracy is critical
/// </para>
/// </remarks>
public interface IExternalToolVerifier<T>
{
    /// <summary>
    /// Verifies a reasoning step using external tools.
    /// </summary>
    /// <param name="step">The reasoning step to verify.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Verification result with pass/fail and details.</returns>
    Task<VerificationResult<T>> VerifyStepAsync(
        ReasoningStep<T> step,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the name of the external tool used for verification.
    /// </summary>
    string ToolName { get; }

    /// <summary>
    /// Checks if this verifier can handle the given reasoning step.
    /// </summary>
    /// <param name="step">The step to check.</param>
    /// <returns>True if this verifier can verify this type of step.</returns>
    bool CanVerify(ReasoningStep<T> step);
}

/// <summary>
/// Represents the result of external tool verification.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This stores the outcome of verification - did it pass or fail,
/// what was the actual result, and how confident are we in the verification?
/// </para>
/// </remarks>
public class VerificationResult<T>
{
    /// <summary>
    /// Whether the verification passed.
    /// </summary>
    public bool Passed { get; set; }

    /// <summary>
    /// The actual result from the external tool.
    /// </summary>
    public string ActualResult { get; set; } = string.Empty;

    /// <summary>
    /// The expected result (from the reasoning step).
    /// </summary>
    public string ExpectedResult { get; set; } = string.Empty;

    /// <summary>
    /// Detailed explanation of the verification outcome.
    /// </summary>
    public string Explanation { get; set; } = string.Empty;

    /// <summary>
    /// Confidence score in the verification (0.0 to 1.0).
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Name of the tool that performed the verification.
    /// </summary>
    public string ToolUsed { get; set; } = string.Empty;

    public override string ToString() =>
        $"Verification {(Passed ? "PASSED" : "FAILED")}: {Explanation} (Tool: {ToolUsed})";
}
