using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for detecting logical contradictions in reasoning chains.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A contradiction detector finds statements that conflict with each other.
/// For example:
/// - Step 2 says: "x is greater than 10"
/// - Step 5 says: "x equals 5"
/// - **Contradiction!** x can't be both greater than 10 and equal to 5.
///
/// Contradictions indicate logical errors in reasoning. Finding and fixing them is crucial for:
/// - Mathematical proofs
/// - Logical reasoning
/// - Scientific analysis
/// - Any task requiring consistency
///
/// The detector can identify contradictions and suggest which steps need revision.
/// </para>
/// </remarks>
public interface IContradictionDetector<T>
{
    /// <summary>
    /// Detects contradictions within a reasoning chain.
    /// </summary>
    /// <param name="chain">The reasoning chain to analyze.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected contradictions with details.</returns>
    Task<List<Contradiction>> DetectContradictionsAsync(
        ReasoningChain<T> chain,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if two specific reasoning steps contradict each other.
    /// </summary>
    /// <param name="step1">First reasoning step.</param>
    /// <param name="step2">Second reasoning step.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if steps contradict, false otherwise.</returns>
    Task<bool> AreContradictoryAsync(
        ReasoningStep<T> step1,
        ReasoningStep<T> step2,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a detected contradiction between reasoning steps.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class stores information about a contradiction that was found,
/// including which steps conflict and why.
/// </para>
/// </remarks>
public class Contradiction
{
    /// <summary>
    /// The step number of the first conflicting statement.
    /// </summary>
    public int Step1Number { get; set; }

    /// <summary>
    /// The step number of the second conflicting statement.
    /// </summary>
    public int Step2Number { get; set; }

    /// <summary>
    /// Description of why these steps contradict each other.
    /// </summary>
    public string Explanation { get; set; } = string.Empty;

    /// <summary>
    /// Severity of the contradiction (0.0 = minor inconsistency, 1.0 = direct logical contradiction).
    /// </summary>
    public double Severity { get; set; }

    public override string ToString() =>
        $"Contradiction between steps {Step1Number} and {Step2Number}: {Explanation} (Severity: {Severity:F2})";
}
