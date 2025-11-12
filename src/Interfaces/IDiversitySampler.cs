using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for sampling diverse reasoning paths to avoid redundant exploration.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A diversity sampler ensures you explore different types of solutions
/// rather than repeatedly trying similar approaches.
///
/// Think of brainstorming rules: instead of listing "red car, blue car, green car, yellow car",
/// you want diverse ideas like "car, bicycle, train, airplane".
///
/// This prevents wasting computation on similar reasoning paths and ensures comprehensive
/// exploration of the solution space. Especially important for:
/// - Creative problem-solving
/// - Multi-faceted questions
/// - Avoiding local optima (getting stuck on one type of solution)
/// </para>
/// </remarks>
public interface IDiversitySampler<T>
{
    /// <summary>
    /// Samples a diverse set of thoughts from a larger pool of candidates.
    /// </summary>
    /// <param name="candidates">Pool of candidate thoughts to sample from.</param>
    /// <param name="numToSample">Number of diverse thoughts to select.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <returns>Diverse subset of thoughts.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Given many possible next steps, picks a smaller subset
    /// that represents different approaches rather than similar ones.
    /// </para>
    /// </remarks>
    List<ThoughtNode<T>> SampleDiverse(
        List<ThoughtNode<T>> candidates,
        int numToSample,
        ReasoningConfig config);

    /// <summary>
    /// Calculates the diversity score between two thoughts (0.0 = identical, 1.0 = completely different).
    /// </summary>
    /// <param name="thought1">First thought.</param>
    /// <param name="thought2">Second thought.</param>
    /// <returns>Diversity score.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Measures how different two thoughts are from each other.
    /// Used to ensure selected thoughts are sufficiently distinct.
    /// </para>
    /// </remarks>
    T CalculateDiversity(ThoughtNode<T> thought1, ThoughtNode<T> thought2);
}
