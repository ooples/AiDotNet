using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for self-refinement engines that improve reasoning based on critic feedback.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A self-refinement engine is like rewriting your essay after getting
/// teacher feedback. When a critic points out problems with a reasoning step, the refinement engine:
/// 1. Understands what went wrong
/// 2. Generates an improved version
/// 3. Checks if the new version is better
/// 4. May iterate multiple times until it's good enough
///
/// This is a key component of advanced reasoning systems like DeepSeek-R1, which continuously
/// refine their reasoning until it passes verification.
///
/// The refinement process mirrors how humans improve their thinking - we make mistakes, get feedback,
/// and revise our approach.
/// </para>
/// </remarks>
public interface ISelfRefinementEngine<T>
{
    /// <summary>
    /// Refines a reasoning step based on critic feedback.
    /// </summary>
    /// <param name="step">The reasoning step to refine.</param>
    /// <param name="critique">The critique feedback to address.</param>
    /// <param name="context">Context for refinement.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Refined reasoning step.</returns>
    Task<ReasoningStep<T>> RefineStepAsync(
        ReasoningStep<T> step,
        CritiqueResult<T> critique,
        ReasoningContext context,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Refines an entire reasoning chain iteratively until it passes verification.
    /// </summary>
    /// <param name="chain">The reasoning chain to refine.</param>
    /// <param name="critic">The critic model to use for evaluation.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Refined reasoning chain.</returns>
    Task<ReasoningChain<T>> RefineChainAsync(
        ReasoningChain<T> chain,
        ICriticModel<T> critic,
        ReasoningConfig config,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the maximum number of refinement iterations to attempt.
    /// </summary>
    int MaxIterations { get; }
}
