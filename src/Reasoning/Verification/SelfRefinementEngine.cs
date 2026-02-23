using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Implements self-refinement by iteratively improving reasoning based on critic feedback.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Self-refinement is like rewriting an essay after getting teacher feedback.
/// When a critic identifies problems in reasoning, the refinement engine:
/// 1. Understands what needs improvement
/// 2. Generates an improved version
/// 3. Gets it critiqued again
/// 4. Repeats until it's good enough (or max attempts reached)
///
/// This is a key technique in advanced reasoning systems like DeepSeek-R1, which continuously
/// refine their reasoning until it passes verification.
///
/// **Example:**
/// Original step: "15 * 240 = 360"
/// Critique: Score 0.3, "Calculation error - check your multiplication"
///
/// Refinement attempt 1: "15 * 240 = 3600"
/// Critique: Score 0.4, "Still incorrect. Remember: 15% means 0.15"
///
/// Refinement attempt 2: "15% of 240: 0.15 * 240 = 36"
/// Critique: Score 0.95, "Correct! Clear calculation."
/// ✓ Passes threshold
/// </para>
/// </remarks>
internal class SelfRefinementEngine<T> : ISelfRefinementEngine<T>
{
    private readonly IChatModel<T> _chatModel;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfRefinementEngine{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for refinement.</param>
    /// <param name="maxIterations">Maximum refinement attempts per step (default: 3).</param>
    public SelfRefinementEngine(IChatModel<T> chatModel, int maxIterations = 3)
    {
        Guard.NotNull(chatModel);
        _chatModel = chatModel;
        _maxIterations = maxIterations > 0 ? maxIterations : 3;
    }

    /// <inheritdoc/>
    public int MaxIterations => _maxIterations;

    /// <inheritdoc/>
    public async Task<ReasoningStep<T>> RefineStepAsync(
        ReasoningStep<T> step,
        CritiqueResult<T> critique,
        ReasoningContext context,
        CancellationToken cancellationToken = default)
    {
        if (step == null)
            throw new ArgumentNullException(nameof(step));

        if (critique == null)
            throw new ArgumentNullException(nameof(critique));

        if (context == null)
            throw new ArgumentNullException(nameof(context));

        // Build refinement prompt
        string prompt = BuildRefinementPrompt(step, critique, context);

        // Generate refined version
        string refinedContent = await _chatModel.GenerateResponseAsync(prompt);

        // Create refined step
        var refinedStep = new ReasoningStep<T>
        {
            StepNumber = step.StepNumber,
            Content = refinedContent.Trim(),
            OriginalContent = step.OriginalContent ?? step.Content, // Preserve original if not already set
            Score = step.Score, // Will be updated by next critique
            RefinementCount = step.RefinementCount + 1,
            CreatedAt = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>(step.Metadata)
        };

        refinedStep.Metadata["refinement_feedback"] = critique.Feedback;
        refinedStep.Metadata["refinement_iteration"] = step.RefinementCount + 1;

        return refinedStep;
    }

    /// <inheritdoc/>
    public async Task<ReasoningChain<T>> RefineChainAsync(
        ReasoningChain<T> chain,
        ICriticModel<T> critic,
        ReasoningConfig config,
        CancellationToken cancellationToken = default)
    {
        if (chain == null)
            throw new ArgumentNullException(nameof(chain));

        if (critic == null)
            throw new ArgumentNullException(nameof(critic));

        var refinedChain = new ReasoningChain<T>
        {
            Query = chain.Query,
            StartedAt = chain.StartedAt
        };

        // Refine each step that needs improvement
        for (int i = 0; i < chain.Steps.Count; i++)
        {
            var step = chain.Steps[i];
            var currentStep = step;

            // Build context for this step
            var context = new ReasoningContext
            {
                Query = chain.Query,
                PreviousSteps = chain.Steps.Take(i).Select(s => s.Content).ToList(),
                Domain = chain.Metadata.ContainsKey("domain") ? chain.Metadata["domain"].ToString() ?? "" : ""
            };

            // Refine this step up to maxIterations times
            for (int attempt = 0; attempt < _maxIterations; attempt++)
            {
                // Get critique
                var critique = await critic.CritiqueStepAsync(currentStep, context, cancellationToken);

                // Update step score
                currentStep.Score = critique.Score;
                currentStep.IsVerified = true;
                currentStep.CriticFeedback = critique.Feedback;

                // If it passes, we're done with this step
                if (critique.PassesThreshold)
                {
                    break;
                }

                // If max attempts reached, use what we have
                if (attempt >= _maxIterations - 1)
                {
                    break;
                }

                // Refine the step
                currentStep = await RefineStepAsync(currentStep, critique, context, cancellationToken);
            }

            refinedChain.AddStep(currentStep);
        }

        refinedChain.FinalAnswer = chain.FinalAnswer;
        refinedChain.CompletedAt = DateTime.UtcNow;
        refinedChain.OverallScore = refinedChain.GetAverageScore();

        return refinedChain;
    }

    /// <summary>
    /// Builds a prompt for refining a reasoning step based on critique.
    /// </summary>
    private string BuildRefinementPrompt(ReasoningStep<T> step, CritiqueResult<T> critique, ReasoningContext context)
    {
        string previousStepsText = context.PreviousSteps.Count > 0
            ? $"\n\nPrevious steps:\n{string.Join("\n", context.PreviousSteps.Select((s, i) => $"{i + 1}. {s}"))}"
            : "";

        string weaknessesText = critique.Weaknesses.Count > 0
            ? $"\n\nIdentified weaknesses:\n{string.Join("\n", critique.Weaknesses.Select(w => $"- {w}"))}"
            : "";

        string suggestionsText = critique.Suggestions.Count > 0
            ? $"\n\nSuggestions for improvement:\n{string.Join("\n", critique.Suggestions.Select(s => $"- {s}"))}"
            : "";

        return $@"You are refining a reasoning step that received critical feedback.

Original problem: {context.Query}{previousStepsText}

Current step (needs improvement):
Step {step.StepNumber}: {step.Content}

Critique feedback:
{critique.Feedback}
Score: {critique.Score}{weaknessesText}{suggestionsText}

Task: Rewrite this reasoning step to address the identified issues while maintaining correctness.

Guidelines:
- Address all weaknesses mentioned
- Incorporate the suggestions
- Keep the step clear and concise
- Ensure logical soundness
- Maintain consistency with previous steps

Provide ONLY the refined step content (no preamble or explanation):";
    }
}
