using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Verification;

namespace AiDotNet.Reasoning.DomainSpecific;

/// <summary>
/// Specialized reasoner for mathematical problems using verified reasoning and external verification.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MathematicalReasoner is like a math tutor that not only solves problems
/// but also checks its work with a calculator and verifies each step makes sense.
///
/// **What makes it special for math:**
/// - Uses Chain-of-Thought to show work step-by-step
/// - Verifies calculations with CalculatorVerifier
/// - Can use Self-Consistency for important problems
/// - Critic model checks mathematical reasoning
/// - Self-refinement fixes calculation errors
///
/// **Example workflow:**
/// Problem: "If John has 15 apples and gives away 40%, how many does he have left?"
///
/// 1. Generate reasoning steps
/// 2. Step 1: "Calculate 40% of 15 = 6"
///    → CalculatorVerifier: ✓ Correct
/// 3. Step 2: "Subtract: 15 - 6 = 9"
///    → CalculatorVerifier: ✓ Correct
/// 4. Step 3: "John has 9 apples left"
/// 5. All steps verified → High confidence answer
///
/// **Used for benchmarks:**
/// - GSM8K (grade school math)
/// - MATH (competition mathematics)
/// - Any mathematical reasoning tasks
/// </para>
/// </remarks>
public class MathematicalReasoner<T>
{
    private readonly IChatModel<T> _chatModel;
    private readonly ChainOfThoughtStrategy<T> _cotStrategy;
    private readonly SelfConsistencyStrategy<T> _selfConsistencyStrategy;
    private readonly CalculatorVerifier<T> _calculatorVerifier;
    private readonly CriticModel<T> _criticModel;
    private readonly SelfRefinementEngine<T> _refinementEngine;

    /// <summary>
    /// Initializes a new instance of the <see cref="MathematicalReasoner{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning.</param>
    /// <param name="tools">Optional mathematical tools (calculator, etc.).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sets up a math reasoner with all the necessary components
    /// for verified mathematical reasoning.
    /// </para>
    /// </remarks>
    public MathematicalReasoner(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _cotStrategy = new ChainOfThoughtStrategy<T>(chatModel, tools);
        _selfConsistencyStrategy = new SelfConsistencyStrategy<T>(chatModel, tools);
        _calculatorVerifier = new CalculatorVerifier<T>();
        _criticModel = new CriticModel<T>(chatModel);
        _refinementEngine = new SelfRefinementEngine<T>(chatModel);
    }

    /// <summary>
    /// Solves a mathematical problem using verified reasoning.
    /// </summary>
    /// <param name="problem">The mathematical problem to solve.</param>
    /// <param name="config">Reasoning configuration (null for default).</param>
    /// <param name="useVerification">Whether to use verification and refinement.</param>
    /// <param name="useSelfConsistency">Whether to use self-consistency (multiple attempts).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reasoning result with verified answer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method to solve math problems.
    /// You can configure how thorough you want the reasoning to be:
    /// - Basic: Just CoT without verification (fast but less reliable)
    /// - Verified: CoT with calculator checks and critic feedback (recommended)
    /// - Self-Consistency: Multiple attempts with voting (most reliable, slower)
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> SolveAsync(
        string problem,
        ReasoningConfig? config = null,
        bool useVerification = true,
        bool useSelfConsistency = false,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(problem))
            throw new ArgumentException("Problem cannot be null or empty", nameof(problem));

        config ??= new ReasoningConfig();

        // Configure for mathematical reasoning
        config.EnableExternalVerification = useVerification;
        config.EnableVerification = useVerification;
        config.EnableSelfRefinement = useVerification;

        ReasoningResult<T> result;

        // Choose strategy based on configuration
        if (useSelfConsistency)
        {
            config.NumSamples = Math.Max(config.NumSamples, 5); // At least 5 samples
            result = await _selfConsistencyStrategy.ReasonAsync(problem, config, cancellationToken);
        }
        else
        {
            result = await _cotStrategy.ReasonAsync(problem, config, cancellationToken);
        }

        // If verification enabled, verify and refine
        if (useVerification && result.Success)
        {
            result = await VerifyAndRefineAsync(result, config, cancellationToken);
        }

        // Add domain metadata
        result.Metadata["domain"] = "mathematics";
        result.Metadata["verification_enabled"] = useVerification;
        result.Metadata["self_consistency_enabled"] = useSelfConsistency;

        return result;
    }

    /// <summary>
    /// Verifies and refines the reasoning result.
    /// </summary>
    private async Task<ReasoningResult<T>> VerifyAndRefineAsync(
        ReasoningResult<T> result,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        var chain = result.ReasoningChain;
        bool hasCalculationErrors = false;
        var verificationResults = new List<string>();

        // Step 1: Verify calculations with calculator
        foreach (var step in chain.Steps)
        {
            if (_calculatorVerifier.CanVerify(step))
            {
                var verification = await _calculatorVerifier.VerifyStepAsync(step, cancellationToken);

                step.IsVerified = verification.Passed;
                step.VerificationMethod = verification.ToolUsed;
                step.ExternalVerificationResult = verification.Explanation;

                verificationResults.Add(verification.ToString());

                if (!verification.Passed)
                {
                    hasCalculationErrors = true;
                    step.Metadata["verification_failed"] = true;
                }
            }
        }

        // Step 2: If calculation errors found, try to refine
        if (hasCalculationErrors && config.EnableSelfRefinement)
        {
            chain = await RefineChainWithCalculationFeedbackAsync(
                chain,
                config,
                cancellationToken);

            result.ReasoningChain = chain;
            result.FinalAnswer = chain.FinalAnswer;
        }

        // Step 3: Critic evaluation for overall quality
        var chainCritique = await _criticModel.CritiqueChainAsync(chain, cancellationToken);

        result.OverallConfidence = chainCritique.Score;
        result.VerificationFeedback.Add(chainCritique.ToString());
        result.VerificationFeedback.AddRange(verificationResults);
        result.Metrics["calculation_errors_found"] = hasCalculationErrors;
        result.Metrics["all_calculations_verified"] = chain.Steps.All(s =>
            !_calculatorVerifier.CanVerify(s) || s.IsVerified);

        return result;
    }

    /// <summary>
    /// Refines a chain based on calculation verification feedback.
    /// </summary>
    private async Task<ReasoningChain<T>> RefineChainWithCalculationFeedbackAsync(
        ReasoningChain<T> chain,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        var refinedChain = new ReasoningChain<T>
        {
            Query = chain.Query,
            StartedAt = chain.StartedAt
        };

        for (int i = 0; i < chain.Steps.Count; i++)
        {
            var step = chain.Steps[i];

            // If step failed verification, create critique and refine
            if (step.Metadata.ContainsKey("verification_failed"))
            {
                var critique = new CritiqueResult<T>
                {
                    Score = MathHelper.GetNumericOperations<T>().FromDouble(0.3),
                    Feedback = $"Calculation error detected: {step.ExternalVerificationResult}",
                    Weaknesses = new List<string>
                    {
                        "Mathematical calculation is incorrect",
                        "Step fails external verification"
                    },
                    Suggestions = new List<string>
                    {
                        "Recalculate carefully",
                        "Show intermediate steps",
                        "Verify the arithmetic operation"
                    },
                    PassesThreshold = false
                };

                var context = new ReasoningContext
                {
                    Query = chain.Query,
                    PreviousSteps = chain.Steps.Take(i).Select(s => s.Content).ToList(),
                    Domain = "mathematics"
                };

                // Refine the step
                var refinedStep = await _refinementEngine.RefineStepAsync(
                    step,
                    critique,
                    context,
                    cancellationToken);

                // Re-verify the refined step
                if (_calculatorVerifier.CanVerify(refinedStep))
                {
                    var verification = await _calculatorVerifier.VerifyStepAsync(
                        refinedStep,
                        cancellationToken);

                    refinedStep.IsVerified = verification.Passed;
                    refinedStep.ExternalVerificationResult = verification.Explanation;
                }

                refinedChain.AddStep(refinedStep);
            }
            else
            {
                refinedChain.AddStep(step);
            }
        }

        refinedChain.FinalAnswer = chain.FinalAnswer;
        refinedChain.CompletedAt = DateTime.UtcNow;
        refinedChain.OverallScore = refinedChain.GetAverageScore();

        return refinedChain;
    }

    /// <summary>
    /// Extracts numerical answer from reasoning result for benchmark evaluation.
    /// </summary>
    /// <param name="result">The reasoning result.</param>
    /// <returns>Extracted numerical answer, or null if not found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Many math benchmarks need numerical answers (like "42")
    /// rather than full sentences. This method extracts just the number from the answer.
    /// </para>
    /// </remarks>
    public string? ExtractNumericalAnswer(ReasoningResult<T> result)
    {
        if (result == null || string.IsNullOrWhiteSpace(result.FinalAnswer))
            return null;

        // Try to extract a number from the final answer
        var match = RegexHelper.Match(
            result.FinalAnswer,
            @"-?[0-9]+\.?[0-9]*",
            System.Text.RegularExpressions.RegexOptions.None);

        return match.Success ? match.Value : null;
    }
}

