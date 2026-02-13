using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning.DomainSpecific;

/// <summary>
/// Specialized reasoner for scientific problems and hypotheses.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This reasoner helps solve scientific problems using the scientific method.
///
/// **What is ScientificReasoner?**
/// A specialized reasoning system for scientific problems that follows the scientific method:
/// 1. Observation → 2. Hypothesis → 3. Prediction → 4. Experiment → 5. Analysis → 6. Conclusion
///
/// **Example - Physics Problem:**
/// ```
/// Problem: "A ball is dropped from a 20-meter tall building. How long does it take to hit the ground?"
///
/// Scientific reasoning:
/// 1. Identify known values: height = 20m, g = 9.8 m/s²
/// 2. Identify unknown: time (t)
/// 3. Choose formula: h = ½gt²
/// 4. Solve: 20 = ½(9.8)t²  →  t² = 4.08  →  t ≈ 2.02 seconds
/// 5. Verify: Units check (seconds), magnitude reasonable
/// 6. Conclusion: The ball takes approximately 2 seconds to hit the ground
/// ```
///
/// **Example - Chemistry Problem:**
/// ```
/// Problem: "Balance this equation: H₂ + O₂ → H₂O"
///
/// Scientific reasoning:
/// 1. Count atoms on each side
///    Left: H=2, O=2
///    Right: H=2, O=1
/// 2. Observe imbalance: Need 2 oxygen on right
/// 3. Add coefficient: H₂ + O₂ → 2H₂O
/// 4. Recount: Left H=2, Right H=4 (imbalanced!)
/// 5. Adjust: 2H₂ + O₂ → 2H₂O
/// 6. Verify: H: 4=4 ✓, O: 2=2 ✓
/// 7. Conclusion: Balanced equation is 2H₂ + O₂ → 2H₂O
/// ```
///
/// **Example - Biology Problem:**
/// ```
/// Problem: "Why do cells need mitochondria?"
///
/// Scientific reasoning:
/// 1. What are mitochondria? Organelles in cells
/// 2. What do they do? Produce ATP (energy)
/// 3. Why is ATP needed? Powers cellular processes
/// 4. What if no mitochondria? No energy production
/// 5. Evidence: Cells without mitochondria die or are inactive
/// 6. Conclusion: Cells need mitochondria to generate energy for survival
/// ```
///
/// **Scientific domains:**
/// - Physics (mechanics, thermodynamics, electromagnetism)
/// - Chemistry (equations, reactions, stoichiometry)
/// - Biology (cells, genetics, evolution, ecology)
/// - Earth Science (geology, meteorology, oceanography)
/// - Astronomy (celestial mechanics, cosmology)
///
/// **Reasoning patterns:**
/// - Hypothesis generation
/// - Experimental design
/// - Data analysis and interpretation
/// - Formula application
/// - Unit conversion and dimensional analysis
/// - Equation balancing
/// - Causal mechanism explanation
/// - Prediction and verification
///
/// **Key features:**
/// - Systematic problem-solving approach
/// - Emphasis on verification and validation
/// - Uses scientific notation and units
/// - Incorporates domain-specific formulas
/// - Checks for physical plausibility
/// - Multiple verification methods
///
/// **Usage example:**
/// ```csharp
/// var reasoner = new ScientificReasoner<double>(chatModel);
///
/// // Physics problem
/// var result = await reasoner.SolveAsync(
///     "Calculate the kinetic energy of a 2kg object moving at 5 m/s",
///     domain: "physics",
///     useFormulas: true
/// );
///
/// // Chemistry problem
/// result = await reasoner.SolveAsync(
///     "Balance the equation: CH₄ + O₂ → CO₂ + H₂O",
///     domain: "chemistry"
/// );
/// ```
/// </para>
/// </remarks>
public class ScientificReasoner<T>
{
    private readonly IChatModel<T> _chatModel;
    private readonly ChainOfThoughtStrategy<T> _cotStrategy;
    private readonly SelfConsistencyStrategy<T> _selfConsistencyStrategy;
    private readonly CriticModel<T>? _criticModel;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ScientificReasoner{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model to use for reasoning.</param>
    /// <param name="enableCriticalValidation">Whether to enable scientific validation with a critic model.</param>
    public ScientificReasoner(
        IChatModel<T> chatModel,
        bool enableCriticalValidation = false)
    {
        Guard.NotNull(chatModel);
        _chatModel = chatModel;
        _criticModel = enableCriticalValidation ? new CriticModel<T>(chatModel) : null;
        _numOps = MathHelper.GetNumericOperations<T>();

        _cotStrategy = new ChainOfThoughtStrategy<T>(chatModel);
        _selfConsistencyStrategy = new SelfConsistencyStrategy<T>(chatModel);
    }

    /// <summary>
    /// Solves a scientific problem using domain-specific reasoning.
    /// </summary>
    /// <param name="problem">The scientific problem to solve.</param>
    /// <param name="domain">Scientific domain (physics, chemistry, biology, etc.).</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="useFormulas">Whether to explicitly identify and use formulas.</param>
    /// <param name="useSelfConsistency">Whether to use multiple reasoning paths.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reasoning result with solution.</returns>
    public async Task<ReasoningResult<T>> SolveAsync(
        string problem,
        string domain = "general",
        ReasoningConfig? config = null,
        bool useFormulas = true,
        bool useSelfConsistency = false,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        // Enhance prompt with scientific method
        string enhancedProblem = BuildScientificPrompt(problem, domain, useFormulas);

        // Choose strategy
        IReasoningStrategy<T> strategy = useSelfConsistency
            ? _selfConsistencyStrategy
            : _cotStrategy;

        // Solve
        var result = await strategy.ReasonAsync(enhancedProblem, config, cancellationToken);

        // Scientific validation
        if (result.Success && _criticModel != null)
        {
            result = await ValidateScientificSolutionAsync(result, domain, config, cancellationToken);
        }

        return result;
    }

    /// <summary>
    /// Generates a hypothesis for an observation or phenomenon.
    /// </summary>
    public async Task<ReasoningResult<T>> GenerateHypothesisAsync(
        string observation,
        string domain = "general",
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string prompt = $@"Scientific Hypothesis Generation:

Observation: {observation}
Domain: {domain}

Generate a scientific hypothesis to explain this observation.

Follow these steps:
1. Identify the key phenomenon
2. Consider possible explanations
3. Propose a testable hypothesis
4. Suggest how to test it
5. Predict expected outcomes

Your hypothesis should be:
- Testable (can be verified experimentally)
- Specific (clear predictions)
- Based on scientific principles
- Falsifiable (can be proven wrong)

Provide your reasoning and hypothesis:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    /// <summary>
    /// Designs an experiment to test a hypothesis.
    /// </summary>
    public async Task<ReasoningResult<T>> DesignExperimentAsync(
        string hypothesis,
        string domain = "general",
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string prompt = $@"Experimental Design:

Hypothesis: {hypothesis}
Domain: {domain}

Design a scientific experiment to test this hypothesis.

Include:
1. Independent variable (what you change)
2. Dependent variable (what you measure)
3. Control variables (what you keep constant)
4. Control group (comparison baseline)
5. Experimental procedure (steps)
6. Expected results if hypothesis is correct
7. Expected results if hypothesis is incorrect

Provide a detailed experimental design:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    /// <summary>
    /// Analyzes experimental data and draws conclusions.
    /// </summary>
    public async Task<ReasoningResult<T>> AnalyzeDataAsync(
        string experimentData,
        string hypothesis,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        config ??= new ReasoningConfig();

        string prompt = $@"Scientific Data Analysis:

Hypothesis: {hypothesis}
Data: {experimentData}

Analyze the data and determine if it supports the hypothesis.

Steps:
1. Summarize the data
2. Identify patterns or trends
3. Compare with predicted outcomes
4. Calculate any necessary statistics
5. Assess uncertainty and errors
6. Draw conclusion (support/reject hypothesis)
7. Suggest further research

Provide your analysis:";

        return await _cotStrategy.ReasonAsync(prompt, config, cancellationToken);
    }

    private string BuildScientificPrompt(string problem, string domain, bool useFormulas)
    {
        string methodPrompt = @"Follow the scientific method:
1. Identify what is known and unknown
2. Determine relevant principles/laws
3. Choose appropriate formulas or approaches
4. Show your calculations step-by-step
5. Check units and dimensions
6. Verify the answer makes physical sense
7. State your conclusion clearly";

        string formulaPrompt = useFormulas
            ? "\n\nIf formulas are needed, explicitly state them before using them."
            : "";

        string domainGuidance = domain.ToLowerInvariant() switch
        {
            "physics" => "\n\nPhysics guidance: Use SI units, check dimensional consistency, verify with limiting cases.",
            "chemistry" => "\n\nChemistry guidance: Balance equations, use molar relationships, check conservation laws.",
            "biology" => "\n\nBiology guidance: Consider structure-function relationships, evolutionary context, homeostasis.",
            _ => ""
        };

        return $@"Scientific Problem ({domain}):

{problem}

{methodPrompt}{formulaPrompt}{domainGuidance}

Solve this problem systematically:";
    }

    private async Task<ReasoningResult<T>> ValidateScientificSolutionAsync(
        ReasoningResult<T> result,
        string domain,
        ReasoningConfig config,
        CancellationToken cancellationToken)
    {
        if (_criticModel == null || result.ReasoningChain == null)
            return result;

        // Build validation context
        var context = new ReasoningContext
        {
            Query = $"Validate scientific solution in {domain}",
            Domain = domain,
            PreviousSteps = new List<string>
            {
                "Check physical plausibility",
                "Verify units and dimensions",
                "Validate formula application",
                "Assess logical consistency",
                "Confirm numerical accuracy"
            }
        };

        // Critique the solution
        foreach (var step in result.ReasoningChain.Steps)
        {
            var critique = await _criticModel.CritiqueStepAsync(step, context, cancellationToken);

            if (Convert.ToDouble(critique.Score) < 0.6)
            {
                // Add warning to result
                string weakness = critique.Weaknesses.Count > 0 ? critique.Weaknesses[0] : "Quality threshold not met";
                result.Metadata["validation_warning"] = $"Step {step.StepNumber}: {weakness}";
            }
        }

        return result;
    }
}
