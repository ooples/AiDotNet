namespace AiDotNet.Reasoning.Models;

/// <summary>
/// Configuration options for reasoning strategies that control how problems are solved.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Think of this class as a control panel with knobs and switches that adjust
/// how the AI thinks about problems. Just like you might adjust the difficulty level in a video game,
/// these settings let you control things like:
/// - How many steps the AI should take when thinking
/// - How thoroughly it should explore different solution paths
/// - Whether it should verify its work
/// - How much computing power to use
///
/// Different problems might need different settings. A simple math problem might only need a few steps,
/// while a complex reasoning task might benefit from exploring many different approaches.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Quick reasoning for simple problems
/// var quickConfig = new ReasoningConfig
/// {
///     MaxSteps = 3,
///     ExplorationDepth = 1,
///     EnableVerification = false
/// };
///
/// // Deep reasoning for complex problems
/// var deepConfig = new ReasoningConfig
/// {
///     MaxSteps = 10,
///     ExplorationDepth = 4,
///     BranchingFactor = 5,
///     EnableVerification = true,
///     EnableSelfRefinement = true
/// };
/// </code>
/// </para>
/// </remarks>
public class ReasoningConfig
{
    /// <summary>
    /// Maximum number of reasoning steps to generate.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many "steps" the AI can take when solving a problem.
    /// Think of it like limiting how many lines you can write when showing your work on a math problem.
    ///
    /// More steps allow for more detailed reasoning but take more time and resources.
    /// Typical values:
    /// - Simple problems: 3-5 steps
    /// - Moderate complexity: 5-10 steps
    /// - Complex problems: 10-20 steps
    /// </para>
    /// </remarks>
    public int MaxSteps { get; set; } = 10;

    /// <summary>
    /// Maximum depth for tree-based reasoning strategies (Tree-of-Thoughts, MCTS).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using tree-based reasoning (exploring multiple solution paths),
    /// this controls how "deep" the tree can grow. Imagine planning chess moves:
    /// - Depth 1: Consider only your next move
    /// - Depth 3: Consider your move, opponent's response, and your counter-response
    /// - Depth 5: Look even further ahead
    ///
    /// Deeper exploration finds better solutions but requires more computation.
    /// Typical values: 2-5 for most problems, 5-10 for very complex reasoning.
    /// </para>
    /// </remarks>
    public int ExplorationDepth { get; set; } = 3;

    /// <summary>
    /// Number of alternative thoughts to generate at each step (for Tree-of-Thoughts).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many different directions the AI considers at each step.
    /// Think of it like brainstorming: instead of following just one idea, you explore multiple alternatives.
    ///
    /// For example, when solving "How to reduce carbon emissions?", the AI might explore:
    /// - Branch 1: Renewable energy solutions
    /// - Branch 2: Transportation improvements
    /// - Branch 3: Industrial process changes
    ///
    /// More branches mean more comprehensive exploration but higher computational cost.
    /// Typical values: 2-5 branches per step.
    /// </para>
    /// </remarks>
    public int BranchingFactor { get; set; } = 3;

    /// <summary>
    /// Number of independent reasoning attempts for self-consistency (majority voting).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like solving a problem multiple times independently and then
    /// comparing answers. If you solve a math problem 5 different ways and 4 of them give you the same
    /// answer, you can be more confident that answer is correct.
    ///
    /// Self-consistency helps filter out random errors or "lucky guesses" by the AI.
    /// Typical values:
    /// - Quick checks: 3-5 attempts
    /// - High confidence needed: 10-20 attempts
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 5;

    /// <summary>
    /// Temperature for sampling diverse reasoning paths (0.0 = deterministic, 1.0+ = creative).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls how "creative" or "random" the AI's thinking is.
    /// Think of it like a creativity dial:
    /// - Temperature 0.0: Always chooses the most likely next step (deterministic, consistent)
    /// - Temperature 0.7: Balanced between consistency and creativity (good default)
    /// - Temperature 1.0+: More creative and varied, explores unusual paths
    ///
    /// Lower temperatures are better for math problems where you want consistent, logical steps.
    /// Higher temperatures are better for brainstorming or creative problem-solving.
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 0.7;

    /// <summary>
    /// Beam width for beam search algorithms.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beam search keeps track of the N most promising solution paths at once.
    /// Think of it like hiking with friends: instead of everyone following one trail, you split up to
    /// explore the 5 most promising trails simultaneously, then compare results.
    ///
    /// Larger beam width = more thorough exploration but higher memory and computation cost.
    /// Typical values: 3-10 for most problems.
    /// </para>
    /// </remarks>
    public int BeamWidth { get; set; } = 5;

    /// <summary>
    /// Whether to enable step-by-step verification with critic models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, each reasoning step is reviewed by a "critic" (another AI model)
    /// that checks if the step is valid, logical, and well-supported. Think of it like having a peer
    /// review your work before submitting it.
    ///
    /// Verification improves accuracy but takes extra time.
    /// Enable for: important decisions, complex reasoning, high-stakes problems
    /// Disable for: quick answers, simple problems, exploratory analysis
    /// </para>
    /// </remarks>
    public bool EnableVerification { get; set; } = false;

    /// <summary>
    /// Whether to enable self-refinement when verification fails.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When a reasoning step fails verification, self-refinement allows the AI
    /// to revise and improve that step based on critic feedback. It's like getting your homework corrected
    /// and then rewriting it to fix the mistakes.
    ///
    /// This only works if EnableVerification is true.
    /// Refinement improves final quality but adds processing time.
    /// </para>
    /// </remarks>
    public bool EnableSelfRefinement { get; set; } = false;

    /// <summary>
    /// Maximum number of refinement attempts per step.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how many times the AI can try to fix a failing step.
    /// It prevents getting stuck in endless revision loops.
    ///
    /// Think of it like homework revisions: you might allow 2-3 rewrites, but not unlimited attempts.
    /// Typical values: 1-3 refinement attempts.
    /// </para>
    /// </remarks>
    public int MaxRefinementAttempts { get; set; } = 2;

    /// <summary>
    /// Minimum verification score to accept a reasoning step (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The critic model gives each reasoning step a score from 0.0 (terrible)
    /// to 1.0 (perfect). This threshold determines what score is "good enough" to accept.
    ///
    /// Think of it like a grading threshold:
    /// - 0.9: Very strict, like requiring an A grade (90%+)
    /// - 0.7: Moderate, like accepting a C grade (70%+)
    /// - 0.5: Lenient, like accepting a barely-passing grade
    ///
    /// Higher thresholds give more reliable reasoning but might reject valid steps.
    /// Lower thresholds are more permissive but might accept flawed reasoning.
    /// </para>
    /// </remarks>
    public double VerificationThreshold { get; set; } = 0.7;

    /// <summary>
    /// Whether to enable external tool verification (calculators, code execution, etc.).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, mathematical calculations and code outputs are verified
    /// by actually running them through real tools (like a calculator or code interpreter) rather than
    /// just trusting the AI's answer.
    ///
    /// This is like checking your work with a calculator instead of just assuming your mental math is correct.
    /// Highly recommended for math, code, and scientific reasoning tasks.
    /// </para>
    /// </remarks>
    public bool EnableExternalVerification { get; set; } = false;

    /// <summary>
    /// Whether to enable test-time compute scaling (adaptive computation based on problem difficulty).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This feature automatically allocates more thinking time and resources
    /// to harder problems. It's like how you spend more time on difficult homework questions than easy ones.
    ///
    /// When enabled, the system detects problem difficulty and adjusts:
    /// - Simple problems: Quick, shallow reasoning
    /// - Hard problems: Deep exploration, multiple attempts, thorough verification
    ///
    /// This mirrors how models like GPT-o1 and DeepSeek-R1 work.
    /// </para>
    /// </remarks>
    public bool EnableTestTimeCompute { get; set; } = false;

    /// <summary>
    /// Multiplier for compute resources based on estimated problem difficulty (1.0 = baseline, 2.0 = double).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how much extra compute to allocate for difficult problems.
    /// Think of it as a "difficulty multiplier" for your thinking budget:
    /// - 1.0: Use normal amount of thinking time
    /// - 2.0: Use twice as much thinking time for hard problems
    /// - 3.0: Use triple the thinking time
    ///
    /// Higher values allow deeper reasoning on hard problems but increase cost.
    /// Only applies when EnableTestTimeCompute is true.
    /// </para>
    /// </remarks>
    public double ComputeScalingFactor { get; set; } = 2.0;

    /// <summary>
    /// Maximum total reasoning time in seconds (0 = no limit).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets a time limit on reasoning, like a timeout on an exam.
    /// The AI must provide an answer within this time, even if it hasn't fully explored all possibilities.
    ///
    /// Setting a timeout is important for production systems to prevent hanging on difficult problems.
    /// Typical values:
    /// - Interactive applications: 5-30 seconds
    /// - Batch processing: 60-300 seconds
    /// - No limit: 0 (use with caution)
    /// </para>
    /// </remarks>
    public int MaxReasoningTimeSeconds { get; set; } = 60;

    /// <summary>
    /// Whether to enable contradiction detection across reasoning steps.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the system checks if different reasoning steps contradict
    /// each other. For example, if step 2 says "X is greater than 10" and step 5 says "X equals 5",
    /// that's a contradiction that needs to be resolved.
    ///
    /// This helps ensure logical consistency throughout the reasoning process.
    /// Recommended for: logical reasoning, mathematical proofs, scientific analysis
    /// </para>
    /// </remarks>
    public bool EnableContradictionDetection { get; set; } = false;

    /// <summary>
    /// Whether to enable diversity sampling for exploring varied reasoning paths.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diversity sampling encourages the AI to explore different types of
    /// reasoning approaches rather than repeatedly trying similar strategies. Think of it like
    /// brainstorming rules: instead of listing 10 similar ideas, you're encouraged to come up with
    /// fundamentally different approaches.
    ///
    /// This is especially useful for creative problem-solving and when you want comprehensive coverage
    /// of possible solutions.
    /// </para>
    /// </remarks>
    public bool EnableDiversitySampling { get; set; } = false;

}
