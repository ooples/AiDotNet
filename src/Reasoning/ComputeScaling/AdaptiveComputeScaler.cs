using AiDotNet.Helpers;
using AiDotNet.Reasoning.Models;

namespace AiDotNet.Reasoning.ComputeScaling;

/// <summary>
/// Implements adaptive test-time compute scaling based on problem difficulty.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Test-time compute scaling means allocating more "thinking time" and resources
/// to harder problems. Just like you spend more time on difficult homework questions than easy ones.
///
/// **How it works:**
/// 1. Estimate problem difficulty (simple heuristics or ML model)
/// 2. Scale reasoning config based on difficulty:
///    - Easy problems: Quick CoT with few steps
///    - Medium problems: Standard CoT with verification
///    - Hard problems: Tree-of-Thoughts with self-consistency and refinement
/// 3. Allocate compute budget accordingly
///
/// **Example:**
/// Easy problem (difficulty: 0.2):
/// - MaxSteps: 3, NumSamples: 1, No verification
/// - Time: ~2 seconds
///
/// Hard problem (difficulty: 0.9):
/// - MaxSteps: 20, NumSamples: 10, Full verification + refinement
/// - Time: ~60 seconds
///
/// **Used in:**
/// - ChatGPT o1/o3: Allocates millions of tokens for hard problems
/// - DeepSeek-R1: Uses RL to learn optimal compute allocation
/// - AlphaGo: Monte Carlo tree search with adaptive depth
///
/// **Research basis:**
/// "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
/// "Let's Think Dot by Dot" (Zelikman et al., 2024)
/// </para>
/// </remarks>
internal class AdaptiveComputeScaler
{
    private readonly ReasoningConfig _baselineConfig;
    private readonly double _maxScalingFactor;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaptiveComputeScaler"/> class.
    /// </summary>
    /// <param name="baselineConfig">Baseline configuration for medium-difficulty problems.</param>
    /// <param name="maxScalingFactor">Maximum scaling multiplier for hardest problems (default: 5.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The baseline config is what you'd use for an average problem.
    /// The scaler will reduce resources for easy problems and increase for hard ones, up to the max factor.
    /// </para>
    /// </remarks>
    public AdaptiveComputeScaler(ReasoningConfig? baselineConfig = null, double maxScalingFactor = 5.0)
    {
        _baselineConfig = baselineConfig ?? new ReasoningConfig();
        _maxScalingFactor = Math.Max(2.0, maxScalingFactor);
    }

    /// <summary>
    /// Scales reasoning configuration based on estimated problem difficulty.
    /// </summary>
    /// <param name="problem">The problem text.</param>
    /// <param name="estimatedDifficulty">Estimated difficulty (0.0-1.0), null to auto-estimate.</param>
    /// <returns>Scaled reasoning configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This takes a problem and returns a config that's appropriate
    /// for its difficulty level. Easy problems get lightweight configs, hard problems get heavyweight.
    /// </para>
    /// </remarks>
    public ReasoningConfig ScaleConfig(string problem, double? estimatedDifficulty = null)
    {
        // Estimate difficulty if not provided
        double difficulty = estimatedDifficulty ?? EstimateDifficulty(problem);

        // Clamp to valid range
        difficulty = MathHelper.Clamp(difficulty, 0.0, 1.0);

        // Calculate scaling factor based on difficulty
        // Easy (0.0-0.3): 0.5x to 1.0x scaling
        // Medium (0.3-0.7): 1.0x to 2.0x scaling
        // Hard (0.7-1.0): 2.0x to maxScalingFactor scaling
        double scalingFactor;
        if (difficulty < 0.3)
        {
            // Easy: reduce compute
            scalingFactor = 0.5 + (difficulty / 0.3) * 0.5; // 0.5x to 1.0x
        }
        else if (difficulty < 0.7)
        {
            // Medium: standard to moderate scaling
            scalingFactor = 1.0 + ((difficulty - 0.3) / 0.4) * 1.0; // 1.0x to 2.0x
        }
        else
        {
            // Hard: significant scaling
            scalingFactor = 2.0 + ((difficulty - 0.7) / 0.3) * (_maxScalingFactor - 2.0); // 2.0x to max
        }

        // Create scaled config
        var scaledConfig = new ReasoningConfig
        {
            // Scale step limits
            MaxSteps = (int)Math.Round(_baselineConfig.MaxSteps * scalingFactor),
            ExplorationDepth = (int)Math.Max(1, Math.Round(_baselineConfig.ExplorationDepth * scalingFactor)),
            BranchingFactor = (int)Math.Max(2, Math.Round(_baselineConfig.BranchingFactor * Math.Sqrt(scalingFactor))),

            // Scale sampling
            NumSamples = difficulty > 0.7 ? (int)Math.Round(_baselineConfig.NumSamples * scalingFactor) : _baselineConfig.NumSamples,
            BeamWidth = (int)Math.Max(2, Math.Round(_baselineConfig.BeamWidth * Math.Sqrt(scalingFactor))),

            // Temperature: lower for easier problems (more deterministic)
            Temperature = Math.Max(0.1, _baselineConfig.Temperature * (0.5 + difficulty * 0.5)),

            // Enable verification for medium+ difficulty
            EnableVerification = difficulty > 0.4,
            EnableSelfRefinement = difficulty > 0.6,
            EnableExternalVerification = difficulty > 0.5,
            EnableContradictionDetection = difficulty > 0.7,
            EnableDiversitySampling = difficulty > 0.6,

            // Refinement attempts scale with difficulty
            MaxRefinementAttempts = difficulty > 0.7 ? 3 : (difficulty > 0.5 ? 2 : 1),

            // Time budget scales
            MaxReasoningTimeSeconds = (int)Math.Round(_baselineConfig.MaxReasoningTimeSeconds * scalingFactor),

            // Other settings from baseline
            VerificationThreshold = _baselineConfig.VerificationThreshold,
            ComputeScalingFactor = scalingFactor
        };

        return scaledConfig;
    }

    /// <summary>
    /// Estimates problem difficulty using heuristics.
    /// </summary>
    /// <param name="problem">The problem text.</param>
    /// <returns>Estimated difficulty (0.0 = trivial, 1.0 = very hard).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This uses simple rules to guess how hard a problem is.
    /// In production, you might use a trained ML model for better estimates.
    ///
    /// **Heuristics used:**
    /// - Length: Longer problems tend to be harder
    /// - Complexity words: "prove", "optimize", "design" indicate harder problems
    /// - Multi-step indicators: Multiple questions or requirements
    /// - Domain keywords: Math symbols, code keywords, etc.
    /// </para>
    /// </remarks>
    public double EstimateDifficulty(string problem)
    {
        if (string.IsNullOrWhiteSpace(problem))
            return 0.5; // Default to medium

        double difficulty = 0.0;
        string lower = problem.ToLowerInvariant();

        // Factor 1: Length (longer = potentially harder)
        int wordCount = problem.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length;
        difficulty += Math.Min(0.3, wordCount / 200.0); // Up to 0.3 based on length

        // Factor 2: Complexity keywords
        var hardKeywords = new[] {
            "prove", "proof", "theorem", "optimize", "design", "analyze",
            "compare", "contrast", "evaluate", "synthesize", "algorithm",
            "complexity", "recursive", "dynamic programming"
        };

        int hardKeywordCount = hardKeywords.Count(kw => lower.Contains(kw));
        difficulty += Math.Min(0.3, hardKeywordCount * 0.1); // Up to 0.3

        // Factor 3: Multi-step indicators
        int questionMarks = problem.Count(c => c == '?');
        int steps = Math.Max(
            problem.Split(new[] { "step" }, StringSplitOptions.RemoveEmptyEntries).Length - 1,
            problem.Split(new[] { "first" }, StringSplitOptions.RemoveEmptyEntries).Length - 1
        );

        difficulty += Math.Min(0.2, (questionMarks + steps) * 0.05); // Up to 0.2

        // Factor 4: Mathematical/technical complexity
        bool hasMath = System.Text.RegularExpressions.Regex.IsMatch(problem, @"[+\-*/=<>∫∑∏√]", System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(1));
        bool hasCode = lower.Contains("function") || lower.Contains("algorithm") || lower.Contains("implement");
        bool hasLogic = lower.Contains("if") && lower.Contains("then");

        if (hasMath || hasCode || hasLogic)
        {
            difficulty += 0.2; // Technical problems are generally harder
        }

        return MathHelper.Clamp(difficulty, 0.0, 1.0);
    }

    /// <summary>
    /// Gets recommended strategy type based on problem difficulty.
    /// </summary>
    /// <param name="difficulty">Problem difficulty (0.0-1.0).</param>
    /// <returns>Recommended strategy name.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different difficulty levels benefit from different reasoning approaches:
    /// - Easy: Simple Chain-of-Thought is sufficient
    /// - Medium: CoT with verification
    /// - Hard: Self-Consistency (multiple attempts) or Tree-of-Thoughts (exploration)
    /// </para>
    /// </remarks>
    public string GetRecommendedStrategy(double difficulty)
    {
        difficulty = MathHelper.Clamp(difficulty, 0.0, 1.0);

        if (difficulty < 0.3)
            return "Chain-of-Thought";
        else if (difficulty < 0.6)
            return "Chain-of-Thought";  // Use same strategy for medium-low difficulty
        else if (difficulty < 0.8)
            return "Self-Consistency";
        else
            return "Tree-of-Thoughts";
    }
}
