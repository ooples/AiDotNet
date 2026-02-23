using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning.Verification;

/// <summary>
/// Hybrid Reward Model that combines Process Reward Model (PRM) and Outcome Reward Model (ORM).
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This reward model gets the best of both worlds by combining:
/// - **PRM**: Rewards good reasoning steps (the journey)
/// - **ORM**: Rewards correct final answers (the destination)
///
/// **Why combine them?**
/// Imagine you're solving a math problem:
/// - PRM alone: You might get partial credit for good steps even with wrong answer
/// - ORM alone: You get zero credit for great reasoning if final answer is slightly off
/// - **Hybrid**: Rewards both good reasoning AND correct results
///
/// **Weighting strategy:**
/// ```
/// Total Reward = (PRM_score × process_weight) + (ORM_score × outcome_weight)
/// ```
///
/// **Common weight configurations:**
/// 1. **Balanced (50/50)**: `processWeight: 0.5, outcomeWeight: 0.5`
///    - Equal importance to process and outcome
///    - Good default for most tasks
///
/// 2. **Process-focused (70/30)**: `processWeight: 0.7, outcomeWeight: 0.3`
///    - Emphasizes learning correct reasoning
///    - Good for training and education
///    - Used in "Let's Verify Step by Step" (Lightman et al., 2023)
///
/// 3. **Outcome-focused (30/70)**: `processWeight: 0.3, outcomeWeight: 0.7`
///    - Emphasizes getting correct answers
///    - Good for competition/evaluation
///    - Used in math competitions
///
/// 4. **Outcome-dominant (10/90)**: `processWeight: 0.1, outcomeWeight: 0.9`
///    - Almost all weight on final answer
///    - Good for benchmarking
///    - Similar to GSM8K evaluation
///
/// **Example scenarios:**
///
/// *Scenario 1: Perfect reasoning, correct answer*
/// - PRM: 0.95 (excellent steps)
/// - ORM: 1.0 (correct)
/// - Hybrid (50/50): 0.975 ← Best possible
///
/// *Scenario 2: Good reasoning, wrong answer*
/// - PRM: 0.85 (good steps)
/// - ORM: 0.0 (incorrect)
/// - Hybrid (50/50): 0.425 ← Partial credit
///
/// *Scenario 3: Poor reasoning, lucky correct answer*
/// - PRM: 0.3 (weak steps)
/// - ORM: 1.0 (correct)
/// - Hybrid (50/50): 0.65 ← Penalized for poor reasoning
///
/// **Code example:**
/// ```csharp
/// // Create hybrid model (50/50 balance)
/// var hybridModel = new HybridRewardModel<double>(
///     prm: new ProcessRewardModel<double>(chatModel),
///     orm: new OutcomeRewardModel<double>(chatModel),
///     processWeight: 0.5,
///     outcomeWeight: 0.5
/// );
///
/// var chain = /* your reasoning chain */;
/// double reward = await hybridModel.CalculateRewardAsync(chain, correctAnswer: "42");
///
/// Console.WriteLine($"Process score: {await prm.CalculateRewardAsync(chain)}");
/// Console.WriteLine($"Outcome score: {await orm.CalculateRewardAsync(chain, "42")}");
/// Console.WriteLine($"Hybrid score: {reward}");
/// ```
///
/// **Research:**
/// - "Let's Verify Step by Step" (Lightman et al., 2023)
///   → Found PRM > ORM for training, but hybrid works best for evaluation
/// - "Math-Shepherd" (Wang et al., 2024)
///   → Used 0.7 PRM / 0.3 ORM for training math reasoners
/// - "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
///   → Original ORM paper, suggested combining with process supervision
///
/// **Adaptive weighting:**
/// You can also adjust weights based on task difficulty:
/// - Easy problems: Higher ORM weight (outcome matters more)
/// - Hard problems: Higher PRM weight (process matters more for learning)
/// </para>
/// </remarks>
internal class HybridRewardModel<T> : IRewardModel<T>
{
    private readonly IRewardModel<T> _prm;
    private readonly IRewardModel<T> _orm;
    private readonly double _processWeight;
    private readonly double _outcomeWeight;
    private readonly INumericOperations<T> _numOps;
    private readonly bool _normalizeWeights;

    /// <summary>
    /// Initializes a new instance of the <see cref="HybridRewardModel{T}"/> class.
    /// </summary>
    /// <param name="prm">Process Reward Model.</param>
    /// <param name="orm">Outcome Reward Model.</param>
    /// <param name="processWeight">Weight for process reward (default: 0.5).</param>
    /// <param name="outcomeWeight">Weight for outcome reward (default: 0.5).</param>
    /// <param name="normalizeWeights">Auto-normalize weights to sum to 1.0 (default: true).</param>
    public HybridRewardModel(
        IRewardModel<T> prm,
        IRewardModel<T> orm,
        double processWeight = 0.5,
        double outcomeWeight = 0.5,
        bool normalizeWeights = true)
    {
        Guard.NotNull(prm);
        _prm = prm;
        Guard.NotNull(orm);
        _orm = orm;
        _numOps = MathHelper.GetNumericOperations<T>();
        _normalizeWeights = normalizeWeights;

        // Normalize weights if requested
        if (_normalizeWeights)
        {
            double sum = processWeight + outcomeWeight;
            if (sum > 0)
            {
                _processWeight = processWeight / sum;
                _outcomeWeight = outcomeWeight / sum;
            }
            else
            {
                _processWeight = 0.5;
                _outcomeWeight = 0.5;
            }
        }
        else
        {
            _processWeight = processWeight;
            _outcomeWeight = outcomeWeight;
        }
    }

    /// <summary>
    /// Creates a balanced hybrid model (50/50 PRM/ORM).
    /// </summary>
    public static HybridRewardModel<T> CreateBalanced(
        IRewardModel<T> prm,
        IRewardModel<T> orm)
    {
        return new HybridRewardModel<T>(prm, orm, 0.5, 0.5);
    }

    /// <summary>
    /// Creates a process-focused hybrid model (70/30 PRM/ORM).
    /// </summary>
    /// <remarks>
    /// Good for training and education where learning the reasoning process is important.
    /// </remarks>
    public static HybridRewardModel<T> CreateProcessFocused(
        IRewardModel<T> prm,
        IRewardModel<T> orm)
    {
        return new HybridRewardModel<T>(prm, orm, 0.7, 0.3);
    }

    /// <summary>
    /// Creates an outcome-focused hybrid model (30/70 PRM/ORM).
    /// </summary>
    /// <remarks>
    /// Good for competitions and evaluation where final answer correctness is most important.
    /// </remarks>
    public static HybridRewardModel<T> CreateOutcomeFocused(
        IRewardModel<T> prm,
        IRewardModel<T> orm)
    {
        return new HybridRewardModel<T>(prm, orm, 0.3, 0.7);
    }

    /// <inheritdoc/>
    public string ModelName => "Hybrid Reward Model (PRM + ORM)";

    /// <inheritdoc/>
    public RewardModelType ModelType => RewardModelType.Hybrid;

    /// <inheritdoc/>
    public string Description =>
        $"Combines Process ({_processWeight:P0}) and Outcome ({_outcomeWeight:P0}) rewards. " +
        "Rewards both correct reasoning steps AND final answer accuracy.";

    /// <summary>
    /// Calculates hybrid reward combining PRM and ORM.
    /// </summary>
    public async Task<T> CalculateRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        if (chain == null)
            throw new ArgumentNullException(nameof(chain));

        // Calculate both rewards in parallel
        var prmTask = _prm.CalculateChainRewardAsync(chain, correctAnswer, cancellationToken);
        var ormTask = _orm.CalculateChainRewardAsync(chain, correctAnswer, cancellationToken);

        await Task.WhenAll(prmTask, ormTask);

        T processReward = prmTask.Result;
        T outcomeReward = ormTask.Result;

        // Combine with weights
        double prmScore = Convert.ToDouble(processReward);
        double ormScore = Convert.ToDouble(outcomeReward);

        double hybridScore = (prmScore * _processWeight) + (ormScore * _outcomeWeight);

        return _numOps.FromDouble(hybridScore);
    }

    /// <summary>
    /// Calculates step reward using PRM (ORM doesn't score individual steps).
    /// </summary>
    public async Task<T> CalculateStepRewardAsync(
        ReasoningStep<T> step,
        ReasoningContext context,
        CancellationToken cancellationToken = default)
    {
        // For individual steps, use PRM (ORM doesn't score steps)
        return await _prm.CalculateStepRewardAsync(step, context, cancellationToken);
    }

    /// <summary>
    /// Calculates chain reward (same as CalculateRewardAsync).
    /// </summary>
    public async Task<T> CalculateChainRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        return await CalculateRewardAsync(chain, correctAnswer, cancellationToken);
    }

    /// <summary>
    /// Gets detailed breakdown of PRM and ORM scores.
    /// </summary>
    public async Task<RewardBreakdown<T>> GetRewardBreakdownAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default)
    {
        // Calculate both rewards
        var prmTask = _prm.CalculateChainRewardAsync(chain, correctAnswer, cancellationToken);
        var ormTask = _orm.CalculateChainRewardAsync(chain, correctAnswer, cancellationToken);

        await Task.WhenAll(prmTask, ormTask);

        T processReward = prmTask.Result;
        T outcomeReward = ormTask.Result;

        // Calculate hybrid
        double prmScore = Convert.ToDouble(processReward);
        double ormScore = Convert.ToDouble(outcomeReward);
        double hybridScore = (prmScore * _processWeight) + (ormScore * _outcomeWeight);

        return new RewardBreakdown<T>
        {
            ProcessReward = processReward,
            OutcomeReward = outcomeReward,
            HybridReward = _numOps.FromDouble(hybridScore),
            ProcessWeight = _processWeight,
            OutcomeWeight = _outcomeWeight,
            ProcessContribution = prmScore * _processWeight,
            OutcomeContribution = ormScore * _outcomeWeight
        };
    }

    /// <summary>
    /// Creates a new hybrid model with adjusted weights.
    /// </summary>
    /// <remarks>
    /// Useful for adaptive weighting based on task difficulty or training phase.
    /// </remarks>
    public HybridRewardModel<T> WithWeights(double processWeight, double outcomeWeight)
    {
        return new HybridRewardModel<T>(
            _prm,
            _orm,
            processWeight,
            outcomeWeight,
            _normalizeWeights
        );
    }

    /// <summary>
    /// Creates adaptive weights based on problem difficulty.
    /// </summary>
    /// <param name="difficulty">Difficulty level (0.0-1.0).</param>
    /// <returns>Hybrid model with adjusted weights.</returns>
    /// <remarks>
    /// - Easy problems (difficulty &lt; 0.3): Focus more on outcome (60% ORM)
    /// - Medium problems (0.3-0.7): Balanced (50/50)
    /// - Hard problems (difficulty &gt; 0.7): Focus more on process (60% PRM)
    /// </remarks>
    public HybridRewardModel<T> WithAdaptiveWeights(double difficulty)
    {
        difficulty = MathHelper.Clamp(difficulty, 0.0, 1.0);

        double processWeight;
        double outcomeWeight;

        if (difficulty < 0.3)
        {
            // Easy: Emphasize outcome
            processWeight = 0.4;
            outcomeWeight = 0.6;
        }
        else if (difficulty > 0.7)
        {
            // Hard: Emphasize process
            processWeight = 0.6;
            outcomeWeight = 0.4;
        }
        else
        {
            // Medium: Balanced
            processWeight = 0.5;
            outcomeWeight = 0.5;
        }

        return new HybridRewardModel<T>(
            _prm,
            _orm,
            processWeight,
            outcomeWeight,
            _normalizeWeights
        );
    }
}

/// <summary>
/// Detailed breakdown of hybrid reward components.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring.</typeparam>
internal class RewardBreakdown<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RewardBreakdown{T}"/> class.
    /// </summary>
    public RewardBreakdown()
    {
        INumericOperations<T> numOps = MathHelper.GetNumericOperations<T>();
        ProcessReward = numOps.Zero;
        OutcomeReward = numOps.Zero;
        HybridReward = numOps.Zero;
    }

    /// <summary>
    /// Process reward from PRM.
    /// </summary>
    public T ProcessReward { get; set; }

    /// <summary>
    /// Outcome reward from ORM.
    /// </summary>
    public T OutcomeReward { get; set; }

    /// <summary>
    /// Combined hybrid reward.
    /// </summary>
    public T HybridReward { get; set; }

    /// <summary>
    /// Weight used for process reward.
    /// </summary>
    public double ProcessWeight { get; set; }

    /// <summary>
    /// Weight used for outcome reward.
    /// </summary>
    public double OutcomeWeight { get; set; }

    /// <summary>
    /// Contribution from process reward to hybrid score.
    /// </summary>
    public double ProcessContribution { get; set; }

    /// <summary>
    /// Contribution from outcome reward to hybrid score.
    /// </summary>
    public double OutcomeContribution { get; set; }

    /// <summary>
    /// Gets a summary of the reward breakdown.
    /// </summary>
    public string GetSummary()
    {
        return $@"Hybrid Reward Breakdown:
  Process Reward (PRM): {Convert.ToDouble(ProcessReward):F3} × {ProcessWeight:P0} = {ProcessContribution:F3}
  Outcome Reward (ORM): {Convert.ToDouble(OutcomeReward):F3} × {OutcomeWeight:P0} = {OutcomeContribution:F3}
  ────────────────────────────────────────
  Hybrid Reward:        {Convert.ToDouble(HybridReward):F3}";
    }

    public override string ToString() => GetSummary();
}
