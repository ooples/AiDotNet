namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Base class for stochastic actor-critic reinforcement learning agents (SAC, etc.).
/// </summary>
/// <typeparam name="TState">The type used to represent the environment state, typically Tensor&lt;T&gt;.</typeparam>
/// <typeparam name="TAction">The type used to represent actions, typically Vector&lt;T&gt; for continuous action spaces.</typeparam>
/// <typeparam name="T">The numeric type used for calculations (float, double, etc.).</typeparam>
/// <remarks>
/// <para>
/// Stochastic actor-critic agents use a stochastic policy (rather than deterministic) and
/// often incorporate entropy regularization to encourage exploration. This base class 
/// extends DualCriticAgentBase with functionality specific to stochastic policies.
/// </para>
/// </remarks>
public abstract class StochasticActorCriticAgentBase<TState, TAction, T> 
    : DualCriticAgentBase<TState, TAction, T, IStochasticPolicy<TState, TAction, T>>
    where TState : Tensor<T>
{
    /// <summary>
    /// Gets or sets the entropy coefficient that controls exploration.
    /// </summary>
    protected T EntropyCoefficient { get; set; }

    /// <summary>
    /// Gets a value indicating whether entropy coefficient is automatically tuned.
    /// </summary>
    protected bool AutoTuneEntropyCoefficient { get; }

    /// <summary>
    /// Gets the target entropy for auto-tuning.
    /// </summary>
    protected T TargetEntropy { get; private set; }
    
    /// <summary>
    /// Gets a value indicating whether a target entropy has been set.
    /// </summary>
    protected bool HasTargetEntropy { get; private set; }

    /// <summary>
    /// Gets the learning rate for entropy coefficient optimization.
    /// </summary>
    protected T EntropyLearningRate { get; }

    /// <summary>
    /// Gets a value indicating whether log probabilities should be clipped.
    /// </summary>
    protected bool ClippedLogProbs { get; }

    /// <summary>
    /// Gets the minimum log probability for clipping.
    /// </summary>
    protected T MinLogProb { get; }

    /// <summary>
    /// Gets the maximum log probability for clipping.
    /// </summary>
    protected T MaxLogProb { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="StochasticActorCriticAgentBase{TState, TAction, T}"/> class.
    /// </summary>
    /// <param name="actor">The actor network.</param>
    /// <param name="critic1">The first critic network.</param>
    /// <param name="critic1Target">The target first critic network.</param>
    /// <param name="critic2">The second critic network.</param>
    /// <param name="critic2Target">The target second critic network.</param>
    /// <param name="replayBuffer">The replay buffer for storing experiences.</param>
    /// <param name="gamma">The discount factor for future rewards.</param>
    /// <param name="tau">The soft update factor for target networks.</param>
    /// <param name="batchSize">The batch size for training.</param>
    /// <param name="warmUpSteps">The number of warm-up steps before learning begins.</param>
    /// <param name="useGradientClipping">Whether to use gradient clipping.</param>
    /// <param name="maxGradientNorm">The maximum gradient norm for clipping.</param>
    /// <param name="policyUpdateFrequency">The frequency of policy updates.</param>
    /// <param name="useMinimumQValue">Whether to use the minimum Q-value from the two critics.</param>
    /// <param name="entropyCoefficient">The initial entropy coefficient.</param>
    /// <param name="autoTuneEntropyCoefficient">Whether to automatically tune the entropy coefficient.</param>
    /// <param name="targetEntropy">The target entropy for auto-tuning.</param>
    /// <param name="entropyLearningRate">The learning rate for entropy coefficient optimization.</param>
    /// <param name="clippedLogProbs">Whether to clip log probabilities.</param>
    /// <param name="minLogProb">The minimum log probability for clipping.</param>
    /// <param name="maxLogProb">The maximum log probability for clipping.</param>
    /// <param name="seed">Optional seed for the random number generator.</param>
    protected StochasticActorCriticAgentBase(
        IStochasticPolicy<TState, TAction, T> actor,
        IActionValueFunction<TState, TAction, T> critic1,
        IActionValueFunction<TState, TAction, T> critic1Target,
        IActionValueFunction<TState, TAction, T> critic2,
        IActionValueFunction<TState, TAction, T> critic2Target,
        IReplayBuffer<TState, TAction, T> replayBuffer,
        double gamma,
        double tau,
        int batchSize,
        int warmUpSteps,
        bool useGradientClipping,
        double maxGradientNorm,
        int policyUpdateFrequency,
        bool useMinimumQValue,
        double entropyCoefficient,
        bool autoTuneEntropyCoefficient,
        double? targetEntropy,
        double entropyLearningRate,
        bool clippedLogProbs,
        double minLogProb,
        double maxLogProb,
        int? seed = null)
        : base(actor, actor, critic1, critic1Target, critic2, critic2Target, replayBuffer, default!, gamma, tau, batchSize, warmUpSteps, useGradientClipping, maxGradientNorm, policyUpdateFrequency, useMinimumQValue, seed)
    {
        EntropyCoefficient = NumOps.FromDouble(entropyCoefficient);
        AutoTuneEntropyCoefficient = autoTuneEntropyCoefficient;
        if (targetEntropy.HasValue)
        {
            TargetEntropy = NumOps.FromDouble(targetEntropy.Value);
            HasTargetEntropy = true;
        }
        else
        {
            TargetEntropy = NumOps.Zero;
            HasTargetEntropy = false;
        }
        EntropyLearningRate = NumOps.FromDouble(entropyLearningRate);
        ClippedLogProbs = clippedLogProbs;
        MinLogProb = NumOps.FromDouble(minLogProb);
        MaxLogProb = NumOps.FromDouble(maxLogProb);
    }

    /// <summary>
    /// Clips the log probability value if clipping is enabled.
    /// </summary>
    /// <param name="logProb">The log probability value to clip.</param>
    /// <returns>The clipped log probability value.</returns>
    protected T ClipLogProb(T logProb)
    {
        if (ClippedLogProbs)
        {
            return MathHelper.Clamp(logProb, MinLogProb, MaxLogProb);
        }
        
        return logProb;
    }

    /// <summary>
    /// Updates the entropy coefficient based on the current policy's entropy.
    /// </summary>
    /// <param name="policyEntropy">The current entropy of the policy.</param>
    protected virtual void UpdateEntropyCoefficient(T policyEntropy)
    {
        if (!AutoTuneEntropyCoefficient || !HasTargetEntropy)
        {
            return;
        }

        // Compute entropy loss: -alpha * (log_prob + target_entropy)
        T entropyDifference = NumOps.Add(policyEntropy, TargetEntropy);
        T entropyLoss = NumOps.Multiply(NumOps.Negate(EntropyCoefficient), entropyDifference);
        
        // Update entropy coefficient using gradient ascent
        T gradient = NumOps.Negate(entropyDifference);
        EntropyCoefficient = NumOps.Add(
            EntropyCoefficient, 
            NumOps.Multiply(EntropyLearningRate, gradient));
        
        // Ensure entropy coefficient is positive
        EntropyCoefficient = MathHelper.Max(EntropyCoefficient, NumOps.FromDouble(1e-6));
    }
}