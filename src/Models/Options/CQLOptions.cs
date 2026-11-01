using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Conservative Q-Learning (CQL) agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CQL is an offline RL algorithm that learns from fixed datasets without environment interaction.
/// It addresses overestimation by adding a conservative penalty to Q-values.
/// </para>
/// <para><b>For Beginners:</b>
/// CQL is designed for learning from logged data without trying new actions.
/// This is useful when you have historical data but can't experiment in the real environment
/// (e.g., medical treatment, autonomous driving).
///
/// Key innovation:
/// - **Conservative Q-Learning**: Penalizes Q-values for unseen actions to prevent overoptimistic estimates
/// - **Offline Learning**: No environment interaction during training
///
/// Think of it like learning to drive from dashcam footage - you can't try new maneuvers,
/// so you need to be conservative about what you haven't seen.
///
/// Based on SAC architecture with conservative regularization.
/// </para>
/// </remarks>
public class CQLOptions<T> : ModelOptions
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; set; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; set; }
    /// <summary>Learning rate of the policy network.</summary>
    public T PolicyLearningRate { get; set; }
    /// <summary>Learning rate of the Q-network.</summary>
    public T QLearningRate { get; set; }
    /// <summary>Learning rate used when tuning the entropy temperature.</summary>
    public T AlphaLearningRate { get; set; }
    /// <summary>Discount factor (gamma) applied to future rewards.</summary>
    public T DiscountFactor { get; set; }
    /// <summary>Soft-update coefficient for the target networks: each update moves them this fraction of the way toward the online networks.</summary>
    public T TargetUpdateTau { get; set; }
    /// <summary>Initial entropy temperature (alpha) of the soft actor-critic objective.</summary>
    public T InitialTemperature { get; set; }
    /// <summary>Whether to tune the entropy temperature automatically toward <see cref="TargetEntropy"/>.</summary>
    public bool AutoTuneTemperature { get; set; } = true;
    /// <summary>Target policy entropy for automatic temperature tuning; when null, the negative action dimension is used, as in SAC.</summary>
    public T? TargetEntropy { get; set; }

    // CQL-specific parameters
    /// <summary>Weight of the conservative penalty that pushes down Q-values of actions outside the dataset.</summary>
    public T CQLAlpha { get; set; }  // Weight of conservative penalty
    /// <summary>Number of actions sampled per state to estimate the conservative penalty's log-sum-exp over Q-values.</summary>
    public int CQLNumActions { get; set; } = 10;  // Number of actions to sample for CQL penalty
    /// <summary>Whether to tune the conservative-penalty weight automatically (the Lagrangian form of CQL) instead of keeping <see cref="CQLAlpha"/> fixed.</summary>
    public bool CQLLagrange { get; set; } = false;  // Use Lagrangian form
    /// <summary>Target gap between out-of-distribution and dataset Q-values; with <see cref="CQLLagrange"/> the conservative weight is tuned to hold this gap.</summary>
    public T CQLTargetActionGap { get; set; }  // Target Q-gap for Lagrangian

    // Standard parameters
    /// <summary>Number of samples in each training batch.</summary>
    public int BatchSize { get; set; } = 256;
    /// <summary>Maximum number of transitions kept in the offline replay buffer.</summary>
    public int BufferSize { get; set; } = 1000000;
    /// <summary>Number of gradient updates performed per training call.</summary>
    public int GradientSteps { get; set; } = 1;
    /// <summary>Loss function used to train the Q-network.</summary>
    public ILossFunction<T> QLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Hidden layer sizes of the policy network.</summary>
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 256, 256 };
    /// <summary>Hidden layer sizes of the Q-network.</summary>
    public List<int> QHiddenLayers { get; set; } = new List<int> { 256, 256 };

    /// <summary>Creates options with the default values.</summary>
    public CQLOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0003);
        QLearningRate = numOps.FromDouble(0.0003);
        AlphaLearningRate = numOps.FromDouble(0.0003);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.005);
        InitialTemperature = numOps.FromDouble(0.2);
        CQLAlpha = numOps.FromDouble(1.0);
        CQLTargetActionGap = numOps.FromDouble(0.0);
    }
}
