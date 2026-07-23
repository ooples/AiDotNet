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
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T PolicyLearningRate { get; set; }
    public T QLearningRate { get; set; }
    public T AlphaLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T TargetUpdateTau { get; set; }
    public T InitialTemperature { get; set; }
    public bool AutoTuneTemperature { get; set; } = true;
    public T? TargetEntropy { get; set; }

    // CQL-specific parameters
    public T CQLAlpha { get; set; }  // Weight of conservative penalty
    public int CQLNumActions { get; set; } = 10;  // Number of actions to sample for CQL penalty
    public bool CQLLagrange { get; set; } = false;  // Use Lagrangian form
    public T CQLTargetActionGap { get; set; }  // Target Q-gap for Lagrangian

    // Standard parameters
    public int BatchSize { get; set; } = 256;
    public int BufferSize { get; set; } = 1000000;
    public int GradientSteps { get; set; } = 1;
    public ILossFunction<T> QLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 256, 256 };
    public List<int> QHiddenLayers { get; set; } = new List<int> { 256, 256 };

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
