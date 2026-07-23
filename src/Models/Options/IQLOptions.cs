using AiDotNet.LossFunctions;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Implicit Q-Learning (IQL) agent.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IQL is an offline RL algorithm that avoids explicit policy constraints or
/// conservative regularization. Instead, it uses expectile regression to extract
/// a policy from the value function.
/// </para>
/// <para><b>For Beginners:</b>
/// IQL is designed for offline learning (learning from fixed datasets).
/// Unlike CQL which adds penalties, IQL uses a clever trick called "expectile regression"
/// to avoid overestimation.
///
/// Key innovation:
/// - **Expectile Regression**: Focus on upper quantiles of value distribution
/// - **Implicit Policy Extraction**: No explicit max over actions
/// - **Simpler than CQL**: Fewer hyperparameters to tune
///
/// Think of it like learning the "typical good outcome" rather than the "best possible outcome"
/// which helps avoid being too optimistic about unseen situations.
///
/// Advantages: Simpler, more stable than CQL in many cases
/// </para>
/// </remarks>
public class IQLOptions<T> : ModelOptions
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
    public T PolicyLearningRate { get; set; }
    public T QLearningRate { get; set; }
    public T ValueLearningRate { get; set; }
    public T DiscountFactor { get; set; }
    public T TargetUpdateTau { get; set; }

    // IQL-specific parameters
    public double Expectile { get; set; } = 0.7;  // Expectile parameter (tau), typically 0.7-0.9
    public T Temperature { get; set; }  // Temperature for advantage-weighted regression

    // Standard parameters
    public int BatchSize { get; set; } = 256;
    public int BufferSize { get; set; } = 1000000;
    public ILossFunction<T> QLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 256, 256 };
    public List<int> QHiddenLayers { get; set; } = new List<int> { 256, 256 };
    public List<int> ValueHiddenLayers { get; set; } = new List<int> { 256, 256 };

    public IQLOptions()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        PolicyLearningRate = numOps.FromDouble(0.0003);
        QLearningRate = numOps.FromDouble(0.0003);
        ValueLearningRate = numOps.FromDouble(0.0003);
        DiscountFactor = numOps.FromDouble(0.99);
        TargetUpdateTau = numOps.FromDouble(0.005);
        Temperature = numOps.FromDouble(3.0);
    }

    /// <summary>
    /// Validates that required properties are set.
    /// </summary>
    public void Validate()
    {
        if (StateSize <= 0)
            throw new ArgumentException("StateSize must be greater than 0", nameof(StateSize));
        if (ActionSize <= 0)
            throw new ArgumentException("ActionSize must be greater than 0", nameof(ActionSize));
        if (BatchSize <= 0)
            throw new ArgumentException("BatchSize must be greater than 0", nameof(BatchSize));
        if (BufferSize <= 0)
            throw new ArgumentException("BufferSize must be greater than 0", nameof(BufferSize));
    }
}
