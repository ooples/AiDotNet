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
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; set; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; set; }
    /// <summary>Learning rate of the policy network.</summary>
    public T PolicyLearningRate { get; set; }
    /// <summary>Learning rate of the Q-network.</summary>
    public T QLearningRate { get; set; }
    /// <summary>Learning rate of the value network.</summary>
    public T ValueLearningRate { get; set; }
    /// <summary>Discount factor (gamma) applied to future rewards.</summary>
    public T DiscountFactor { get; set; }
    /// <summary>Soft-update coefficient for the target networks: each update moves them this fraction of the way toward the online networks.</summary>
    public T TargetUpdateTau { get; set; }

    // IQL-specific parameters
    /// <summary>Expectile (tau) of the value regression; above 0.5 it approximates the maximum over in-dataset actions (0.7-0.9 is typical).</summary>
    public double Expectile { get; set; } = 0.7;  // Expectile parameter (tau), typically 0.7-0.9
    /// <summary>Inverse temperature of advantage-weighted regression: higher values follow high-advantage actions more closely.</summary>
    public T Temperature { get; set; }  // Temperature for advantage-weighted regression

    // Standard parameters
    /// <summary>Number of samples in each training batch.</summary>
    public int BatchSize { get; set; } = 256;
    /// <summary>Maximum number of transitions kept in the offline replay buffer.</summary>
    public int BufferSize { get; set; } = 1000000;
    /// <summary>Loss function used to train the Q-network.</summary>
    public ILossFunction<T> QLossFunction { get; set; } = new MeanSquaredErrorLoss<T>();
    /// <summary>Hidden layer sizes of the policy network.</summary>
    public List<int> PolicyHiddenLayers { get; set; } = new List<int> { 256, 256 };
    /// <summary>Hidden layer sizes of the Q-network.</summary>
    public List<int> QHiddenLayers { get; set; } = new List<int> { 256, 256 };
    /// <summary>Hidden layer sizes of the value network.</summary>
    public List<int> ValueHiddenLayers { get; set; } = new List<int> { 256, 256 };

    /// <summary>Creates options with the default values.</summary>
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
