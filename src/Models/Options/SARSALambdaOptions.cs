using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>Configuration options for SARSA(lambda), on-policy TD control with eligibility traces.</summary>
public class SARSALambdaOptions<T> : ReinforcementLearningOptions<T>
{
    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize { get; init; }
    /// <summary>Number of discrete actions, or the dimension of the action vector for a continuous action space.</summary>
    public int ActionSize { get; init; }
    /// <summary>Trace-decay parameter (lambda) of the eligibility traces.</summary>
    public double Lambda { get; init; } = 0.9;  // Eligibility trace decay
}
