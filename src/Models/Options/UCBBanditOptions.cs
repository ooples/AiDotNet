using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Validation;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the UCB bandit agent.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Controls number of arms and exploration strength in UCB.</para>
/// </remarks>
public class UCBBanditOptions<T> : ReinforcementLearningOptions<T>
{
    public UCBBanditOptions()
    {
    }

    public UCBBanditOptions(UCBBanditOptions<T> other)
    {
        Guard.NotNull(other);

        NumArms = other.NumArms;
        ExplorationParameter = other.ExplorationParameter;
    }

    /// <summary>Number of bandit arms.</summary>
    /// <value>Default: 10.</value>
    public int NumArms { get; init; } = 10;

    /// <summary>Exploration coefficient (UCB c parameter).</summary>
    /// <value>Default: 2.0.</value>
    public double ExplorationParameter { get; init; } = 2.0;
}
