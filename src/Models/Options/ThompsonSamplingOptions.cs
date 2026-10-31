using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>Configuration options for the Thompson-sampling bandit, which picks the arm whose sampled posterior reward is highest.</summary>
public class ThompsonSamplingOptions<T> : ReinforcementLearningOptions<T>
{
    // A k-armed bandit needs arms: with the previous implicit default of 0 the success/failure
    // count vectors were empty, so SelectAction indexed an empty result vector and threw. Default
    // to a usable 10-arm bandit (overridable), matching UCBBanditOptions and GradientBanditOptions.
    /// <summary>Number of arms (actions) the bandit chooses among.</summary>
    public int NumArms { get; init; } = 10;
}
