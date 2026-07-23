using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class GradientBanditOptions<T> : ReinforcementLearningOptions<T>
{
    // A multi-armed bandit needs arms: with the previous implicit default of 0 the preference vector was
    // empty, so the agent exposed no parameters and training could not change anything. Default to a usable
    // 10-arm bandit (overridable); callers with a known action space still set this explicitly.
    public int NumArms { get; init; } = 10;
    public double Alpha { get; init; } = 0.1;  // Step size
    public bool UseBaseline { get; init; } = true;
}
