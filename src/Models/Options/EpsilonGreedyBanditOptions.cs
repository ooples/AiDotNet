using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class EpsilonGreedyBanditOptions<T> : ReinforcementLearningOptions<T>
{
    // A k-armed bandit needs arms: with the previous implicit default of 0 the value-estimate
    // vector was empty, so SelectAction indexed an empty result vector and threw. Default to a
    // usable 10-arm bandit (overridable), matching UCBBanditOptions and GradientBanditOptions.
    private int _numArms = 10;

    public int NumArms
    {
        get => _numArms;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("NumArms must be greater than 0", nameof(NumArms));
            }
            _numArms = value;
        }
    }

    public double Epsilon { get; init; } = 0.1;
}
