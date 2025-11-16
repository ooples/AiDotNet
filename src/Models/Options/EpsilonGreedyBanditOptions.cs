using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class EpsilonGreedyBanditOptions<T> : ReinforcementLearningOptions<T>
{
    private int _numArms;

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
