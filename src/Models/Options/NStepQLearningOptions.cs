using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>Configuration options for n-step Q-learning.</summary>
public class NStepQLearningOptions<T> : ReinforcementLearningOptions<T>
{
    private int _stateSize;
    private int _actionSize;
    private int _nSteps = 3;

    /// <summary>Number of features in one observation of the environment state.</summary>
    public int StateSize
    {
        get => _stateSize;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("StateSize must be greater than 0", nameof(StateSize));
            }
            _stateSize = value;
        }
    }

    /// <summary>Number of discrete actions.</summary>
    public int ActionSize
    {
        get => _actionSize;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("ActionSize must be greater than 0", nameof(ActionSize));
            }
            _actionSize = value;
        }
    }

    /// <summary>Number of rewards (n) accumulated before bootstrapping from the Q-value estimate.</summary>
    public int NSteps
    {
        get => _nSteps;
        init
        {
            if (value <= 0)
            {
                throw new ArgumentException("NSteps must be greater than 0", nameof(NSteps));
            }
            _nSteps = value;
        }
    }
}
