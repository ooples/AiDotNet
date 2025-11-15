using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

public class NStepQLearningOptions<T> : ReinforcementLearningOptions<T>
{
    private int _stateSize;
    private int _actionSize;
    private int _nSteps = 3;

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
