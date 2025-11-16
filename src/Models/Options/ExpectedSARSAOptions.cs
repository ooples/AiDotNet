using AiDotNet.ReinforcementLearning.Agents;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Expected SARSA agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExpectedSARSAOptions<T> : ReinforcementLearningOptions<T>
{
    private int _stateSize;
    private int _actionSize;

    /// <summary>
    /// Gets or initializes the size of the state space.
    /// </summary>
    /// <exception cref="System.ArgumentOutOfRangeException">
    /// Thrown when the value is less than or equal to zero.
    /// </exception>
    public int StateSize
    {
        get => _stateSize;
        init
        {
            if (value <= 0)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(StateSize),
                    value,
                    "StateSize must be greater than zero.");
            }
            _stateSize = value;
        }
    }

    /// <summary>
    /// Gets or initializes the size of the action space.
    /// </summary>
    /// <exception cref="System.ArgumentOutOfRangeException">
    /// Thrown when the value is less than or equal to zero.
    /// </exception>
    public int ActionSize
    {
        get => _actionSize;
        init
        {
            if (value <= 0)
            {
                throw new System.ArgumentOutOfRangeException(
                    nameof(ActionSize),
                    value,
                    "ActionSize must be greater than zero.");
            }
            _actionSize = value;
        }
    }
}
