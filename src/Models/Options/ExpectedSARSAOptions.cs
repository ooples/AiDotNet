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
    /// Initializes a new instance of the ExpectedSARSAOptions class with validated state and action sizes.
    /// </summary>
    /// <param name="stateSize">The size of the state space.</param>
    /// <param name="actionSize">The size of the action space.</param>
    /// <exception cref="System.ArgumentOutOfRangeException">
    /// Thrown when stateSize or actionSize is less than or equal to zero.
    /// </exception>
    public ExpectedSARSAOptions(int stateSize, int actionSize)
    {
        if (stateSize <= 0)
        {
            throw new System.ArgumentOutOfRangeException(
                nameof(stateSize),
                stateSize,
                "StateSize must be greater than zero.");
        }

        if (actionSize <= 0)
        {
            throw new System.ArgumentOutOfRangeException(
                nameof(actionSize),
                actionSize,
                "ActionSize must be greater than zero.");
        }

        _stateSize = stateSize;
        _actionSize = actionSize;
    }

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
