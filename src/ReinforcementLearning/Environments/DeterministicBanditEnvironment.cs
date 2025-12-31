using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ReinforcementLearning.Environments;

/// <summary>
/// A deterministic multi-armed bandit environment for testing purposes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double).</typeparam>
/// <remarks>
/// <para>
/// This environment provides a simple bandit setting where each action has a fixed deterministic reward.
/// It's useful for testing RL data loaders and agents with predictable outcomes.
/// </para>
/// <para><b>For Beginners:</b>
/// A bandit is like a slot machine. Each "arm" (action) has a fixed reward.
/// This deterministic version always gives the same reward for the same action,
/// making it perfect for testing - you always know what reward to expect.
/// </para>
/// </remarks>
public class DeterministicBanditEnvironment<T> : IEnvironment<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _actionSpaceSize;
    private readonly int _observationSpaceDimension;
    private readonly int _maxSteps;
    private readonly T[] _armRewards;
    private Random _random;
    private int _currentStep;
    private Vector<T> _currentState;

    /// <inheritdoc/>
    public int ObservationSpaceDimension => _observationSpaceDimension;

    /// <inheritdoc/>
    public int ActionSpaceSize => _actionSpaceSize;

    /// <inheritdoc/>
    public bool IsContinuousActionSpace => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="DeterministicBanditEnvironment{T}"/> class.
    /// </summary>
    /// <param name="actionSpaceSize">The number of arms/actions (default: 10).</param>
    /// <param name="observationSpaceDimension">The observation dimension (default: 1).</param>
    /// <param name="maxSteps">Maximum steps per episode (default: 100).</param>
    /// <param name="seed">Random seed for reproducibility (default: 42).</param>
    public DeterministicBanditEnvironment(
        int actionSpaceSize = 10,
        int observationSpaceDimension = 1,
        int maxSteps = 100,
        int seed = 42)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _actionSpaceSize = actionSpaceSize;
        _observationSpaceDimension = observationSpaceDimension;
        _maxSteps = maxSteps;
        _random = RandomHelper.CreateSeededRandom(seed);
        _currentStep = 0;

        // Initialize deterministic rewards for each arm
        _armRewards = new T[_actionSpaceSize];
        for (int i = 0; i < _actionSpaceSize; i++)
        {
            // Arm i gives reward i / (actionSpaceSize - 1), so rewards range from 0 to 1
            double reward = _actionSpaceSize > 1 ? (double)i / (_actionSpaceSize - 1) : 1.0;
            _armRewards[i] = _numOps.FromDouble(reward);
        }

        // Initialize current state
        _currentState = new Vector<T>(_observationSpaceDimension);
    }

    /// <inheritdoc/>
    public Vector<T> Reset()
    {
        _currentStep = 0;
        // State is always zero for bandit
        _currentState = new Vector<T>(_observationSpaceDimension);
        return _currentState;
    }

    /// <inheritdoc/>
    public (Vector<T> NextState, T Reward, bool Done, Dictionary<string, object> Info) Step(Vector<T> action)
    {
        // Get action index from one-hot or direct index
        int actionIndex = GetActionIndex(action);

        // Clamp to valid range
        actionIndex = Math.Max(0, Math.Min(actionIndex, _actionSpaceSize - 1));

        // Get deterministic reward
        T reward = _armRewards[actionIndex];
        _currentStep++;

        bool done = _currentStep >= _maxSteps;

        var info = new Dictionary<string, object>
        {
            ["action"] = actionIndex,
            ["step"] = _currentStep
        };

        return (_currentState, reward, done, info);
    }

    /// <inheritdoc/>
    public void Seed(int seed)
    {
        _random = RandomHelper.CreateSeededRandom(seed);
    }

    /// <inheritdoc/>
    public void Close()
    {
        // Nothing to clean up
    }

    private int GetActionIndex(Vector<T> action)
    {
        if (action.Length == 1)
        {
            // Direct index
            return (int)_numOps.ToDouble(action[0]);
        }

        // One-hot encoded - find max
        int maxIndex = 0;
        T maxValue = action[0];
        for (int i = 1; i < action.Length; i++)
        {
            if (_numOps.Compare(action[i], maxValue) > 0)
            {
                maxValue = action[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
