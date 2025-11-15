using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Environments;

/// <summary>
/// Classic CartPole-v1 environment for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double).</typeparam>
/// <remarks>
/// <para>
/// The CartPole environment simulates balancing a pole on a cart. The agent must move the cart
/// left or right to keep the pole balanced. The episode ends if:
/// - The pole angle exceeds ±12 degrees
/// - The cart position exceeds ±2.4 units
/// - The maximum number of steps is reached
/// </para>
/// <para><b>For Beginners:</b>
/// Think of this like balancing a broomstick on your hand - you move your hand left and right
/// to keep the stick upright. The CartPole is a classic RL problem that's simple to understand
/// but requires learning to balance competing forces.
///
/// State (4 dimensions):
/// - Cart position: where the cart is (-2.4 to 2.4)
/// - Cart velocity: how fast it's moving
/// - Pole angle: how tilted the pole is (-12° to 12°)
/// - Pole angular velocity: how fast it's rotating
///
/// Actions (2 discrete):
/// - 0: Push cart left
/// - 1: Push cart right
///
/// Reward: +1 for each timestep the pole remains balanced
/// </para>
/// </remarks>
public class CartPoleEnvironment<T> : IEnvironment<T>
{
    private readonly INumericOperations<T> _numOps;
    private Random _random;
    private readonly int _maxSteps;

    // Physics constants
    private readonly double _gravity = 9.8;
    private readonly double _massCart = 1.0;
    private readonly double _massPole = 0.1;
    private readonly double _totalMass;
    private readonly double _length = 0.5;  // Half-pole length
    private readonly double _poleMassLength;
    private readonly double _forceMag = 10.0;
    private readonly double _tau = 0.02;  // Seconds between state updates

    // Thresholds for episode termination
    private readonly double _thetaThresholdRadians = 12 * Math.PI / 180;  // ±12 degrees
    private readonly double _xThreshold = 2.4;

    // Current state
    private double _x;           // Cart position
    private double _xDot;        // Cart velocity
    private double _theta;       // Pole angle
    private double _thetaDot;    // Pole angular velocity
    private int _steps;

    /// <inheritdoc/>
    public int ObservationSpaceDimension => 4;

    /// <inheritdoc/>
    public int ActionSpaceSize => 2;

    /// <inheritdoc/>
    public bool IsContinuousActionSpace => false;

    /// <summary>
    /// Initializes a new instance of the CartPoleEnvironment class.
    /// </summary>
    /// <param name="maxSteps">Maximum steps per episode (default 500).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public CartPoleEnvironment(int maxSteps = 500, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _maxSteps = maxSteps;
        _totalMass = _massCart + _massPole;
        _poleMassLength = _massPole * _length;

        Reset();
    }

    /// <inheritdoc/>
    public Vector<T> Reset()
    {
        // Initialize state with small random values
        _x = (_random.NextDouble() - 0.5) * 0.1;
        _xDot = (_random.NextDouble() - 0.5) * 0.1;
        _theta = (_random.NextDouble() - 0.5) * 0.1;
        _thetaDot = (_random.NextDouble() - 0.5) * 0.1;
        _steps = 0;

        return GetStateVector();
    }

    /// <inheritdoc/>
    public (Vector<T> NextState, T Reward, bool Done, Dictionary<string, object> Info) Step(Vector<T> action)
    {
        // Parse action (one-hot or single index)
        int actionIndex;
        if (action.Length == 1)
        {
            // Single element containing action index
            actionIndex = (int)Convert.ToDouble(_numOps.ToDouble(action[0]));
        }
        else
        {
            // One-hot encoded
            actionIndex = 0;
            for (int i = 1; i < action.Length; i++)
            {
                if (_numOps.ToDouble(action[i]) > _numOps.ToDouble(action[actionIndex]))
                {
                    actionIndex = i;
                }
            }
        }

        if (actionIndex < 0 || actionIndex >= ActionSpaceSize)
            throw new ArgumentException($"Invalid action: {actionIndex}. Must be 0 or 1.", nameof(action));

        // Apply force
        double force = actionIndex == 1 ? _forceMag : -_forceMag;

        // Physics simulation (using Euler's method)
        double cosTheta = Math.Cos(_theta);
        double sinTheta = Math.Sin(_theta);

        double temp = (force + _poleMassLength * _thetaDot * _thetaDot * sinTheta) / _totalMass;
        double thetaAcc = (_gravity * sinTheta - cosTheta * temp) /
                          (_length * (4.0 / 3.0 - _massPole * cosTheta * cosTheta / _totalMass));
        double xAcc = temp - _poleMassLength * thetaAcc * cosTheta / _totalMass;

        // Update state
        _x += _tau * _xDot;
        _xDot += _tau * xAcc;
        _theta += _tau * _thetaDot;
        _thetaDot += _tau * thetaAcc;
        _steps++;

        // Check termination conditions
        bool done = _x < -_xThreshold || _x > _xThreshold ||
                    _theta < -_thetaThresholdRadians || _theta > _thetaThresholdRadians ||
                    _steps >= _maxSteps;

        // Reward: +1 for each step the pole is balanced
        T reward = done ? _numOps.Zero : _numOps.One;

        var info = new Dictionary<string, object>
        {
            ["steps"] = _steps,
            ["x"] = _x,
            ["theta"] = _theta
        };

        return (GetStateVector(), reward, done, info);
    }

    /// <inheritdoc/>
    public void Seed(int seed)
    {
        _random = new Random(seed);
    }

    /// <inheritdoc/>
    public void Close()
    {
        // No resources to clean up for this simple environment
    }

    private Vector<T> GetStateVector()
    {
        var state = new Vector<T>(ObservationSpaceDimension);
        state[0] = _numOps.FromDouble(_x);
        state[1] = _numOps.FromDouble(_xDot);
        state[2] = _numOps.FromDouble(_theta);
        state[3] = _numOps.FromDouble(_thetaDot);
        return state;
    }
}
