using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Environments;

/// <summary>
/// Implements a CartPole environment, a classic reinforcement learning benchmark problem.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The CartPole environment simulates a pole balanced on a cart. The goal is to keep the pole upright
/// by moving the cart left or right. This is a classic control problem used to test reinforcement
/// learning algorithms. The environment terminates when the pole falls too far from vertical or the
/// cart moves too far from the center.
/// </para>
/// <para><b>For Beginners:</b> CartPole is like balancing a broomstick on your hand.
///
/// The setup:
/// - A cart that can move left or right on a track
/// - A pole attached to the cart that can tip over
/// - Your goal: Keep the pole balanced upright
///
/// What you control:
/// - Action 0: Push cart left
/// - Action 1: Push cart right
///
/// What you observe (state):
/// - Cart position (where the cart is on the track)
/// - Cart velocity (how fast it's moving)
/// - Pole angle (how tilted the pole is)
/// - Pole angular velocity (how fast it's tipping)
///
/// Rewards:
/// - +1 for each time step the pole stays up
/// - Episode ends when pole falls too far or cart goes off track
///
/// Why it's useful:
/// - Simple enough to solve quickly
/// - Complex enough to require learning
/// - Standard benchmark to compare algorithms
/// - Teaches basics of continuous state control
///
/// Think of it like a video game where you're trying to balance something - keep it going as long
/// as possible to get a high score!
/// </para>
/// </remarks>
public class CartPoleEnvironment<T> : IEnvironment<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;

    // Physics constants
    private readonly T _gravity;
    private readonly T _massCart;
    private readonly T _massPole;
    private readonly T _totalMass;
    private readonly T _length; // Half-length of pole
    private readonly T _poleMassLength;
    private readonly T _forceMag;
    private readonly T _tau; // Time step

    // Thresholds for episode termination
    private readonly T _thetaThresholdRadians;
    private readonly T _xThreshold;

    // Current state
    private T _x; // Cart position
    private T _xDot; // Cart velocity
    private T _theta; // Pole angle
    private T _thetaDot; // Pole angular velocity

    private int _steps;
    private readonly int _maxSteps;

    /// <inheritdoc/>
    public int ObservationSpaceDimension => 4; // x, x_dot, theta, theta_dot

    /// <inheritdoc/>
    public int ActionSpaceSize => 2; // left or right

    /// <summary>
    /// Initializes a new instance of the <see cref="CartPoleEnvironment{T}"/> class.
    /// </summary>
    /// <param name="maxSteps">The maximum number of steps per episode. Default is 500.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// Creates a new CartPole environment with standard physics parameters. The environment will
    /// automatically terminate after maxSteps to prevent infinite episodes.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new CartPole game.
    ///
    /// Parameters:
    /// - maxSteps: How long each episode can last (500 is standard)
    ///   * Prevents episodes from going on forever
    ///   * Also acts as a "success" threshold - if you reach 500 steps, you've solved it!
    /// - seed: Optional number for predictable randomness (useful for debugging)
    ///
    /// The environment is set up with realistic physics:
    /// - Gravity: 9.8 m/s²
    /// - Cart mass: 1.0 kg
    /// - Pole mass: 0.1 kg
    /// - Pole length: 0.5 m
    /// - Time step: 0.02 seconds (50 FPS)
    ///
    /// Episode ends when:
    /// - Pole tilts more than 12 degrees (0.2095 radians)
    /// - Cart moves more than 2.4 units from center
    /// - Maximum steps reached
    /// </para>
    /// </remarks>
    public CartPoleEnvironment(int maxSteps = 500, int? seed = null)
    {
        _numOps = NumericOperations<T>.Instance;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        // Physics constants (standard CartPole parameters)
        _gravity = _numOps.FromDouble(9.8);
        _massCart = _numOps.FromDouble(1.0);
        _massPole = _numOps.FromDouble(0.1);
        _totalMass = _numOps.Add(_massCart, _massPole);
        _length = _numOps.FromDouble(0.5); // Half-length of pole
        _poleMassLength = _numOps.Multiply(_massPole, _length);
        _forceMag = _numOps.FromDouble(10.0);
        _tau = _numOps.FromDouble(0.02); // Time step (50 FPS)

        // Termination thresholds
        _thetaThresholdRadians = _numOps.FromDouble(12.0 * 2.0 * Math.PI / 360.0); // 12 degrees
        _xThreshold = _numOps.FromDouble(2.4);

        _maxSteps = maxSteps;
        _steps = 0;

        // Initialize state (will be randomized in Reset)
        _x = _numOps.Zero;
        _xDot = _numOps.Zero;
        _theta = _numOps.Zero;
        _thetaDot = _numOps.Zero;
    }

    /// <inheritdoc/>
    public Tensor<T> Reset()
    {
        // Initialize state with small random values
        _x = _numOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
        _xDot = _numOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
        _theta = _numOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
        _thetaDot = _numOps.FromDouble(_random.NextDouble() * 0.1 - 0.05);
        _steps = 0;

        return GetState();
    }

    /// <inheritdoc/>
    public (Tensor<T> nextState, T reward, bool done, Dictionary<string, object>? info) Step(int action)
    {
        if (action < 0 || action >= ActionSpaceSize)
        {
            throw new ArgumentException($"Invalid action: {action}. Must be 0 or 1.", nameof(action));
        }

        // Determine force direction
        T force = action == 1 ? _forceMag : _numOps.Negate(_forceMag);

        // Physics simulation (semi-implicit Euler method)
        T cosTheta = _numOps.FromDouble(Math.Cos(_numOps.ToDouble(_theta)));
        T sinTheta = _numOps.FromDouble(Math.Sin(_numOps.ToDouble(_theta)));

        // Calculate angular acceleration
        T temp = _numOps.Divide(
            _numOps.Add(force, _numOps.Multiply(_poleMassLength, _numOps.Multiply(_thetaDot, _thetaDot), sinTheta)),
            _totalMass
        );

        T thetaAcc = _numOps.Divide(
            _numOps.Subtract(
                _numOps.Multiply(_gravity, sinTheta),
                _numOps.Multiply(cosTheta, temp)
            ),
            _numOps.Multiply(
                _length,
                _numOps.Subtract(
                    _numOps.FromDouble(4.0 / 3.0),
                    _numOps.Divide(_numOps.Multiply(_massPole, _numOps.Multiply(cosTheta, cosTheta)), _totalMass)
                )
            )
        );

        // Calculate cart acceleration
        T xAcc = _numOps.Subtract(
            temp,
            _numOps.Divide(_numOps.Multiply(_poleMassLength, thetaAcc, cosTheta), _totalMass)
        );

        // Update velocities and positions
        _x = _numOps.Add(_x, _numOps.Multiply(_tau, _xDot));
        _xDot = _numOps.Add(_xDot, _numOps.Multiply(_tau, xAcc));
        _theta = _numOps.Add(_theta, _numOps.Multiply(_tau, _thetaDot));
        _thetaDot = _numOps.Add(_thetaDot, _numOps.Multiply(_tau, thetaAcc));

        _steps++;

        // Check termination conditions
        bool done = _numOps.LessThan(_x, _numOps.Negate(_xThreshold)) ||
                    _numOps.GreaterThan(_x, _xThreshold) ||
                    _numOps.LessThan(_theta, _numOps.Negate(_thetaThresholdRadians)) ||
                    _numOps.GreaterThan(_theta, _thetaThresholdRadians) ||
                    _steps >= _maxSteps;

        // Reward is 1 for each step the pole stays up
        T reward = done ? _numOps.Zero : _numOps.FromDouble(1.0);

        var info = new Dictionary<string, object>
        {
            { "steps", _steps }
        };

        return (GetState(), reward, done, info);
    }

    /// <inheritdoc/>
    public void Render()
    {
        // Simple text-based rendering
        Console.WriteLine($"Step: {_steps}");
        Console.WriteLine($"Cart Position: {_numOps.ToDouble(_x):F3}");
        Console.WriteLine($"Cart Velocity: {_numOps.ToDouble(_xDot):F3}");
        Console.WriteLine($"Pole Angle: {_numOps.ToDouble(_theta) * 180 / Math.PI:F3}°");
        Console.WriteLine($"Pole Angular Velocity: {_numOps.ToDouble(_thetaDot):F3}");
        Console.WriteLine();
    }

    /// <inheritdoc/>
    public void Close()
    {
        // No resources to clean up for this simple environment
    }

    /// <inheritdoc/>
    public void Seed(int seed)
    {
        _random = new Random(seed);
    }

    /// <summary>
    /// Gets the current state as a tensor.
    /// </summary>
    /// <returns>A tensor containing [x, x_dot, theta, theta_dot].</returns>
    private Tensor<T> GetState()
    {
        return new Tensor<T>(new[] { _x, _xDot, _theta, _thetaDot }, [4]);
    }
}
