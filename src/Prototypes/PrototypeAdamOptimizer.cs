

namespace AiDotNet.Prototypes;

/// <summary>
/// Prototype Adam optimizer using vectorized operations via PrototypeVector.
/// Demonstrates GPU acceleration through the Execution Engine pattern.
/// </summary>
/// <typeparam name="T">The numeric type for parameters and gradients.</typeparam>
/// <remarks>
/// <para>
/// This is a PROTOTYPE for Phase A validation. The production version will integrate
/// with the existing AdamOptimizer class.
/// </para>
/// <para>
/// Key Difference from Current AdamOptimizer:
/// - BEFORE: Element-wise for-loops (CPU only, slow)
/// - AFTER: Vectorized operations (GPU accelerated when using float)
/// </para>
/// <para>
/// Example:
/// <code>
/// // BEFORE (element-wise):
/// for (int i = 0; i &lt; length; i++)
/// {
///     m[i] = m[i] * beta1 + gradient[i] * (1 - beta1);
/// }
///
/// // AFTER (vectorized):
/// m = m.Multiply(beta1).Add(gradient.Multiply(oneMinusBeta1));
/// </code>
/// </para>
/// </remarks>
public class PrototypeAdamOptimizer<T>
{
    private PrototypeVector<T>? _m;  // First moment estimate
    private PrototypeVector<T>? _v;  // Second moment estimate
    private int _t;  // Time step

    private readonly T _beta1;
    private readonly T _beta2;
    private readonly T _epsilon;
    private readonly T _learningRate;
    private readonly T _one;
    private readonly T _oneMinusBeta1;
    private readonly T _oneMinusBeta2;

    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the PrototypeAdamOptimizer.
    /// </summary>
    /// <param name="learningRate">Learning rate (default: 0.001).</param>
    /// <param name="beta1">Exponential decay rate for first moment (default: 0.9).</param>
    /// <param name="beta2">Exponential decay rate for second moment (default: 0.999).</param>
    /// <param name="epsilon">Small constant for numerical stability (default: 1e-8).</param>
    public PrototypeAdamOptimizer(
        double learningRate = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        _learningRate = _numOps.FromDouble(learningRate);
        _beta1 = _numOps.FromDouble(beta1);
        _beta2 = _numOps.FromDouble(beta2);
        _epsilon = _numOps.FromDouble(epsilon);
        _one = _numOps.One;
        _oneMinusBeta1 = _numOps.FromDouble(1 - beta1);
        _oneMinusBeta2 = _numOps.FromDouble(1 - beta2);

        _t = 0;
    }

    /// <summary>
    /// Updates parameters using the Adam optimization algorithm with vectorized operations.
    /// </summary>
    /// <param name="parameters">Current parameters.</param>
    /// <param name="gradient">Computed gradient.</param>
    /// <returns>Updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// This implementation uses VECTORIZED operations instead of element-wise for-loops.
    /// When using float type with GPU engine enabled, these operations run on the GPU
    /// for massive speedups (10-100x for large parameter vectors).
    /// </para>
    /// <para>
    /// Adam Update Algorithm:
    /// 1. m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
    /// 2. v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
    /// 3. m_hat = m_t / (1 - beta1^t)
    /// 4. v_hat = v_t / (1 - beta2^t)
    /// 5. parameters -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    /// </para>
    /// </remarks>
    public PrototypeVector<T> UpdateParameters(PrototypeVector<T> parameters, PrototypeVector<T> gradient)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException($"Parameters and gradient lengths must match. Got {parameters.Length} and {gradient.Length}");
        }

        // Initialize moment estimates on first call
        if (_m == null || _v == null || _m.Length != parameters.Length)
        {
            _m = PrototypeVector<T>.Zeros(parameters.Length);
            _v = PrototypeVector<T>.Zeros(parameters.Length);
            _t = 0;
        }

        _t++;

        // Compute bias correction terms
        var biasCorrection1 = _numOps.FromDouble(1 - Math.Pow(_numOps.ToDouble(_beta1), _t));
        var biasCorrection2 = _numOps.FromDouble(1 - Math.Pow(_numOps.ToDouble(_beta2), _t));

        // ============================================================
        // VECTORIZED OPERATIONS (GPU Accelerated for float)
        // ============================================================

        // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
        // BEFORE (element-wise): for (int i = 0; i < length; i++) { m[i] = ... }
        // AFTER (vectorized): Single operation on entire vector
        _m = _m.Multiply(_beta1).Add(gradient.Multiply(_oneMinusBeta1));

        // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
        var gradientSquared = gradient.Multiply(gradient);
        _v = _v.Multiply(_beta2).Add(gradientSquared.Multiply(_oneMinusBeta2));

        // Compute bias-corrected first moment: m_hat = m_t / (1 - beta1^t)
        var mHat = _m.Divide(biasCorrection1);

        // Compute bias-corrected second moment: v_hat = v_t / (1 - beta2^t)
        var vHat = _v.Divide(biasCorrection2);

        // Compute update: update = m_hat / (sqrt(v_hat) + epsilon)
        var sqrtVHat = vHat.Sqrt();
        var denominator = sqrtVHat.Add(CreateScalarVector(parameters.Length, _epsilon));
        var update = mHat.Divide(denominator);

        // Apply update: parameters -= learning_rate * update
        var scaledUpdate = update.Multiply(_learningRate);
        var updatedParameters = parameters.Subtract(scaledUpdate);

        return updatedParameters;
    }

    /// <summary>
    /// Resets the optimizer state (clears moment estimates).
    /// </summary>
    public void Reset()
    {
        _m = null;
        _v = null;
        _t = 0;
    }

    /// <summary>
    /// Gets the current time step (number of updates performed).
    /// </summary>
    public int TimeStep => _t;

    /// <summary>
    /// Helper method to create a vector filled with a scalar value.
    /// </summary>
    private PrototypeVector<T> CreateScalarVector(int length, T value)
    {
        var vec = new PrototypeVector<T>(length);
        for (int i = 0; i < length; i++)
        {
            vec[i] = value;
        }
        return vec;
    }

    /// <summary>
    /// Returns a string representation of the optimizer configuration.
    /// </summary>
    public override string ToString()
    {
        return $"PrototypeAdamOptimizer<{typeof(T).Name}>(lr={_numOps.ToDouble(_learningRate)}, " +
               $"beta1={_numOps.ToDouble(_beta1)}, beta2={_numOps.ToDouble(_beta2)}, " +
               $"epsilon={_numOps.ToDouble(_epsilon)}, t={_t})";
    }
}
