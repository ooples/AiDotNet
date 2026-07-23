namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Neural Network (Arc-Cosine) kernel that corresponds to infinitely wide neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Neural Network kernel (also called Arc-Cosine kernel) is fascinating
/// because it exactly corresponds to the behavior of infinitely wide neural networks with
/// specific activation functions.
///
/// Key insight: As a neural network gets wider and wider (more neurons in each layer),
/// its behavior becomes more predictable and can be described by a Gaussian Process with
/// this kernel. This connection is called the "Neural Network Gaussian Process" (NNGP).
///
/// The kernel depends on the activation function:
/// - Order 0: Step function (Heaviside) - Measures angle between inputs
/// - Order 1: ReLU - Captures piecewise linear behavior
/// - Order 2: ReQU (Rectified Quadratic Unit) - Smoother, more expressive
/// </para>
/// <para>
/// Mathematical form:
/// k_n(x, x') = (1/π) × ||x||^n × ||x'||^n × J_n(θ)
///
/// Where:
/// - θ = angle between x and x' (arccos of normalized dot product)
/// - J_n(θ) = (-1)^n × (sin(θ))^(2n+1) × (∂/∂t)^n [(π-θ)/sin(θ)] evaluated at t=cos(θ)
///
/// For n=1 (ReLU): J_1(θ) = sin(θ) + (π-θ)cos(θ)
/// </para>
/// <para>
/// Why use this kernel?
///
/// 1. **Neural Network Connection**: Analyze what infinite neural networks can learn
/// 2. **Deep Architectures**: Can be composed for "deep" kernel behavior
/// 3. **Non-stationary**: Unlike RBF, it's not translation-invariant
/// 4. **Theoretical Insights**: Helps understand deep learning through GP lens
/// </para>
/// </remarks>
public class NeuralNetworkKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The order of the kernel (corresponds to activation function type).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - Order 0: Step activation → Measures just angle similarity
    /// - Order 1: ReLU activation → Most common, captures linear+nonlinear
    /// - Order 2: ReQU activation → Smoother, better gradients
    /// </para>
    /// </remarks>
    private readonly int _order;

    /// <summary>
    /// The weight variance parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This scales the kernel output. Corresponds to the
    /// variance of the weight distribution in the equivalent neural network.
    /// Larger values = more variable outputs.
    /// </para>
    /// </remarks>
    private readonly double _weightVariance;

    /// <summary>
    /// The bias variance parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adds a constant component to the kernel.
    /// Corresponds to the variance of the bias terms in the neural network.
    /// Set to 0 for no bias.
    /// </para>
    /// </remarks>
    private readonly double _biasVariance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Neural Network kernel.
    /// </summary>
    /// <param name="order">The kernel order (0, 1, or 2). Default is 1 (ReLU).</param>
    /// <param name="weightVariance">The weight variance. Default is 1.0.</param>
    /// <param name="biasVariance">The bias variance. Default is 0.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Neural Network kernel.
    ///
    /// Recommended settings:
    /// - Order 1 (ReLU) is most common and works well for most problems
    /// - weightVariance=1.0 is a good default
    /// - biasVariance=0.0 makes the kernel simpler (no constant term)
    ///
    /// Example:
    /// var kernel = new NeuralNetworkKernel&lt;double&gt;(order: 1);
    /// </para>
    /// </remarks>
    public NeuralNetworkKernel(int order = 1, double weightVariance = 1.0, double biasVariance = 0.0)
    {
        if (order < 0 || order > 2)
            throw new ArgumentException("Order must be 0, 1, or 2.", nameof(order));
        if (weightVariance <= 0)
            throw new ArgumentException("Weight variance must be positive.", nameof(weightVariance));
        if (biasVariance < 0)
            throw new ArgumentException("Bias variance must be non-negative.", nameof(biasVariance));

        _order = order;
        _weightVariance = weightVariance;
        _biasVariance = biasVariance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the kernel order.
    /// </summary>
    public int Order => _order;

    /// <summary>
    /// Gets the weight variance.
    /// </summary>
    public double WeightVariance => _weightVariance;

    /// <summary>
    /// Gets the bias variance.
    /// </summary>
    public double BiasVariance => _biasVariance;

    /// <summary>
    /// Calculates the Neural Network kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the covariance between outputs of an infinitely
    /// wide neural network given inputs x1 and x2.
    ///
    /// The calculation:
    /// 1. Compute norms: ||x1|| and ||x2||
    /// 2. Compute angle: θ = arccos(dot(x1,x2) / (||x1|| × ||x2||))
    /// 3. Apply the arc-cosine function J_n(θ)
    /// 4. Scale by norms and variance
    ///
    /// The result tells you how correlated the neural network outputs would be
    /// for these two inputs, averaged over all possible weight initializations.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Compute norms squared with bias term
        double norm1Sq = _biasVariance;
        double norm2Sq = _biasVariance;
        double dotProduct = _biasVariance;

        for (int i = 0; i < x1.Length; i++)
        {
            double v1 = _numOps.ToDouble(x1[i]);
            double v2 = _numOps.ToDouble(x2[i]);

            norm1Sq += _weightVariance * v1 * v1;
            norm2Sq += _weightVariance * v2 * v2;
            dotProduct += _weightVariance * v1 * v2;
        }

        double norm1 = Math.Sqrt(norm1Sq);
        double norm2 = Math.Sqrt(norm2Sq);

        // Handle degenerate cases
        if (norm1 < 1e-10 || norm2 < 1e-10)
        {
            return _numOps.FromDouble(0.0);
        }

        // Compute cosine of angle
        double cosTheta = dotProduct / (norm1 * norm2);
        // Clamp to valid range for arccos
        cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));

        // Compute arc-cosine kernel function
        double theta = Math.Acos(cosTheta);
        double jn = ComputeArcCosineFunction(theta, cosTheta);

        // Final kernel value
        double result = (1.0 / Math.PI) * Math.Pow(norm1, _order) * Math.Pow(norm2, _order) * jn;

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Computes the arc-cosine function J_n(θ).
    /// </summary>
    /// <param name="theta">The angle in radians.</param>
    /// <param name="cosTheta">The cosine of the angle.</param>
    /// <returns>The J_n function value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the heart of the Neural Network kernel.
    /// Different orders give different behaviors:
    ///
    /// - J_0(θ) = π - θ
    ///   Simplest form, just depends on angle
    ///
    /// - J_1(θ) = sin(θ) + (π - θ)cos(θ)
    ///   Corresponds to ReLU activation
    ///
    /// - J_2(θ) = 3sin(θ)cos(θ) + (π - θ)(1 + 2cos²(θ))
    ///   Corresponds to ReQU activation
    /// </para>
    /// </remarks>
    private double ComputeArcCosineFunction(double theta, double cosTheta)
    {
        double sinTheta = Math.Sin(theta);
        double piMinusTheta = Math.PI - theta;

        switch (_order)
        {
            case 0:
                // J_0(θ) = π - θ
                return piMinusTheta;

            case 1:
                // J_1(θ) = sin(θ) + (π - θ)cos(θ)
                return sinTheta + piMinusTheta * cosTheta;

            case 2:
                // J_2(θ) = 3sin(θ)cos(θ) + (π - θ)(1 + 2cos²(θ))
                return 3 * sinTheta * cosTheta + piMinusTheta * (1 + 2 * cosTheta * cosTheta);

            default:
                throw new InvalidOperationException($"Order {_order} not supported.");
        }
    }

    /// <summary>
    /// Creates a "deep" version of the kernel by composing multiple layers.
    /// </summary>
    /// <param name="numLayers">The number of layers.</param>
    /// <returns>A composed deep kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Just like neural networks have multiple layers,
    /// we can compose this kernel to create a "deep" GP kernel.
    ///
    /// Each additional layer:
    /// - Takes the output of the previous layer as input
    /// - Applies another arc-cosine transformation
    ///
    /// This can capture more complex patterns, similar to how deeper
    /// neural networks can learn more complex functions.
    ///
    /// Note: This returns a new kernel that represents the composition.
    /// The actual implementation computes the kernel iteratively.
    /// </para>
    /// </remarks>
    public DeepNeuralNetworkKernel<T> ToDeep(int numLayers)
    {
        if (numLayers < 1)
            throw new ArgumentException("Must have at least one layer.", nameof(numLayers));

        return new DeepNeuralNetworkKernel<T>(_order, _weightVariance, _biasVariance, numLayers);
    }
}

/// <summary>
/// Implements a deep (multi-layer) Neural Network kernel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This kernel corresponds to a deep neural network with multiple layers.
///
/// In each layer:
/// 1. Input from previous layer goes through the arc-cosine kernel transformation
/// 2. Output becomes input to the next layer
///
/// More layers = more compositional expressiveness, similar to deep neural networks.
/// However, very deep kernels can have vanishing/exploding gradients just like deep NNs.
/// </para>
/// </remarks>
public class DeepNeuralNetworkKernel<T> : IKernelFunction<T>
{
    private readonly int _order;
    private readonly double _weightVariance;
    private readonly double _biasVariance;
    private readonly int _numLayers;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a deep Neural Network kernel.
    /// </summary>
    /// <param name="order">The kernel order for each layer.</param>
    /// <param name="weightVariance">The weight variance for each layer.</param>
    /// <param name="biasVariance">The bias variance for each layer.</param>
    /// <param name="numLayers">The number of layers.</param>
    public DeepNeuralNetworkKernel(int order, double weightVariance, double biasVariance, int numLayers)
    {
        if (order < 0 || order > 2)
            throw new ArgumentException("Order must be 0, 1, or 2.", nameof(order));
        if (weightVariance <= 0)
            throw new ArgumentException("Weight variance must be positive.", nameof(weightVariance));
        if (biasVariance < 0)
            throw new ArgumentException("Bias variance must be non-negative.", nameof(biasVariance));
        if (numLayers < 1)
            throw new ArgumentException("Must have at least one layer.", nameof(numLayers));

        _order = order;
        _weightVariance = weightVariance;
        _biasVariance = biasVariance;
        _numLayers = numLayers;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of layers.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Calculates the deep Neural Network kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value after all layers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the kernel by passing through multiple layers.
    ///
    /// For each layer l:
    /// - Compute K^(l)_11 = kernel(x1, x1) from previous layer
    /// - Compute K^(l)_22 = kernel(x2, x2) from previous layer
    /// - Compute K^(l)_12 = kernel(x1, x2) from previous layer
    /// - These become the "norms" and "dot product" for the next layer
    ///
    /// The first layer uses the actual input vectors.
    /// Subsequent layers use the kernel values from the previous layer.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Initialize with input norms
        double K11 = _biasVariance;  // K(x1, x1)
        double K22 = _biasVariance;  // K(x2, x2)
        double K12 = _biasVariance;  // K(x1, x2)

        for (int i = 0; i < x1.Length; i++)
        {
            double v1 = _numOps.ToDouble(x1[i]);
            double v2 = _numOps.ToDouble(x2[i]);

            K11 += _weightVariance * v1 * v1;
            K22 += _weightVariance * v2 * v2;
            K12 += _weightVariance * v1 * v2;
        }

        // Apply layers
        for (int layer = 0; layer < _numLayers; layer++)
        {
            double norm1 = Math.Sqrt(K11);
            double norm2 = Math.Sqrt(K22);

            if (norm1 < 1e-10 || norm2 < 1e-10)
            {
                return _numOps.FromDouble(0.0);
            }

            double cosTheta = K12 / (norm1 * norm2);
            cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));
            double theta = Math.Acos(cosTheta);

            double jn = ComputeJ(theta, cosTheta);
            double scaleFactor = (1.0 / Math.PI) * Math.Pow(norm1, _order) * Math.Pow(norm2, _order);

            // Update for next layer
            // Self-kernels: K_n^(l)(x,x) = σ_b² + (σ_w² / π) * K^(l-1)(x,x)^n * J_n(0, 1)
            // Fixed: removed extra scaleFactor and norm terms - K11^n = (norm1^2)^n = norm1^(2n)
            double j0 = ComputeJ(0, 1.0); // J_n at θ=0 for self-kernel
            double newK11 = _biasVariance + _weightVariance * j0 / Math.PI * Math.Pow(K11, _order);
            double newK22 = _biasVariance + _weightVariance * j0 / Math.PI * Math.Pow(K22, _order);
            // Cross-kernel: K_n^(l)(x,y) = σ_b² + (σ_w² / π) * ||x||^n * ||y||^n * J_n(θ)
            double newK12 = _biasVariance + _weightVariance * scaleFactor * jn;

            K11 = newK11;
            K22 = newK22;
            K12 = newK12;
        }

        return _numOps.FromDouble(K12);
    }

    /// <summary>
    /// Computes the arc-cosine function J_n(θ).
    /// </summary>
    private double ComputeJ(double theta, double cosTheta)
    {
        double sinTheta = Math.Sin(theta);
        double piMinusTheta = Math.PI - theta;

        return _order switch
        {
            0 => piMinusTheta,
            1 => sinTheta + piMinusTheta * cosTheta,
            2 => 3 * sinTheta * cosTheta + piMinusTheta * (1 + 2 * cosTheta * cosTheta),
            _ => throw new InvalidOperationException($"Order {_order} not supported.")
        };
    }
}
