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

        // Compute norms squared with bias term: sigma_b^2 + sigma_w^2 * <x, y>
        var engine = AiDotNetEngine.Current;
        T biasVariance = _numOps.FromDouble(_biasVariance);
        T weightVariance = _numOps.FromDouble(_weightVariance);
        T norm1Sq = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, engine.DotProduct(x1, x1)));
        T norm2Sq = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, engine.DotProduct(x2, x2)));
        T dotProduct = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, engine.DotProduct(x1, x2)));

        T norm1 = _numOps.Sqrt(norm1Sq);
        T norm2 = _numOps.Sqrt(norm2Sq);

        // Handle degenerate cases
        T degenerateNorm = _numOps.FromDouble(1e-10);
        if (_numOps.LessThan(norm1, degenerateNorm) || _numOps.LessThan(norm2, degenerateNorm))
        {
            return _numOps.Zero;
        }

        // Compute cosine of angle
        T cosTheta = _numOps.Divide(dotProduct, _numOps.Multiply(norm1, norm2));
        // Clamp to valid range for arccos
        cosTheta = MathHelper.Clamp(cosTheta, _numOps.FromDouble(-1.0), _numOps.One);

        // Compute arc-cosine kernel function
        T theta = MathHelper.ArcCos(cosTheta);
        T jn = ComputeArcCosineFunction(theta, cosTheta);

        // Final kernel value
        T order = _numOps.FromDouble(_order);
        T scale = _numOps.Divide(_numOps.Multiply(_numOps.Power(norm1, order), _numOps.Power(norm2, order)), MathHelper.Pi<T>());

        return _numOps.Multiply(scale, jn);
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
    private T ComputeArcCosineFunction(T theta, T cosTheta)
    {
        return ArcCosineJ(_order, theta, cosTheta, _numOps);
    }

    /// <summary>
    /// The arc-cosine angular function J_n(theta) of Cho &amp; Saul (2009) for n = 0, 1, 2.
    /// </summary>
    internal static T ArcCosineJ(int order, T theta, T cosTheta, INumericOperations<T> numOps)
    {
        T sinTheta = MathHelper.Sin(theta);
        T piMinusTheta = numOps.Subtract(MathHelper.Pi<T>(), theta);

        switch (order)
        {
            case 0:
                // J_0(θ) = π - θ
                return piMinusTheta;

            case 1:
                // J_1(θ) = sin(θ) + (π - θ)cos(θ)
                return numOps.Add(sinTheta, numOps.Multiply(piMinusTheta, cosTheta));

            case 2:
                // J_2(θ) = 3sin(θ)cos(θ) + (π - θ)(1 + 2cos²(θ))
                T threeSinCos = numOps.Multiply(numOps.FromDouble(3), numOps.Multiply(sinTheta, cosTheta));
                T onePlusTwoCosSquared = numOps.Add(numOps.One, numOps.Multiply(numOps.FromDouble(2), numOps.Multiply(cosTheta, cosTheta)));
                return numOps.Add(threeSinCos, numOps.Multiply(piMinusTheta, onePlusTwoCosSquared));

            default:
                throw new InvalidOperationException($"Order {order} not supported.");
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

        // Initialize with input norms: sigma_b^2 + sigma_w^2 * <x, y>
        var engine = AiDotNetEngine.Current;
        T biasVariance = _numOps.FromDouble(_biasVariance);
        T weightVariance = _numOps.FromDouble(_weightVariance);
        T K11 = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, engine.DotProduct(x1, x1)));  // K(x1, x1)
        T K22 = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, engine.DotProduct(x2, x2)));  // K(x2, x2)
        T K12 = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, engine.DotProduct(x1, x2)));  // K(x1, x2)

        // Apply layers; each layer is one step of the Cho & Saul (2009) recursion on (K11, K22, K12).
        for (int layer = 0; layer < _numLayers; layer++)
        {
            if (!TryApplyLayer(ref K11, ref K22, ref K12, biasVariance, weightVariance))
            {
                return _numOps.Zero;
            }
        }

        return K12;
    }

    /// <summary>
    /// Advances (K11, K22, K12) by one arc-cosine layer. Returns false when a norm degenerates to zero.
    /// </summary>
    private bool TryApplyLayer(ref T K11, ref T K22, ref T K12, T biasVariance, T weightVariance)
    {
        T order = _numOps.FromDouble(_order);
        T degenerateNorm = _numOps.FromDouble(1e-10);
        T weightOverPi = _numOps.Divide(weightVariance, MathHelper.Pi<T>());

        T norm1 = _numOps.Sqrt(K11);
        T norm2 = _numOps.Sqrt(K22);

        if (_numOps.LessThan(norm1, degenerateNorm) || _numOps.LessThan(norm2, degenerateNorm))
        {
            return false;
        }

        T cosTheta = _numOps.Divide(K12, _numOps.Multiply(norm1, norm2));
        cosTheta = MathHelper.Clamp(cosTheta, _numOps.FromDouble(-1.0), _numOps.One);
        T theta = MathHelper.ArcCos(cosTheta);

        T jn = ComputeJ(theta, cosTheta);
        T scaleFactor = _numOps.Divide(_numOps.Multiply(_numOps.Power(norm1, order), _numOps.Power(norm2, order)), MathHelper.Pi<T>());

        // Update for next layer
        // Self-kernels: K_n^(l)(x,x) = σ_b² + (σ_w² / π) * K^(l-1)(x,x)^n * J_n(0, 1)
        // Fixed: removed extra scaleFactor and norm terms - K11^n = (norm1^2)^n = norm1^(2n)
        T j0 = ComputeJ(_numOps.Zero, _numOps.One); // J_n at θ=0 for self-kernel
        T newK11 = _numOps.Add(biasVariance, _numOps.Multiply(_numOps.Multiply(weightOverPi, j0), _numOps.Power(K11, order)));
        T newK22 = _numOps.Add(biasVariance, _numOps.Multiply(_numOps.Multiply(weightOverPi, j0), _numOps.Power(K22, order)));
        // Cross-kernel: K_n^(l)(x,y) = σ_b² + (σ_w² / π) * ||x||^n * ||y||^n * J_n(θ)
        T newK12 = _numOps.Add(biasVariance, _numOps.Multiply(weightVariance, _numOps.Multiply(scaleFactor, jn)));

        K11 = newK11;
        K22 = newK22;
        K12 = newK12;
        return true;
    }

    /// <summary>
    /// Computes the arc-cosine function J_n(θ).
    /// </summary>
    private T ComputeJ(T theta, T cosTheta)
    {
        return NeuralNetworkKernel<T>.ArcCosineJ(_order, theta, cosTheta, _numOps);
    }
}
