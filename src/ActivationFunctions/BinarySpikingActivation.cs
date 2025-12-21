using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Binary Spiking activation function for neural networks, particularly for spiking neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Binary Spiking activation models how real neurons in your brain generate electrical pulses.
/// 
/// Unlike standard activation functions that output continuous values, Binary Spiking:
/// - Outputs only 1 (spike/fire) or 0 (no spike)
/// - Neurons "fire" only when their input exceeds a threshold
/// - After firing, neurons typically have a "refractory period" before they can fire again
/// 
/// This activation creates the discrete, all-or-nothing behavior of biological neurons:
/// - Input below threshold ? Output = 0 (neuron remains silent)
/// - Input at or above threshold ? Output = 1 (neuron fires a spike)
/// 
/// Common uses include:
/// - Spiking Neural Networks (SNNs)
/// - Neuromorphic computing systems
/// - Models that process temporal information
/// - Energy-efficient neural networks for specialized hardware
/// </para>
/// </remarks>
public class BinarySpikingActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The firing threshold for neurons. Inputs equal to or greater than this value will cause the neuron to fire.
    /// </summary>
    private readonly T _threshold;

    /// <summary>
    /// The slope of the approximated derivative curve used during training.
    /// </summary>
    private readonly T _derivativeSlope;

    /// <summary>
    /// The width of the region around the threshold where the derivative is non-zero.
    /// </summary>
    private readonly T _derivativeWidth;

    /// <summary>
    /// Initializes a new instance of the Binary Spiking activation function with default parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a Binary Spiking activation with standard settings.
    /// 
    /// The default settings:
    /// - Threshold = 1.0: Neurons fire when their input reaches or exceeds 1.0
    /// - Derivative Slope = 1.0: Controls how quickly the function changes around the threshold
    /// - Derivative Width = 0.2: Sets the size of the region where learning can occur
    /// 
    /// These settings work well for most spiking neural networks, but you can customize them
    /// with the other constructor if you have specific requirements.
    /// </para>
    /// </remarks>
    public BinarySpikingActivation() : this(NumOps.One, NumOps.One, NumOps.FromDouble(0.2))
    {
    }

    /// <summary>
    /// Initializes a new instance of the Binary Spiking activation function with custom parameters.
    /// </summary>
    /// <param name="threshold">The firing threshold for neurons.</param>
    /// <param name="derivativeSlope">The slope of the approximated derivative used during training.</param>
    /// <param name="derivativeWidth">The width of the region around the threshold where the derivative is non-zero.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor lets you customize the spiking behavior of neurons.
    /// 
    /// You can adjust:
    /// - Threshold: How much input is needed before neurons fire (higher = harder to activate)
    /// - Derivative Slope: Affects how quickly neurons learn (higher = faster but potentially unstable learning)
    /// - Derivative Width: Controls how precise inputs need to be (smaller = more precise targeting)
    /// 
    /// Customizing these parameters can help optimize for different types of data or learning tasks.
    /// For example, a lower threshold might be better for detecting subtle patterns, while
    /// a higher threshold could help filter out noise.
    /// </para>
    /// </remarks>
    public BinarySpikingActivation(T threshold, T derivativeSlope, T derivativeWidth)
    {
        _threshold = threshold;
        _derivativeSlope = derivativeSlope;
        _derivativeWidth = derivativeWidth;
    }

    /// <summary>
    /// Indicates whether this activation function can operate on individual scalar values.
    /// </summary>
    /// <returns>Always returns true as Binary Spiking can be applied to individual values.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Binary Spiking activation function to a single value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>1 if the input meets or exceeds the threshold, 0 otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines whether a single neuron should fire based on its input.
    /// 
    /// The simple rule is:
    /// - If input = threshold: Output = 1 (neuron fires)
    /// - If input < threshold: Output = 0 (neuron stays silent)
    /// 
    /// This binary (on/off) behavior is what gives spiking neurons their distinctive property
    /// and makes them more similar to biological neurons than traditional artificial neurons.
    /// </para>
    /// </remarks>
    public override T Activate(T x)
    {
        return NumOps.GreaterThanOrEquals(x, _threshold) ? NumOps.One : NumOps.Zero;
    }

    /// <summary>
    /// Applies the Binary Spiking activation function to transform input vectors into binary spike outputs.
    /// </summary>
    /// <param name="input">The input vector to transform.</param>
    /// <returns>A vector containing binary values (0 or 1) for each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method processes multiple neurons at once, determining which ones should fire.
    /// 
    /// For each neuron in the input:
    /// - If its value = threshold: Output = 1 (neuron fires)
    /// - If its value < threshold: Output = 0 (neuron stays silent)
    /// 
    /// The result is a binary pattern of active and inactive neurons, similar to how
    /// groups of neurons in your brain form firing patterns when processing information.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> output = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }
        return output;
    }

    /// <summary>
    /// Calculates the approximate derivative of the Binary Spiking function for a single value.
    /// </summary>
    /// <param name="x">The input value at which to calculate the derivative.</param>
    /// <returns>An approximation of the derivative at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The true derivative of a step function is zero almost everywhere, with an
    /// infinite spike at the threshold - which isn't useful for learning. Instead, we use an approximation.
    /// 
    /// This method returns:
    /// - A positive value when the input is near the threshold (within the derivative width)
    /// - Zero when the input is far from the threshold
    /// 
    /// This approximation allows the neural network to learn even though the actual function
    /// has a "jump" at the threshold. It's like creating a small "hill" around the threshold
    /// that guides the learning process.
    /// </para>
    /// </remarks>
    public override T Derivative(T x)
    {
        // Binary spiking function has a discontinuous derivative (Dirac delta)
        // We approximate it with a triangular function centered at the threshold
        T distance = NumOps.Abs(NumOps.Subtract(x, _threshold));

        if (NumOps.LessThan(distance, _derivativeWidth))
        {
            // Create a triangular pulse around the threshold
            T normalizedDistance = NumOps.Divide(distance, _derivativeWidth);
            return NumOps.Multiply(_derivativeSlope, NumOps.Subtract(NumOps.One, normalizedDistance));
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Calculates the derivative (gradient) of the Binary Spiking function for a vector input.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A diagonal Jacobian matrix containing the approximate derivatives.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how sensitive each output is to small changes in each input.
    /// 
    /// Since each output only depends on its corresponding input (not on other inputs), 
    /// the result is a diagonal matrix where:
    /// - Diagonal elements show how quickly each neuron's output changes near the threshold
    /// - Off-diagonal elements are zero (one neuron's output doesn't depend on other neurons' inputs)
    /// 
    /// This information helps the neural network learn by showing which weights need adjusting
    /// and in what direction to adjust them during training.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        // For element-wise functions, the Jacobian is diagonal
        Matrix<T> jacobian = new Matrix<T>(input.Length, input.Length);

        for (int i = 0; i < input.Length; i++)
        {
            // Set the diagonal elements to the derivative values
            jacobian[i, i] = Derivative(input[i]);

            // Off-diagonal elements remain zero as there's no cross-dependency
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the Binary Spiking activation function to a tensor input.
    /// </summary>
    /// <param name="input">The input tensor to transform.</param>
    /// <returns>A tensor containing binary values (0 or 1) for each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method handles multi-dimensional data structures like 2D or 3D arrays.
    /// 
    /// It applies the same binary threshold rule to every element in the tensor:
    /// - If element = threshold: Output = 1 (neuron fires)
    /// - If element < threshold: Output = 0 (neuron stays silent)
    /// 
    /// This is useful for processing structured data like images or time sequences in
    /// spiking neural networks while preserving the structure of the data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);

        // Apply the activation to each element in the tensor
        for (int i = 0; i < input.Length; i++)
        {
            output.SetFlatIndex(i, Activate(input.GetFlatIndexValue(i)));
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative of the Binary Spiking function for a tensor input.
    /// </summary>
    /// <param name="input">The input tensor at which to calculate the derivative.</param>
    /// <returns>A tensor containing the approximate derivatives.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates sensitivity information for multi-dimensional data.
    /// 
    /// For each element in the input tensor, it computes how sensitive that element's output
    /// is to small changes in the input value. The result has the same shape as the input,
    /// with each position containing the derivative at that position.
    /// 
    /// This helps the network learn efficiently when working with structured data like
    /// images or sequences in spiking neural networks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> derivatives = new Tensor<T>(input.Shape);

        // Calculate the derivative for each element in the tensor
        for (int i = 0; i < input.Length; i++)
        {
            derivatives.SetFlatIndex(i, Derivative(input.GetFlatIndexValue(i)));
        }

        return derivatives;
    }

    /// <summary>
    /// Gets the firing threshold value used by this activation function.
    /// </summary>
    /// <returns>The current threshold value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method returns the current threshold that determines when neurons fire.
    /// 
    /// The threshold is the minimum input value needed for a neuron to generate a spike.
    /// Having access to this value can be useful for:
    /// - Debugging network behavior
    /// - Adjusting the threshold dynamically during training
    /// - Understanding how sensitive your neurons are
    /// </para>
    /// </remarks>
    public T GetThreshold()
    {
        return _threshold;
    }

    /// <summary>
    /// Creates a new instance of the Binary Spiking activation with a different threshold.
    /// </summary>
    /// <param name="newThreshold">The new threshold value to use.</param>
    /// <returns>A new BinarySpikingActivation instance with the updated threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a version of this activation function with a different threshold.
    /// 
    /// Since the original activation function is immutable (can't be changed after creation),
    /// this method gives you a way to get a new version with different settings.
    /// 
    /// This is useful when:
    /// - You want to try different thresholds without modifying the original
    /// - You need to adjust sensitivity during training
    /// - You're implementing adaptive spiking networks where thresholds change over time
    /// </para>
    /// </remarks>
    public BinarySpikingActivation<T> WithThreshold(T newThreshold)
    {
        return new BinarySpikingActivation<T>(newThreshold, _derivativeSlope, _derivativeWidth);
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.SurrogateSpike provides surrogate gradient support for spiking networks.</value>
    /// <remarks>
    /// <para>
    /// Binary spiking supports JIT compilation using surrogate gradients. The forward pass produces
    /// hard spikes (0 or 1), while the backward pass uses a sigmoid surrogate for gradient flow.
    /// This enables training of spiking neural networks with standard backpropagation.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with surrogate spike activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.SurrogateSpike(input) which uses the
    /// straight-through estimator pattern: hard spikes in forward pass, sigmoid surrogate
    /// gradients in backward pass.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double threshold = Convert.ToDouble(_threshold);
        double surrogateBeta = Convert.ToDouble(_derivativeSlope);
        return TensorOperations<T>.SurrogateSpike(input, threshold, surrogateBeta);
    }
}
