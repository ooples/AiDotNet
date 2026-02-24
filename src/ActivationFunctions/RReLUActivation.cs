using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Randomized Rectified Linear Unit (RReLU) activation function.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RReLU is a variation of the ReLU activation function that introduces randomness during training.
/// For positive inputs, it behaves like standard ReLU (returning the input unchanged).
/// For negative inputs, it multiplies the input by a random factor (alpha) between the specified lower and upper bounds.
/// </para>
/// <para>
/// <b>For Beginners:</b> RReLU (Randomized Rectified Linear Unit) is like a "smart filter" for neural networks.
/// When data is positive, it lets it pass through unchanged. When data is negative, instead of setting it to zero
/// (like regular ReLU), it reduces the value by multiplying it by a small random number. This randomness helps
/// prevent "dead neurons" (neurons that stop learning) and can improve the network's ability to learn.
/// During training, this random factor changes; during testing/inference, a fixed average value is used.
/// </para>
/// </remarks>
public class RReLUActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Random number generator used to create the alpha value during training.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// The minimum value for the random alpha parameter.
    /// </summary>
    private readonly T _lowerBound;

    /// <summary>
    /// The maximum value for the random alpha parameter.
    /// </summary>
    private readonly T _upperBound;

    /// <summary>
    /// The current alpha value used to scale negative inputs.
    /// </summary>
    private T _alpha;

    /// <summary>
    /// Indicates whether the activation function is in training mode (true) or inference mode (false).
    /// </summary>
    private bool _isTraining;

    /// <summary>
    /// Initializes a new instance of the RReLU activation function with specified bounds for the random alpha parameter.
    /// </summary>
    /// <param name="lowerBound">The lower bound for the random alpha parameter. Default is 1/8 (0.125).</param>
    /// <param name="upperBound">The upper bound for the random alpha parameter. Default is 1/3 (0.333...).</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the RReLU function with boundaries for how much negative values
    /// will be reduced. The default values (1/8 to 1/3) are commonly used in practice and mean that negative values
    /// will be multiplied by a random number between 0.125 and 0.333, making them smaller but not zero.
    /// </remarks>
    public RReLUActivation(double lowerBound = 1.0 / 8, double upperBound = 1.0 / 3)
    {
        _random = RandomHelper.CreateSecureRandom();
        _lowerBound = NumOps.FromDouble(lowerBound);
        _upperBound = NumOps.FromDouble(upperBound);
        _alpha = NumOps.FromDouble((_random.NextDouble() * (upperBound - lowerBound)) + lowerBound);
        _isTraining = true;
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as RReLU supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the RReLU activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>
    /// The input value if it's greater than or equal to zero, 
    /// otherwise the input multiplied by a random alpha value.
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method processes a single number through the RReLU function.
    /// If the number is positive or zero, it's returned unchanged.
    /// If the number is negative, it's multiplied by a small random value (during training)
    /// or by a fixed value (during testing/inference) to make it smaller but not zero.
    /// </remarks>
    public override T Activate(T input)
    {
        if (_isTraining)
        {
            _alpha = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(_random.NextDouble()), NumOps.Subtract(_upperBound, _lowerBound)), _lowerBound);
        }

        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return input;
        }
        else
        {
            return NumOps.Multiply(_alpha, input);
        }
    }

    /// <summary>
    /// Calculates the derivative of the RReLU function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>1 if the input is greater than or equal to zero, otherwise the current alpha value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// For RReLU, the derivative is 1 for positive inputs (meaning the output changes at the same rate as the input),
    /// and the alpha value for negative inputs (meaning the output changes at a reduced rate when the input changes).
    /// </remarks>
    public override T Derivative(T input)
    {
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return NumOps.One;
        }
        else
        {
            return _alpha;
        }
    }

    /// <summary>
    /// Sets the activation function to either training or inference mode.
    /// </summary>
    /// <param name="isTraining">True to set to training mode, false to set to inference mode.</param>
    /// <remarks>
    /// <para>
    /// In training mode, a new random alpha value is generated for each activation.
    /// In inference mode, alpha is fixed to the average of the lower and upper bounds.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method lets you switch between two modes:
    /// - Training mode: Uses random values for processing negative inputs, which helps with learning
    /// - Inference mode (testing/production): Uses a fixed value (the average of your bounds) for consistency
    /// 
    /// You should typically use training mode when training your neural network and inference mode
    /// when using the trained network to make predictions.
    /// </para>
    /// </remarks>
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
        if (!_isTraining)
        {
            // Set alpha to the average of lower and upper bounds for inference
            _alpha = NumOps.Divide(NumOps.Add(_lowerBound, _upperBound), NumOps.FromDouble(2));
        }
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.RReLU provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// RReLU supports JIT compilation with the following behavior:
    /// - In inference mode (default for JIT): uses fixed alpha = (lower + upper) / 2
    /// - In training mode: samples alpha once per forward pass (not per-element)
    /// </para>
    /// <para>
    /// This is a reasonable compromise that enables JIT while preserving the randomization benefit during training.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with RReLU activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.RReLU(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        double lower = Convert.ToDouble(_lowerBound);
        double upper = Convert.ToDouble(_upperBound);
        return TensorOperations<T>.RReLU(input, lower, upper, _isTraining);
    }
}
