using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Maxout activation function for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Maxout activation function is different from most activation functions.
/// Instead of applying a mathematical formula to each value, it:
/// 
/// 1. Groups your input values into small sets (e.g., groups of 2, 3, or more values)
/// 2. For each group, it selects only the largest (maximum) value
/// 3. The output is smaller than the input (it's reduced by a factor equal to the group size)
/// 
/// For example, with groups of 2 (numPieces = 2):
/// Input: [1, 5, 3, 7]
/// Groups: [1, 5] and [3, 7]
/// Output: [5, 7] (the maximum value from each group)
/// 
/// Maxout is powerful because it can learn to approximate many different activation functions,
/// making it very flexible. However, it requires more parameters in your neural network.
/// </para>
/// </remarks>
public class MaxoutActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The number of pieces (group size) for the Maxout function.
    /// </summary>
    private readonly int _numPieces;

    /// <summary>
    /// Creates a new instance of the Maxout activation function.
    /// </summary>
    /// <param name="numPieces">The number of pieces (group size) to use.</param>
    /// <exception cref="ArgumentException">Thrown when numPieces is less than 2.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The numPieces parameter determines how many input values are grouped together.
    /// For example, if numPieces is 3, then every 3 consecutive values in your input will be grouped,
    /// and only the maximum value from each group will be kept in the output.
    /// 
    /// The minimum value is 2 because with only 1 piece, there would be no "maximum" to select.
    /// </para>
    /// </remarks>
    public MaxoutActivation(int numPieces)
    {
        if (numPieces < 2)
        {
            throw new ArgumentException("Number of pieces must be at least 2.", nameof(numPieces));
        }

        _numPieces = numPieces;
    }

    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>False, as Maxout requires a vector of values to operate.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns false because Maxout needs to compare multiple values
    /// to find the maximum in each group. It can't process just one number at a time.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Maxout activation function to a vector of values.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector with the maximum value from each group.</returns>
    /// <exception cref="ArgumentException">Thrown when the input length is not divisible by the number of pieces.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your input values and:
    /// 1. Divides them into groups of size _numPieces
    /// 2. Finds the maximum value in each group
    /// 3. Returns these maximum values as a new, smaller vector
    /// 
    /// The input length must be divisible by _numPieces so that all values can be properly grouped.
    /// For example, if _numPieces is 3, your input length could be 3, 6, 9, etc., but not 4 or 5.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        if (input.Length % _numPieces != 0)
        {
            throw new ArgumentException("Input vector length must be divisible by the number of pieces.");
        }

        int outputSize = input.Length / _numPieces;
        Vector<T> output = new Vector<T>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            T maxValue = input[i * _numPieces];
            for (int j = 1; j < _numPieces; j++)
            {
                maxValue = MathHelper.Max(maxValue, input[i * _numPieces + j]);
            }

            output[i] = maxValue;
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative (Jacobian matrix) of the Maxout function for a vector input.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A Jacobian matrix representing the derivative of Maxout.</returns>
    /// <exception cref="ArgumentException">Thrown when the input length is not divisible by the number of pieces.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative helps us understand how the output changes when we slightly change the input.
    /// 
    /// For Maxout, the derivative is simple but special:
    /// - For each group, only the maximum value affects the output
    /// - The derivative is 1 for the maximum value in each group
    /// - The derivative is 0 for all other values in the group
    /// 
    /// This creates a "sparse" matrix (mostly filled with zeros) where:
    /// - Each row corresponds to an output value
    /// - Each column corresponds to an input value
    /// - A value of 1 indicates which input was the maximum in its group
    /// 
    /// During neural network training, this helps the network understand which inputs
    /// were most important (the maximum ones) and should be adjusted.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        if (input.Length % _numPieces != 0)
        {
            throw new ArgumentException("Input vector length must be divisible by the number of pieces.");
        }

        int outputSize = input.Length / _numPieces;
        Matrix<T> jacobian = new Matrix<T>(outputSize, input.Length);

        for (int i = 0; i < outputSize; i++)
        {
            int maxIndex = i * _numPieces;
            T maxValue = input[maxIndex];

            for (int j = 1; j < _numPieces; j++)
            {
                int currentIndex = i * _numPieces + j;
                if (NumOps.GreaterThan(input[currentIndex], maxValue))
                {
                    maxIndex = currentIndex;
                    maxValue = input[currentIndex];
                }
            }

            jacobian[i, maxIndex] = NumOps.One;
        }

        return jacobian;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.Maxout provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// Maxout supports JIT compilation with sparse gradient routing.
    /// The backward pass routes gradients only to the maximum element in each group.
    /// </para>
    /// <para>
    /// Note: Currently implemented for 2D tensors (batch, features) where features is divisible by numPieces.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Maxout activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.Maxout(input) which handles both
    /// forward and backward passes for JIT compilation with argmax tracking.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Maxout(input, _numPieces);
    }
}
