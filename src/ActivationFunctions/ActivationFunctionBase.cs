namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Base class for all activation functions used in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Activation functions are mathematical operations that determine the output
/// of a neural network node. They introduce non-linearity into the network, allowing it to
/// learn complex patterns. Think of them as decision-makers that determine how strongly a
/// neuron "fires" based on its inputs.
/// 
/// Common activation functions include:
/// - Sigmoid: Outputs values between 0 and 1 (like probabilities)
/// - ReLU: Returns 0 for negative inputs, or the input value for positive inputs
/// - Tanh: Similar to sigmoid but outputs values between -1 and 1
/// 
/// The "derivative" methods are used during training to determine how to adjust the network's
/// weights to improve its accuracy.
/// </para>
/// </remarks>
public abstract class ActivationFunctionBase<T> : IActivationFunction<T>, IVectorActivationFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Determines if the activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>True if scalar operations are supported; otherwise, false.</returns>
    protected abstract bool SupportsScalarOperations();

    /// <summary>
    /// Applies the activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The activated output value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a single number using the activation function.
    /// The default implementation is the identity function (returns the input unchanged).
    /// Derived classes will override this with specific activation functions like sigmoid or ReLU.
    /// </para>
    /// </remarks>
    public virtual T Activate(T input)
    {
        return input; // Default to identity function
    }

    /// <summary>
    /// Calculates the derivative of the activation function for a single input value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <returns>The derivative value at the input point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative measures how much the activation function's output changes
    /// when its input changes slightly. This is essential for training neural networks through
    /// backpropagation. The default implementation returns 1, meaning the output changes at the
    /// same rate as the input.
    /// </para>
    /// </remarks>
    public virtual T Derivative(T input)
    {
        return NumOps.One; // Default to constant derivative of 1
    }

    /// <summary>
    /// Applies the activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A new vector with the activation function applied to each element.</returns>
    public virtual Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    /// <summary>
    /// Calculates the derivative matrix for a vector input.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A diagonal matrix containing derivatives for each input element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a special matrix where the diagonal contains the derivatives
    /// for each input value. This matrix is used during backpropagation to efficiently calculate
    /// how errors propagate through the network.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> Derivative(Vector<T> input)
    {
        return Matrix<T>.CreateDiagonal(input.Transform(Derivative));
    }

    /// <summary>
    /// Applies the activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with the activation function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A tensor is a multi-dimensional array that can represent complex data
    /// structures like images (3D tensors) or video (4D tensors). This method applies the
    /// activation function to every single value in the tensor.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative for each element in a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor containing derivatives for each input element.</returns>
    public virtual Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }

        return output;
    }
}