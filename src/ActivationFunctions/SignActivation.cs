using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Sign activation function, which returns -1 for negative inputs, 1 for positive inputs, and 0 for zero.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Sign function is a simple non-linear activation function that categorizes inputs into three distinct outputs:
/// -1 (negative), 0 (zero), or 1 (positive). Unlike smooth activation functions, Sign has sharp transitions.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Sign function is like a simple categorizer that looks at a number and tells you if it's 
/// negative (-1), zero (0), or positive (1). It's one of the simplest activation functions and is useful when 
/// you want your neural network to make clear-cut decisions rather than produce probabilities or continuous values.
/// However, because it has sharp "jumps" in its output and its derivative is zero almost everywhere, it's rarely 
/// used in modern neural networks that rely on gradient-based learning.
/// </para>
/// </remarks>
public class SignActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns true as Sign supports scalar operations.</returns>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the Sign activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>-1 if input is negative, 1 if input is positive, 0 if input is zero.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method looks at a single number and returns:
    /// - -1 if the number is negative (less than zero)
    /// - 1 if the number is positive (greater than zero)
    /// - 0 if the number is exactly zero
    /// 
    /// It's like a simple judge that categorizes numbers into three groups based on their sign.
    /// </remarks>
    public override T Activate(T input)
    {
        if (NumOps.LessThan(input, NumOps.Zero))
            return NumOps.FromDouble(-1);
        else if (NumOps.GreaterThan(input, NumOps.Zero))
            return NumOps.One;
        else
            return NumOps.Zero;
    }

    /// <summary>
    /// Calculates the derivative of the Sign function for a single input value.
    /// </summary>
    /// <param name="input">The input value to calculate the derivative for.</param>
    /// <returns>Always returns 0, as the Sign function's derivative is 0 everywhere except at 0, where it's undefined.</returns>
    /// <remarks>
    /// <para>
    /// Mathematically, the derivative of the Sign function is 0 everywhere except at x=0, 
    /// where it's undefined (technically, it's the Dirac delta function). For practical purposes,
    /// this implementation returns 0 for all inputs.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative tells us how much the output changes when we slightly change the input.
    /// For the Sign function, the output doesn't change at all when you make small changes to the input
    /// (except exactly at zero). This is why we return 0 for all inputs. This property makes the Sign function
    /// problematic for gradient-based learning algorithms like backpropagation, as there's no useful gradient
    /// information to guide the learning process.
    /// </para>
    /// </remarks>
    public override T Derivative(T input)
    {
        // The derivative of the sign function is 0 everywhere except at 0,
        // where it's undefined. We'll return 0 for all inputs.
        return NumOps.Zero;
    }

    /// <summary>
    /// Applies the Sign activation function to each element in a vector.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A new vector with the Sign function applied to each element.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method applies the Sign function to a whole list of numbers at once.
    /// It processes each number in the list individually using the same Sign function and returns
    /// a new list with all the transformed values (-1, 0, or 1 for each element).
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
    /// Calculates the Jacobian matrix of the Sign function for a vector input.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A matrix of zeros, as the Sign function's derivative is 0 almost everywhere.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix represents how each output element changes with respect to each input element.
    /// For the Sign function, this matrix contains all zeros since the derivative is 0 almost everywhere.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method creates a matrix (a grid of numbers) that represents how sensitive each 
    /// output is to changes in the input. For the Sign function, this matrix is filled with zeros because 
    /// small changes to the input almost never change the output. This is another reason why the Sign function 
    /// isn't commonly used in neural networks that learn through backpropagation - it doesn't provide useful 
    /// information about how to adjust the network's weights.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);
        // The Jacobian matrix will be all zeros
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                jacobian[i, j] = NumOps.Zero;
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the Sign activation function to each element in a tensor.
    /// </summary>
    /// <param name="input">The tensor of input values.</param>
    /// <returns>A new tensor with the Sign function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This method processes each element in the tensor individually using the scalar Activate method.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A tensor is like a multi-dimensional array or a container for data with multiple dimensions.
    /// This method applies the Sign function to every single number in that container, regardless of its position
    /// or how many dimensions the container has. Each number is independently converted to -1, 0, or 1 based on
    /// whether it's negative, zero, or positive.
    /// </para>
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        int totalElements = input.Shape.Aggregate(1, (a, b) => a * b);

        for (int i = 0; i < totalElements; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Calculates the derivative of the Sign function for a tensor input.
    /// </summary>
    /// <param name="input">The tensor of input values.</param>
    /// <returns>A tensor of zeros representing the derivative of the Sign function.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a tensor of derivatives by applying the vector derivative method to each "slice" of the input tensor.
    /// The resulting tensor has an additional dimension that represents the Jacobian matrices for each input vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method handles the derivative calculation for multi-dimensional data (tensors).
    /// It processes the data in batches, where each batch contains multiple vectors. For each vector, it calculates
    /// a matrix of derivatives (which for the Sign function is all zeros). The result is a higher-dimensional tensor
    /// that contains all these matrices. This is a more advanced operation that's used when working with batches of
    /// data in neural networks, particularly in deep learning frameworks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        int[] outputShape = new int[input.Shape.Length + 1];
        Array.Copy(input.Shape, outputShape, input.Shape.Length);
        outputShape[outputShape.Length - 1] = input.Shape[input.Shape.Length - 1];

        Tensor<T> output = new Tensor<T>(outputShape);
        int batchSize = input.Shape[0];
        int vectorLength = input.Shape[1];

        for (int i = 0; i < batchSize; i++)
        {
            Vector<T> vector = new Vector<T>(vectorLength);
            for (int j = 0; j < vectorLength; j++)
            {
                vector[j] = input[i, j];
            }

            Matrix<T> jacobian = Derivative(vector);

            for (int j = 0; j < vectorLength; j++)
            {
                for (int k = 0; k < vectorLength; k++)
                {
                    output[i, j, k] = jacobian[j, k];
                }
            }
        }

        return output;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.Sign provides surrogate gradient support for training.</value>
    /// <remarks>
    /// <para>
    /// Sign supports JIT compilation using surrogate gradients. The forward pass produces
    /// the hard sign function (-1, 0, or 1), while the backward pass uses a sigmoid surrogate
    /// for gradient flow. This enables training despite the discontinuous nature of the sign function.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Sign activation applied using surrogate gradients.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.Sign(input) which uses the
    /// straight-through estimator pattern: hard sign in forward pass, sigmoid surrogate
    /// gradients in backward pass.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Sign(input);
    }
}
