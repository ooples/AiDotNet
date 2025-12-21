using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Squash activation function, which normalizes vectors to have a magnitude between 0 and 1.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Squash activation function maps input vectors to have a magnitude (length) between 0 and 1 while
/// preserving their direction. This is particularly useful in capsule networks and other neural network
/// architectures where vector orientation is important.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Squash function is different from most activation functions because it works on
/// entire vectors (groups of numbers) rather than individual numbers. Think of a vector as an arrow
/// pointing in some direction with a certain length.
/// 
/// What Squash does:
/// 1. It keeps the arrow pointing in the same direction
/// 2. It adjusts the length of the arrow to be between 0 and 1
/// 3. Short vectors remain almost the same length
/// 4. Long vectors get "squashed" to be closer to length 1
/// 
/// This is useful in advanced neural networks like capsule networks, where the direction of a vector
/// represents important features, and the length represents the probability that those features exist.
/// </para>
/// </remarks>
public class SquashActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns false as Squash only operates on vectors, not individual values.</returns>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// This method is not supported by SquashActivation as it only operates on vectors.
    /// </summary>
    /// <param name="input">A scalar input value.</param>
    /// <returns>This method never returns as it throws an exception.</returns>
    /// <exception cref="NotSupportedException">Always thrown when this method is called.</exception>
    /// <remarks>
    /// The Squash activation function is designed to work with vectors, not individual scalar values.
    /// Use the <see cref="Activate(Vector{T})"/> method instead.
    /// </remarks>
    public override T Activate(T input)
    {
        throw new NotSupportedException("SquashActivation does not support scalar operations.");
    }

    /// <summary>
    /// This method is not supported by SquashActivation as it only operates on vectors.
    /// </summary>
    /// <param name="input">A scalar input value.</param>
    /// <returns>This method never returns as it throws an exception.</returns>
    /// <exception cref="NotSupportedException">Always thrown when this method is called.</exception>
    /// <remarks>
    /// The derivative of the Squash activation function is only defined for vector inputs.
    /// Use the <see cref="Derivative(Vector{T})"/> method instead.
    /// </remarks>
    public override T Derivative(T input)
    {
        throw new NotSupportedException("SquashActivation does not support scalar operations.");
    }

    /// <summary>
    /// Applies the Squash activation function to a vector input.
    /// </summary>
    /// <param name="input">The input vector to be squashed.</param>
    /// <returns>A new vector with the same direction as the input but with magnitude between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// The Squash function is defined as: v * (||v||² / (1 + ||v||²)) / ||v||
    /// where ||v|| is the Euclidean norm (length) of the vector v.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes a vector (think of it as an arrow with direction and length)
    /// and returns a new vector that:
    /// - Points in the same direction as the original vector
    /// - Has a length between 0 and 1
    /// 
    /// The formula works so that:
    /// - Very short vectors (length close to 0) stay almost the same length
    /// - Medium-length vectors get moderately compressed
    /// - Very long vectors get "squashed" to have a length close to 1
    /// 
    /// This is useful in neural networks where we want to represent both the direction of something
    /// (like a feature) and how confident we are about it (the length).
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        T normSquared = input.DotProduct(input);
        T norm = NumOps.Sqrt(normSquared);
        T scale = NumOps.Divide(normSquared, NumOps.Add(NumOps.One, normSquared));

        return input.Multiply(NumOps.Divide(scale, norm));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Squash function at the given input vector.
    /// </summary>
    /// <param name="input">The input vector at which to calculate the derivative.</param>
    /// <returns>A matrix representing the Jacobian of the Squash function at the input vector.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix contains the partial derivatives of each output component with respect to each input component.
    /// For a vector function, the Jacobian is a matrix where element (i,j) is the derivative of the i-th output
    /// component with respect to the j-th input component.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative (or in this case, the "Jacobian matrix") tells us how the output
    /// of the Squash function changes when we make small changes to the input vector.
    /// 
    /// Think of it this way:
    /// - The Squash function takes a vector and returns another vector
    /// - The Jacobian matrix tells us, for each element in the output vector, how it would change
    ///   if we changed each element in the input vector
    /// - This matrix is essential during neural network training to determine how to adjust the weights
    /// 
    /// This is a more complex concept than regular derivatives because we're dealing with vectors
    /// (multiple numbers) rather than single numbers. The matrix structure helps us track how each
    /// input value affects each output value.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        T normSquared = input.DotProduct(input);
        T norm = NumOps.Sqrt(normSquared);
        T scale = NumOps.Divide(normSquared, NumOps.Add(NumOps.One, normSquared));

        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    T term1 = NumOps.Divide(scale, norm);
                    T term2 = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(input[i], input[i]));
                    T term3 = NumOps.Multiply(NumOps.Add(NumOps.One, normSquared), norm);
                    term2 = NumOps.Divide(term2, term3);
                    jacobian[i, j] = NumOps.Subtract(term1, term2);
                }
                else
                {
                    T term = NumOps.Multiply(input[i], input[j]);
                    term = NumOps.Multiply(NumOps.FromDouble(2), term);
                    term = NumOps.Divide(term, NumOps.Multiply(NumOps.Add(NumOps.One, normSquared), norm));
                    term = NumOps.Multiply(scale, term);
                    jacobian[i, j] = term;
                }
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the Squash activation function to a batch of vectors stored in a tensor.
    /// </summary>
    /// <param name="input">The input tensor containing a batch of vectors to be squashed.</param>
    /// <returns>A new tensor with the same shape as the input, where each vector has been squashed.</returns>
    /// <remarks>
    /// <para>
    /// This method processes a batch of vectors by applying the Squash function to each vector independently.
    /// The input tensor is expected to have shape [batchSize, vectorLength].
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A tensor in this context is essentially a collection of vectors (like a batch of examples).
    /// This method:
    /// 1. Takes each vector from the collection
    /// 2. Applies the Squash function to it (as described in the Activate method)
    /// 3. Puts the result back in the same position in a new collection
    /// 
    /// This allows us to process many examples at once, which is more efficient than processing them one by one.
    /// </para>
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        int batchSize = input.Shape[0];
        int vectorLength = input.Shape[1];

        for (int i = 0; i < batchSize; i++)
        {
            Vector<T> vector = new Vector<T>(vectorLength);
            for (int j = 0; j < vectorLength; j++)
            {
                vector[j] = input[i, j];
            }

            Vector<T> activatedVector = Activate(vector);

            for (int j = 0; j < vectorLength; j++)
            {
                output[i, j] = activatedVector[j];
            }
        }

        return output;
    }

    /// <summary>
    /// Calculates the Jacobian matrices for a batch of vectors stored in a tensor.
    /// </summary>
    /// <param name="input">The input tensor containing a batch of vectors.</param>
    /// <returns>A tensor containing the Jacobian matrices for each vector in the batch.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the derivative (Jacobian matrix) of the Squash function for each vector in the input tensor.
    /// The input tensor is expected to have shape [batchSize, vectorLength], and the output tensor will have shape
    /// [batchSize, vectorLength, vectorLength], where each [i, :, :] slice is the Jacobian matrix for the i-th input vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method helps us understand how the Squash function would change if we made small changes to our inputs.
    /// 
    /// Think of it this way:
    /// - We have a batch (collection) of vectors
    /// - For each vector, we need to know how changing any element would affect all elements in the output
    /// - This method calculates a special matrix (called a "Jacobian") for each vector in our batch
    /// - Each Jacobian matrix shows how every output element responds to changes in every input element
    /// 
    /// The result is a 3D tensor (think of it as a stack of matrices):
    /// - The first dimension is the batch (which example we're looking at)
    /// - The second and third dimensions form the Jacobian matrix for that example
    /// 
    /// This information is crucial during neural network training as it helps determine how to adjust weights
    /// to minimize errors.
    /// </para>
    /// </remarks>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new([.. input.Shape, input.Shape.Last()]);
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
    /// <value>True because TensorOperations.Squash provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// Squash supports JIT compilation with gradient computation for capsule networks.
    /// The squash function normalizes vectors: v * (||v||² / (1 + ||v||²)) / ||v||.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Squash activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.Squash(input) which handles both
    /// forward and backward passes for JIT compilation in capsule networks.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Squash(input);
    }
}
