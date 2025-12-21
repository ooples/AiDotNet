using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Sparsemax activation function, which is an alternative to Softmax that can produce sparse probability distributions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sparsemax maps input vectors to probability distributions, similar to Softmax, but with the key difference
/// that Sparsemax can assign exactly zero probability to low-scoring classes, creating sparse outputs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sparsemax is an advanced activation function used primarily in the output layer of 
/// neural networks for classification tasks. While Softmax always gives some probability to every possible 
/// class (even if very small), Sparsemax can assign exactly zero probability to unlikely classes.
/// 
/// Think of it like this:
/// - Softmax is like saying "I'm 80% sure it's a dog, 15% sure it's a cat, and 5% sure it's something else"
/// - Sparsemax might say "I'm 90% sure it's a dog, 10% sure it's a cat, and 0% sure it's anything else"
/// 
/// This "sparsity" (having many zeros) can be useful when you have many possible classes but only a few
/// are likely to be correct. It makes the model's predictions more focused and interpretable.
/// </para>
/// </remarks>
public class SparsemaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns false as Sparsemax requires a vector of values.</returns>
    /// <remarks>
    /// <para>
    /// Unlike functions like ReLU that can operate on individual values, Sparsemax needs to consider
    /// all elements in a vector together to compute the probability distribution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method returning false means that Sparsemax cannot work on just one number at a time.
    /// It needs to see all the values together (like all scores for different classes) to determine
    /// which ones should get non-zero probabilities.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Sparsemax activation function to a vector of inputs.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector where some elements may be exactly zero, and all elements sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// Sparsemax projects the input vector onto the probability simplex, potentially setting some values to exactly zero.
    /// The algorithm finds a threshold value and sets all elements below this threshold to zero,
    /// while shifting the remaining elements to ensure they sum to 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a vector of numbers (like scores for different classes) into 
    /// probabilities that add up to 1. Here's how it works:
    /// 
    /// 1. Sort all the input values from highest to lowest
    /// 2. Find a "threshold" value by checking when the running average becomes greater than the next value
    /// 3. Subtract this threshold from all input values
    /// 4. Set any negative results to zero
    /// 
    /// The result is a probability distribution where:
    /// - Some values may be exactly zero (unlike Softmax where all values are positive)
    /// - All values add up to 1
    /// - Higher input values get higher probabilities
    /// 
    /// This creates "sparse" outputs where only the most important classes get non-zero probabilities,
    /// making the model's predictions easier to interpret.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        int k = 1;
        int d = input.Length;
        var z = input.OrderByDescending(x => x).ToArray();
        T sum = NumOps.Zero;
        T threshold = NumOps.Zero;

        for (int i = 0; i < d; i++)
        {
            sum = NumOps.Add(sum, z[i]);
            T average = NumOps.Divide(sum, NumOps.FromDouble(i + 1));
            if (NumOps.GreaterThan(average, z[i]))
            {
                k = i;
                threshold = average;
                break;
            }
        }

        if (k == d)
        {
            threshold = NumOps.Divide(sum, NumOps.FromDouble(d));
        }

        return input.Transform(x => MathHelper.Max(NumOps.Zero, NumOps.Subtract(x, threshold)));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Sparsemax function for a given input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A matrix representing the partial derivatives of each output with respect to each input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix contains the partial derivatives of each output element with respect to each input element.
    /// For Sparsemax, the Jacobian has a specific structure based on which outputs are non-zero.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative of Sparsemax is more complex than simpler activation functions because
    /// changing one input can affect multiple outputs. This method returns a "Jacobian matrix" which shows
    /// how each output value changes when each input value changes.
    /// 
    /// Some key points about the Sparsemax derivative:
    /// - For outputs that are zero, the derivatives are also zero (these outputs don't respond to small input changes)
    /// - For non-zero outputs, the derivatives form a specific pattern:
    ///   - When i=j (diagonal elements): the derivative is 1
    ///   - When i?j: the derivative is negative and depends on the output values
    /// 
    /// During neural network training, this matrix helps determine how to adjust the weights based on errors.
    /// The sparsity of Sparsemax can make training more efficient because many derivatives will be zero.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        var output = Activate(input);
        int d = input.Length;
        var jacobian = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (NumOps.GreaterThan(output[i], NumOps.Zero))
                {
                    if (i == j)
                    {
                        jacobian[i, j] = NumOps.One;
                    }
                    else
                    {
                        jacobian[i, j] = NumOps.Negate(NumOps.Divide(output[j], output[i]));
                    }
                }
                else
                {
                    jacobian[i, j] = NumOps.Zero;
                }
            }
        }

        return jacobian;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.Sparsemax provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// Sparsemax supports JIT compilation with support set tracking for correct gradient computation.
    /// The backward pass routes gradients only through the support set (non-zero outputs),
    /// computing the mean gradient within the support and subtracting it from each element.
    /// </para>
    /// <para>
    /// Note: Currently implemented for 2D tensors (batch, features) along axis=-1.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with Sparsemax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.Sparsemax(input) which handles both
    /// forward and backward passes for JIT compilation with support set tracking.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.Sparsemax(input);
    }
}
