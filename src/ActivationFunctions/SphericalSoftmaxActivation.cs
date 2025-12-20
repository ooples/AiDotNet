using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Spherical Softmax activation function, which normalizes inputs to the unit sphere before applying softmax.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Spherical Softmax is a variation of the standard Softmax function that first normalizes the input vector
/// to have unit length (projects it onto the unit sphere) before applying the exponential and normalization steps.
/// This can help improve numerical stability and performance in certain neural network architectures.
/// </para>
/// <para>
/// <b>For Beginners:</b> Spherical Softmax is a special version of the Softmax function that's often used in the
/// output layer of neural networks for classification tasks. 
/// 
/// The standard Softmax converts a vector of numbers into probabilities that sum to 1, but it can sometimes
/// have numerical issues with very large or very small numbers. Spherical Softmax adds an extra step by first
/// "normalizing" the input vector (making its length equal to 1) before applying the regular Softmax steps.
/// 
/// Think of it like this:
/// 1. First, we adjust all the input values so that they form a point on a sphere with radius 1
/// 2. Then we apply the regular Softmax calculation to these adjusted values
/// 
/// This approach can make the neural network training more stable and sometimes leads to better performance,
/// especially when dealing with high-dimensional data or when the input values can vary widely in magnitude.
/// </para>
/// </remarks>
public class SphericalSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns false as Spherical Softmax requires a vector of values.</returns>
    /// <remarks>
    /// <para>
    /// Unlike functions like ReLU that can operate on individual values, Spherical Softmax needs to consider
    /// all elements in a vector together to compute the probability distribution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method returning false means that Spherical Softmax cannot work on just one number at a time.
    /// It needs to see all the values together (like all scores for different classes) to normalize them properly
    /// and convert them into probabilities.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Spherical Softmax activation function to a vector of inputs.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector where all elements are positive and sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// The Spherical Softmax function performs the following steps:
    /// 1. Normalizes the input vector to have unit length (L2 norm = 1)
    /// 2. Applies the exponential function to each element
    /// 3. Divides each element by the sum of all exponential values
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a vector of numbers (like scores for different classes) into 
    /// probabilities that add up to 1. Here's how it works step by step:
    /// 
    /// 1. First, it calculates the "length" of the input vector (called the L2 norm)
    ///    - This is like finding the distance from the origin to a point in multi-dimensional space
    ///    - Mathematically, it's the square root of the sum of squares of all elements
    /// 
    /// 2. Then it divides each input value by this length
    ///    - This "normalizes" the vector so its length becomes exactly 1
    ///    - The direction of the vector stays the same, but its magnitude is now 1
    /// 
    /// 3. Next, it applies the exponential function (e^x) to each normalized value
    ///    - This makes all values positive and emphasizes larger values
    /// 
    /// 4. Finally, it divides each exponential value by the sum of all exponential values
    ///    - This ensures all values are between 0 and 1 and sum to exactly 1
    ///    - These values can now be interpreted as probabilities
    /// 
    /// The result is a probability distribution where higher input values get higher probabilities,
    /// but the initial normalization step helps prevent numerical issues that can occur with regular Softmax.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        // Compute the L2 norm of the input vector
        T norm = NumOps.Sqrt(input.Transform(x => NumOps.Multiply(x, x)).Sum());

        // Normalize the input vector
        Vector<T> normalizedInput = input.Transform(x => NumOps.Divide(x, norm));

        // Apply exponential function to each element
        Vector<T> expValues = normalizedInput.Transform(NumOps.Exp);

        // Compute the sum of exponential values
        T sum = expValues.Sum();

        // Divide each exponential value by the sum
        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Spherical Softmax function for a given input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A matrix representing the partial derivatives of each output with respect to each input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix contains the partial derivatives of each output element with respect to each input element.
    /// For Spherical Softmax, the Jacobian combines the derivatives of both the normalization step and the softmax step.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative of Spherical Softmax is quite complex because it involves two transformations:
    /// first normalizing the vector to the unit sphere, and then applying softmax.
    /// 
    /// This method returns a "Jacobian matrix" which shows how each output value changes when each input value changes.
    /// It's used during the training process of neural networks to determine how to adjust the weights.
    /// 
    /// The calculation has two main parts:
    /// 1. How the softmax output changes with respect to the normalized input
    /// 2. How the normalized input changes with respect to the original input
    /// 
    /// For diagonal elements (when i=j):
    /// - term1: How the softmax output changes if its own input increases
    /// - term2: How the normalization affects this relationship
    /// 
    /// For off-diagonal elements (when i?j):
    /// - term1: How one output changes when a different input increases
    /// - term2: How the normalization creates interdependencies between inputs
    /// 
    /// Don't worry if the math seems complex - the neural network library handles these calculations automatically.
    /// The important thing to understand is that this function helps the network learn by showing how changes in
    /// the input affect the output probabilities.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> output = Activate(input);
        int d = input.Length;
        Matrix<T> jacobian = new Matrix<T>(d, d);

        T norm = NumOps.Sqrt(input.Transform(x => NumOps.Multiply(x, x)).Sum());

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j)
                {
                    T term1 = NumOps.Multiply(output[i], NumOps.Subtract(NumOps.One, output[i]));
                    T term2 = NumOps.Divide(NumOps.Multiply(input[i], input[i]), NumOps.Multiply(norm, norm));
                    jacobian[i, j] = NumOps.Divide(NumOps.Subtract(term1, term2), norm);
                }
                else
                {
                    T term1 = NumOps.Multiply(NumOps.Negate(output[i]), output[j]);
                    T term2 = NumOps.Divide(NumOps.Multiply(input[i], input[j]), NumOps.Multiply(norm, norm));
                    jacobian[i, j] = NumOps.Divide(NumOps.Subtract(term1, term2), norm);
                }
            }
        }

        return jacobian;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.SphericalSoftmax provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// SphericalSoftmax supports JIT compilation by composing L2 normalization with softmax.
    /// The backward pass correctly applies the chain rule through both operations.
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
    /// <returns>A new computation node with SphericalSoftmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.SphericalSoftmax(input) which handles both
    /// forward and backward passes for JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.SphericalSoftmax(input);
    }
}
