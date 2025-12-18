using AiDotNet.Autodiff;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Taylor Softmax activation function, which is a computationally efficient approximation of the standard Softmax function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Taylor Softmax function is a variation of the standard Softmax function that uses a mathematical
/// technique called "Taylor series expansion" to approximate the exponential function. This makes it computationally
/// more efficient while maintaining similar behavior to the standard Softmax.
/// 
/// Softmax functions are commonly used in the output layer of neural networks for multi-class classification problems.
/// They convert a vector of numbers into a probability distribution (all values are positive and sum to 1).
/// 
/// For example, if you have three output neurons with values [2.0, 1.0, 0.5], the Softmax function will convert
/// these to probabilities like [0.6, 0.25, 0.15], which sum to 1.0. This makes it easy to interpret the outputs
/// as probabilities for each class.
/// 
/// The "Taylor" part refers to using a mathematical approximation (Taylor series) instead of calculating the
/// full exponential function, which can be faster but slightly less accurate.
/// </para>
/// </remarks>
public class TaylorSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// The order of the Taylor series approximation used for the exponential function.
    /// </summary>
    /// <remarks>
    /// Higher values provide more accurate approximations but require more computation.
    /// </remarks>
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the TaylorSoftmaxActivation class with the specified order of approximation.
    /// </summary>
    /// <param name="order">The order of the Taylor series approximation. Default is 2.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The "order" parameter determines how accurate the approximation will be.
    /// 
    /// Think of it like drawing a curve:
    /// - Order 1: You're approximating with a straight line (very rough)
    /// - Order 2: You're using a curved line (better approximation)
    /// - Order 3 and higher: You're using increasingly complex curves (more accurate)
    /// 
    /// Higher orders give more accurate results but require more computation time.
    /// The default value of 2 provides a good balance between accuracy and speed for most applications.
    /// </para>
    /// </remarks>
    public TaylorSoftmaxActivation(int order = 2)
    {
        _order = order;
    }

    /// <summary>
    /// Indicates whether this activation function supports operations on individual scalar values.
    /// </summary>
    /// <returns>Always returns false as Taylor Softmax only operates on vectors.</returns>
    /// <remarks>
    /// Softmax functions require a vector of values to create a probability distribution, so they cannot
    /// be applied to single scalar values.
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Taylor Softmax activation function to a vector of input values.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A vector of probabilities that sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a vector of any numbers into a vector of probabilities
    /// (positive numbers that sum to 1). It does this in three steps:
    /// 
    /// 1. It approximates e^x for each input value using the Taylor series approximation
    /// 2. It sums up all these approximated values
    /// 3. It divides each approximated value by the sum to get probabilities
    /// 
    /// The result is a vector where:
    /// - All values are positive
    /// - All values sum to exactly 1.0
    /// - Larger input values correspond to larger output probabilities
    /// 
    /// This is commonly used in the final layer of classification neural networks, where each
    /// output represents the probability of the input belonging to a particular class.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> expValues = input.Transform(x => TaylorExp(x, _order));
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    /// <summary>
    /// Calculates the Jacobian matrix of partial derivatives for the Taylor Softmax function.
    /// </summary>
    /// <param name="input">The vector of input values.</param>
    /// <returns>A matrix of partial derivatives.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The derivative of the Softmax function is more complex than most activation functions
    /// because changing one input affects all outputs. This method calculates how each output probability
    /// changes with respect to each input value.
    /// 
    /// The result is a matrix (a grid of numbers) where:
    /// - Each row corresponds to an output
    /// - Each column corresponds to an input
    /// - Each value shows how much that particular output changes when that particular input changes
    /// 
    /// This matrix is used during backpropagation to update the weights in the neural network.
    /// 
    /// The diagonal elements (where i=j) represent how an output is affected by its corresponding input.
    /// The off-diagonal elements (where i?j) represent how an output is affected by other inputs.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> output = Activate(input);
        int d = input.Length;
        Matrix<T> jacobian = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(output[i], NumOps.Subtract(NumOps.One, output[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Multiply(NumOps.Negate(output[i]), output[j]);
                }
            }
        }

        return jacobian;
    }

    /// <summary>
    /// Approximates the exponential function e^x using a Taylor series expansion.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <param name="order">The order of the Taylor series approximation.</param>
    /// <returns>An approximation of e^x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method approximates the exponential function (e^x) using a mathematical
    /// technique called a Taylor series. Instead of calculating the exact value of e^x, which can be
    /// computationally expensive, it uses a sum of simpler terms to get close to the right answer.
    /// 
    /// The formula used is: e^x ˜ 1 + x + x²/2! + x³/3! + ... + xn/n!
    /// 
    /// Where:
    /// - x is the input value
    /// - n is the order of approximation
    /// - n! (factorial) means n × (n-1) × (n-2) × ... × 1
    /// 
    /// Higher orders give more accurate results but require more computation.
    /// </para>
    /// </remarks>
    private T TaylorExp(T x, int order)
    {
        T result = NumOps.One;
        T term = NumOps.One;

        for (int n = 1; n <= order; n++)
        {
            term = NumOps.Divide(NumOps.Multiply(term, x), NumOps.FromDouble(n));
            result = NumOps.Add(result, term);
        }

        return result;
    }


    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    /// <value>True because TensorOperations.TaylorSoftmax provides full forward and backward pass support.</value>
    /// <remarks>
    /// <para>
    /// TaylorSoftmax supports JIT compilation using polynomial Taylor series expansion.
    /// The backward pass computes gradients through the polynomial approximation of exp.
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
    /// <returns>A new computation node with TaylorSoftmax activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <remarks>
    /// <para>
    /// This method maps to TensorOperations&lt;T&gt;.TaylorSoftmax(input) which handles both
    /// forward and backward passes for JIT compilation using Taylor series polynomial.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return TensorOperations<T>.TaylorSoftmax(input, _order);
    }
}
