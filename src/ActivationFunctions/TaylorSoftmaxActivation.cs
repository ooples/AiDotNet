using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationTask(ActivationTask.NormalizationOutput)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.High)]
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
        if (input.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Same Taylor polynomial and normalization as the tensor overload
        return Activate(Tensor<T>.FromVector(input)).ToVector();
    }

    /// <summary>
    /// Applies TaylorSoftmax to a tensor via engine primitives so the gradient tape
    /// records every step. Builds the Taylor polynomial
    /// <c>1 + x + x^2/2! + ... + x^n/n!</c> iteratively using TensorAdd / TensorMultiply /
    /// TensorMultiplyScalar, then normalizes by the reduced sum along the last axis.
    /// </summary>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        int lastAxis = input.Shape.Length - 1;

        var result = TaylorPolynomial(input, _order);
        var sum = Engine.ReduceSum(result, new[] { lastAxis }, keepDims: true);
        return Engine.TensorDivide(result, sum);
    }

    /// <summary>
    /// The Taylor polynomial of the exponential, 1 + x + x^2/2! + ... + x^order/order!, element by element as
    /// whole-tensor operations: each power is the previous one times x, scaled by 1/n!.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private Matrix<T> ScalarJacobian(Vector<T> input)
    {
        int n = input.Length;
        var x = input.AsSpan();
        var value = new T[n];
        var slope = new T[n];
        T z = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            // f = sum_{k<=order} x^k / k!, f' = sum_{k<order} x^k / k!
            T term = NumOps.One;
            T f = NumOps.One;
            T df = NumOps.Zero;
            for (int k = 1; k <= _order; k++)
            {
                df = NumOps.Add(df, term);
                term = NumOps.Divide(NumOps.Multiply(term, x[i]), NumOps.FromDouble(k));
                f = NumOps.Add(f, term);
            }

            value[i] = f;
            slope[i] = df;
            z = NumOps.Add(z, f);
        }

        var jacobian = new Matrix<T>(n, n);
        var cells = jacobian.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            T y = NumOps.Divide(value[i], z);
            int row = i * n;
            for (int j = 0; j < n; j++)
            {
                cells[row + j] = NumOps.Negate(NumOps.Multiply(y, NumOps.Divide(slope[j], z)));
            }

            T g = NumOps.Divide(slope[i], z);
            cells[row + i] = NumOps.Subtract(g, NumOps.Multiply(y, g));
        }

        return jacobian;
    }

    private Tensor<T> TaylorPolynomial(Tensor<T> input, int order)
    {
        var result = order >= 1
            ? Engine.TensorAddScalar(input, NumOps.One)               // 1 + x
            : Engine.TensorOnesLike(input);                           // 1
        if (order < 2)
        {
            return result;
        }

        // 1/2!, 1/3!, ... in the numeric type, converted once
        var reciprocals = new double[order - 1];
        double factorial = 1.0;
        for (int n = 2; n <= order; n++)
        {
            factorial *= n;
            reciprocals[n - 2] = 1.0 / factorial;
        }

        var scales = new Tensor<double>(reciprocals, new[] { reciprocals.Length }).Cast<T>().AsSpan();
        Tensor<T> power = input;
        for (int n = 2; n <= order; n++)
        {
            power = Engine.TensorMultiply(power, input);              // x^n
            result = Engine.TensorAdd(result, Engine.TensorMultiplyScalar(power, scales[n - 2]));
        }

        return result;
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
        if (input.Length == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // y[i] = f(x[i]) / Z with f the order-n polynomial and Z = sum f(x[k]), so
        // dy[i]/dx[j] = (delta_ij - y[i]) * g[j] with g = f'(x) / Z and f' the order-(n-1) polynomial:
        // J = diag(g) - y g^T. (The softmax form y[i] * (delta_ij - y[j]) needs f' = f, true only for exp.)
        if (UseScalarJacobian(input.Length))
        {
            return ScalarJacobian(input);
        }

        var engine = Engine;
        var x = Tensor<T>.FromVector(input);
        var numerator = TaylorPolynomial(x, _order);
        var slope = _order >= 1 ? TaylorPolynomial(x, _order - 1) : engine.TensorZerosLike(x);
        T z = engine.TensorSum(numerator);
        var y = engine.TensorDivideScalar(numerator, z);
        var g = engine.TensorDivideScalar(slope, z);
        return OuterWithDiagonal(engine.TensorNegate(y), g, engine.TensorSubtract(g, engine.TensorMultiply(y, g)));
    }

}
