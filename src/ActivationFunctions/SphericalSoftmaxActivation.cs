using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationTask(ActivationTask.NormalizationOutput)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.High)]
public class SphericalSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Numerical-stability epsilon added to the squared-norm before the sqrt to avoid
    /// division-by-zero when the input vector is exactly zero. The value 1e-12 is well
    /// below single-precision rounding error and is the standard guard used in
    /// L2-normalization layers (e.g. PyTorch's <c>F.normalize(eps=1e-12)</c>).
    /// </summary>
    private static readonly T NormalizationEpsilon = MathHelper.GetNumericOperations<T>().FromDouble(1e-12);

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
    /// The norm is <c>sqrt(sum(x^2) + 1e-12)</c>, the same stabilised norm the tensor overload uses, so the zero
    /// vector gives the uniform distribution instead of 0 / 0. It is computed directly rather than through the engine,
    /// so it agrees with the tensor overload to rounding, not bit for bit.
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
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (input.Length == 0)
            return new Vector<T>(0);

        return ActivateDirect(input);
    }

    /// <summary>
    /// The vector overload, computed directly: the same stabilised norm and softmax as the tensor overload, without
    /// a rank-1 tensor through the engine. A vector activation is called once per sample, where the engine's fixed
    /// per-call cost (2-6 us on the CPU engine, more on a GPU engine) is many times the arithmetic.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("Per-sample vector activation: one pass for the norm, one for the exponentials and "
        + "one to normalize, over a vector usually too short for an engine dispatch to pay off.")]
    private static Vector<T> ActivateDirect(Vector<T> input)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var x = input.AsSpan();
        T sumSquared = ops.Zero;
        for (int i = 0; i < x.Length; i++)
            sumSquared = ops.Add(sumSquared, ops.Multiply(x[i], x[i]));
        T norm = ops.Sqrt(ops.Add(sumSquared, NormalizationEpsilon));

        // Shift by the largest normalized value before exponentiating, as a softmax does; mathematically a no-op.
        T max = ops.Divide(x[0], norm);
        for (int i = 1; i < x.Length; i++)
        {
            T scaled = ops.Divide(x[i], norm);
            if (ops.GreaterThan(scaled, max)) max = scaled;
        }

        var result = new Vector<T>(x.Length);
        var r = result.AsWritableSpan();
        T sum = ops.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            r[i] = ops.Exp(ops.Subtract(ops.Divide(x[i], norm), max));
            sum = ops.Add(sum, r[i]);
        }

        for (int i = 0; i < r.Length; i++)
            r[i] = ops.Divide(r[i], sum);
        return result;
    }

    /// <summary>
    /// Applies SphericalSoftmax to a tensor via engine primitives so the gradient tape
    /// records every step. Decomposes as <c>softmax(x / ||x||_2)</c> along the last axis.
    /// </summary>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (input.Shape.Length == 0)
            throw new ArgumentException("SphericalSoftmax requires at least one tensor dimension.", nameof(input));

        int lastAxis = input.Shape.Length - 1;
        var squared = Engine.TensorMultiply(input, input);
        var sumSquared = Engine.ReduceSum(squared, new[] { lastAxis }, keepDims: true);
        var eps = Tensor<T>.CreateDefault(sumSquared._shape, NormalizationEpsilon);
        var safeSum = Engine.TensorAdd(sumSquared, eps);
        var norm = Engine.TensorSqrt(safeSum);
        // The [..., 1] norm is broadcast to the input's shape explicitly: an element-wise divide of two different
        // shapes is not something every engine shape policy accepts.
        var normalized = Engine.TensorDivide(input, Engine.TensorBroadcastTo(norm, input.Shape.ToArray()));
        return Engine.Softmax(normalized, axis: lastAxis);
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
    /// The chain rule multiplies the two: with u = x / ||x|| and y = softmax(u),
    /// J = (diag(y) - y y^T) * (I - u u^T) / ||x||.
    /// The first factor is the ordinary softmax Jacobian; the second is the Jacobian of the normalization. With the
    /// exact norm it removes the part of any input change that points along x, because scaling x does not change
    /// the output. With the stabilising epsilon in the norm, sqrt(||x||^2 + eps), the output is scale invariant
    /// only approximately: a change along x is attenuated by eps / (||x||^2 + eps), not removed.
    /// 
    /// Don't worry if the math seems complex - the neural network library handles these calculations automatically.
    /// The important thing to understand is that this function helps the network learn by showing how changes in
    /// the input affect the output probabilities.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        if (UseScalarJacobian(n))
        {
            return ScalarJacobian(input, Activate(input));
        }

        var engine = Engine;
        var values = Tensor<T>.FromVector(input);
        var p = Activate(values);

        // The same stabilised norm as the forward pass: sqrt(sum(x^2) + epsilon)
        T norm = NumOps.Sqrt(NumOps.Add(engine.TensorSum(engine.TensorMultiply(values, values)), NormalizationEpsilon));
        var unit = engine.TensorDivideScalar(values, norm);

        // J = S (I - u u^T) / norm with S = diag(p) - p p^T the softmax Jacobian at u = x / norm.
        // S u = p * u - p (p . u), so J = (diag(p) - p p^T - (S u) u^T) / norm.
        var pu = engine.TensorMultiply(p, unit);
        var su = engine.TensorSubtract(pu, engine.TensorMultiplyScalar(p, engine.TensorSum(pu)));
        var outers = engine.TensorAdd(engine.TensorOuter(p, p), engine.TensorOuter(su, unit));
        var diagonal = engine.TensorSubtract(
            p, engine.TensorAdd(engine.TensorMultiply(p, p), engine.TensorMultiply(su, unit)));
        return SquareWithDiagonal(
            engine.TensorDivideScalar(engine.TensorNegate(outers), norm), engine.TensorDivideScalar(diagonal, norm));
    }

    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private Matrix<T> ScalarJacobian(Vector<T> input, Vector<T> output)
    {
        int n = input.Length;
        var x = input.AsSpan();
        var p = output.AsSpan();
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(x[i], x[i]));
        }

        // The same stabilised norm as the forward pass, and the same J = (diag(p) - p p^T - (S u) u^T) / norm
        T norm = NumOps.Sqrt(NumOps.Add(sumSquares, NormalizationEpsilon));
        var unit = new T[n];
        T dot = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            unit[i] = NumOps.Divide(x[i], norm);
            dot = NumOps.Add(dot, NumOps.Multiply(p[i], unit[i]));
        }

        var jacobian = new Matrix<T>(n, n);
        var cells = jacobian.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            T su = NumOps.Subtract(NumOps.Multiply(p[i], unit[i]), NumOps.Multiply(p[i], dot));
            int row = i * n;
            for (int j = 0; j < n; j++)
            {
                T outers = NumOps.Add(NumOps.Multiply(p[i], p[j]), NumOps.Multiply(su, unit[j]));
                cells[row + j] = NumOps.Divide(NumOps.Negate(outers), norm);
            }

            T diagonal = NumOps.Subtract(
                p[i], NumOps.Add(NumOps.Multiply(p[i], p[i]), NumOps.Multiply(su, unit[i])));
            cells[row + i] = NumOps.Divide(diagonal, norm);
        }

        return jacobian;
    }

}
