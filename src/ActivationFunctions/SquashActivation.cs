using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ActivationCategory(ActivationCategory.General)]
[ActivationTask(ActivationTask.CapsuleSquash)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = true, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.Medium)]
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
        // Scalar squash: sign(x) * x² / (1 + x²)
        // Equivalently: x * |x| / (1 + x²)
        T absInput = NumOps.Abs(input);
        if (NumOps.Equals(absInput, NumOps.Zero))
        {
            return NumOps.Zero;
        }
        T inputSquared = NumOps.Multiply(input, input);
        T denominator = NumOps.Add(NumOps.One, inputSquared);
        return NumOps.Divide(NumOps.Multiply(input, absInput), denominator);
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
        // Derivative of scalar squash: 2|x| / (1 + x²)²
        T absInput = NumOps.Abs(input);
        T inputSquared = NumOps.Multiply(input, input);
        T onePlusXSq = NumOps.Add(NumOps.One, inputSquared);
        T denominator = NumOps.Multiply(onePlusXSq, onePlusXSq);
        return NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2.0), absInput), denominator);
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
        int n = input.Length;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // One vector: the norm terms are scalars, so the Jacobian is a single outer product plus a multiple of the
        // identity: J = scale * (I - x x^T / |x|^2) / |x| + 2 x x^T / ((1 + |x|^2)^2 |x|).
        bool scalar = UseScalarJacobian(n);
        var engine = Engine;
        var x = scalar ? null : Tensor<T>.FromVector(input);
        T normSquared = x is null ? ScalarSumOfSquares(input) : engine.TensorSum(engine.TensorMultiply(x, x));
        T norm = NumOps.Sqrt(normSquared);
        T normSqPlusOne = NumOps.Add(normSquared, NumOps.One);
        T scale = NumOps.Divide(normSquared, normSqPlusOne);
        T normInv = NumOps.GreaterThan(norm, NumOps.FromDouble(1e-10)) ? NumOps.Divide(NumOps.One, norm) : NumOps.Zero;

        // J = a I + b x x^T with a = scale / |x| and b = 2 / ((1 + |x|^2)^2 |x|) - a / |x|^2
        T a = NumOps.Multiply(scale, normInv);
        T normSqInv = NumOps.GreaterThan(normSquared, NumOps.FromDouble(1e-20))
            ? NumOps.Divide(NumOps.One, normSquared)
            : NumOps.Zero;
        T denominator = NumOps.Multiply(NumOps.Multiply(normSqPlusOne, normSqPlusOne), norm);
        T b = NumOps.Subtract(NumOps.Divide(NumOps.FromDouble(2), denominator), NumOps.Multiply(a, normSqInv));

        if (x is null)
        {
            return ScalarOuterPlusIdentity(input, b, a);
        }

        var scaled = engine.TensorMultiplyScalar(x, b);
        return OuterWithDiagonal(scaled, x, engine.TensorAddScalar(engine.TensorMultiply(scaled, x), a));
    }

    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private T ScalarSumOfSquares(Vector<T> input)
    {
        var x = input.AsSpan();
        T sum = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(x[i], x[i]));
        }

        return sum;
    }

    /// <summary><c>b x x^T + a I</c> in scalar arithmetic.</summary>
    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private Matrix<T> ScalarOuterPlusIdentity(Vector<T> input, T b, T a)
    {
        int n = input.Length;
        var x = input.AsSpan();
        var jacobian = new Matrix<T>(n, n);
        var cells = jacobian.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            T scaled = NumOps.Multiply(x[i], b);
            int row = i * n;
            for (int j = 0; j < n; j++)
            {
                cells[row + j] = NumOps.Multiply(scaled, x[j]);
            }

            cells[row + i] = NumOps.Add(NumOps.Multiply(scaled, x[i]), a);
        }

        return jacobian;
    }

    /// <summary>
    /// The squash Jacobian of every row of a [batch, n] tensor, as a [batch, n, n] tensor.
    /// </summary>
    /// <param name="vectors">The capsule vectors, one per row.</param>
    /// <returns>For each row x, the matrix dy_i/dx_j of y = squash(x).</returns>
    private Tensor<T> SquashJacobians(Tensor<T> vectors)
    {
        var engine = Engine;
        int batch = vectors.Shape[0];
        int n = vectors.Shape[1];
        int[] full = { batch, n, n };

        // Squash: y = a * x/||x|| where a = ||x||^2 / (1 + ||x||^2)
        // Jacobian: dy_i/dx_j = a * (delta_ij - x_i*x_j/||x||^2) / ||x||
        //                     + 2*x_i*x_j / ((1+||x||^2)^2 * ||x||)
        // Per-vector scalars are kept as [batch, 1, 1] and repeated over each [n, n] matrix.
        var normSquared = engine.Reshape(
            engine.ReduceSum(engine.TensorMultiply(vectors, vectors), new[] { 1 }, keepDims: true),
            new[] { batch, 1, 1 });
        var norm = engine.TensorSqrt(normSquared);
        var normSqPlusOne = engine.TensorAddScalar(normSquared, NumOps.One);
        var scale = engine.TensorDivide(normSquared, normSqPlusOne);
        var zeros = engine.TensorZerosLike(normSquared);
        var ones = engine.TensorOnesLike(normSquared);

        // 1 / ||x|| where ||x|| > 1e-10, else 0
        var normInv = engine.TensorWhere(
            engine.TensorGreaterThan(norm, NumOps.FromDouble(1e-10)), engine.TensorDivide(ones, norm), zeros);

        // outer[b, i, j] = x_i * x_j
        var outer = engine.TensorMultiply(
            engine.TensorTile(engine.Reshape(vectors, new[] { batch, n, 1 }), new[] { 1, 1, n }),
            engine.TensorTile(engine.Reshape(vectors, new[] { batch, 1, n }), new[] { 1, n, 1 }));

        // x_i*x_j/||x||^2 where ||x||^2 > 1e-20, else 0
        var normSquaredFull = engine.TensorTile(normSquared, new[] { 1, n, n });
        var outerOverNormSq = engine.TensorWhere(
            engine.TensorGreaterThan(normSquaredFull, NumOps.FromDouble(1e-20)),
            engine.TensorDivide(outer, normSquaredFull),
            engine.TensorZerosLike(outer));

        // Term 1: a * (delta_ij - x_i*x_j/||x||^2) / ||x||
        var identity = engine.TensorTile(engine.Reshape(engine.TensorEye<T>(n), new[] { 1, n, n }), new[] { batch, 1, 1 });
        var normPart = engine.TensorMultiply(
            engine.TensorTile(scale, new[] { 1, n, n }),
            engine.TensorMultiply(
                engine.TensorSubtract(identity, outerOverNormSq), engine.TensorTile(normInv, new[] { 1, n, n })));

        // Term 2: 2*x_i*x_j / ((1+||x||^2)^2 * ||x||)
        var denominator = engine.TensorMultiply(engine.TensorMultiply(normSqPlusOne, normSqPlusOne), norm);
        var daPart = engine.TensorMultiplyScalar(
            engine.TensorDivide(outer, engine.TensorTile(denominator, new[] { 1, n, n })), NumOps.FromDouble(2));

        return engine.Reshape(engine.TensorAdd(normPart, daPart), full);
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
        // Capsule squash: y = v * ||v||^2 / ((1 + ||v||^2) * ||v||). AiDotNet.Tensors
        // ships a native tape-tracked TensorSquash with a matching TensorSquashBackward
        // grad node, so delegate directly instead of hand-decomposing the formula.
        return Engine.TensorSquash(input, axis: input.Shape.Length - 1);
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
        if (input.Shape.Length < 2)
        {
            throw new ArgumentException("Input tensor must have at least 2 dimensions.", nameof(input));
        }

        int vectorLength = input.Shape[input.Shape.Length - 1];
        int batchElements = input.Length / vectorLength;

        var flatOutput = SquashJacobians(Engine.Reshape(input, new[] { batchElements, vectorLength }));

        var outputShape = new int[input.Shape.Length + 1];
        for (int i = 0; i < input.Shape.Length; i++)
        {
            outputShape[i] = input.Shape[i];
        }
        outputShape[outputShape.Length - 1] = vectorLength;

        return Engine.Reshape(flatOutput, outputShape);
    }

}
