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

        // y = g(r) u with r = |x|, u = x / r and g(r) = r^2 / (1 + r^2), so
        //   J = a (I - u u^T) + g'(r) u u^T,   a = g(r) / r = r / (1 + r^2),   g'(r) = 2 r / (1 + r^2)^2.
        // The directional term a (I - u u^T) and the radial term g'(r) u u^T are evaluated apart: folded into one
        // coefficient of x x^T, the radial term was rounded away against a (for [2^30] the derivative 2^-89
        // became 0). At x = 0 both terms vanish (y ~ |x| x), and u is taken as 0 there.
        bool scalar = UseScalarJacobian(n);
        var engine = Engine;
        var x = scalar ? null : Tensor<T>.FromVector(input);
        T normSquared = x is null ? ScalarSumOfSquares(input) : engine.TensorSum(engine.TensorMultiply(x, x));
        T norm = NumOps.Sqrt(normSquared);
        T normSqPlusOne = NumOps.Add(normSquared, NumOps.One);
        T a = NumOps.Divide(norm, normSqPlusOne);
        T radial = NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2), norm), NumOps.Multiply(normSqPlusOne, normSqPlusOne));
        T normInv = NumOps.GreaterThan(norm, NumOps.Zero) ? NumOps.Divide(NumOps.One, norm) : NumOps.Zero;

        if (x is null)
        {
            return ScalarDirectionalPlusRadial(input, normInv, a, radial);
        }

        // Off the diagonal (g' - a) u_i u_j; on it a (1 - u_i^2) + g' u_i^2.
        var u = engine.TensorMultiplyScalar(x, normInv);
        var uSquared = engine.TensorMultiply(u, u);
        var diagonal = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorSubtract(engine.TensorOnesLike(uSquared), uSquared), a),
            engine.TensorMultiplyScalar(uSquared, radial));
        return OuterWithDiagonal(engine.TensorMultiplyScalar(u, NumOps.Subtract(radial, a)), u, diagonal);
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

    /// <summary>
    /// <c>a (I - u u^T) + radial u u^T</c> with <c>u = x * normInv</c>, in scalar arithmetic; the diagonal keeps the
    /// two terms apart so neither is rounded away against the other.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private Matrix<T> ScalarDirectionalPlusRadial(Vector<T> input, T normInv, T a, T radial)
    {
        int n = input.Length;
        var x = input.AsSpan();
        var jacobian = new Matrix<T>(n, n);
        var cells = jacobian.AsWritableSpan();
        T offDiagonal = NumOps.Subtract(radial, a);
        for (int i = 0; i < n; i++)
        {
            T ui = NumOps.Multiply(x[i], normInv);
            T scaled = NumOps.Multiply(ui, offDiagonal);
            int row = i * n;
            for (int j = 0; j < n; j++)
            {
                cells[row + j] = NumOps.Multiply(scaled, NumOps.Multiply(x[j], normInv));
            }

            T uiSquared = NumOps.Multiply(ui, ui);
            cells[row + i] = NumOps.Add(
                NumOps.Multiply(a, NumOps.Subtract(NumOps.One, uiSquared)), NumOps.Multiply(radial, uiSquared));
        }

        return jacobian;
    }

    /// <summary>
    /// The epsilon <c>Engine.TensorSquash</c> adds to the capsule norm: y = x r^2 / ((1 + r^2)(r + eps)).
    /// </summary>
    private const double CapsuleSquashEpsilon = 1e-8;

    /// <summary>
    /// The squash Jacobian of every row of a [batch, n] tensor, as a [batch, n, n] tensor.
    /// </summary>
    /// <param name="vectors">The capsule vectors, one per row.</param>
    /// <returns>For each row x, the matrix dy_i/dx_j of y = squash(x).</returns>
    /// <remarks>
    /// <para>
    /// This differentiates the forward the tensor overload runs, <c>Engine.TensorSquash</c>, which divides by the
    /// stabilised norm r + eps. Its derivative is J = A I + B x x^T with the coefficients of
    /// <c>TensorSquashBackward</c>: A = r^2 / ((1 + r^2)(r + eps)) and B = (r + 2 eps - r^3) / ((1 + r^2)^2 (r + eps)^2).
    /// A zero capsule has A = 0 and a finite B times x x^T = 0, so its Jacobian is zero without a 0 / 0.
    /// </para>
    /// <para>
    /// The per-row coefficients come from engine reductions over [batch] values; the [batch, n, n] result is the
    /// only result-sized buffer and is written directly, one outer product per row, rather than through tiled
    /// full-batch operands.
    /// </para>
    /// </remarks>
    private Tensor<T> SquashJacobians(Tensor<T> vectors)
    {
        var engine = Engine;
        int batch = vectors.Shape[0];
        int n = vectors.Shape[1];

        var normSquared = engine.ReduceSum(engine.TensorMultiply(vectors, vectors), new[] { 1 }, keepDims: false);
        var norm = engine.TensorSqrt(normSquared);
        var onePlusNormSquared = engine.TensorAddScalar(normSquared, NumOps.One);
        var normPlusEpsilon = engine.TensorAddScalar(norm, NumOps.FromDouble(CapsuleSquashEpsilon));
        var diagonal = engine.TensorDivide(normSquared, engine.TensorMultiply(onePlusNormSquared, normPlusEpsilon));
        var outerNumerator = engine.TensorAddScalar(
            engine.TensorSubtract(norm, engine.TensorMultiply(normSquared, norm)),
            NumOps.FromDouble(2 * CapsuleSquashEpsilon));
        var outerDenominator = engine.TensorMultiply(
            engine.TensorMultiply(onePlusNormSquared, onePlusNormSquared),
            engine.TensorMultiply(normPlusEpsilon, normPlusEpsilon));
        var outer = engine.TensorDivide(outerNumerator, outerDenominator);

        var result = new Tensor<T>(new[] { batch, n, n });
        FillJacobians(vectors.Contiguous(), diagonal.Contiguous(), outer.Contiguous(), result, batch, n);
        return result;
    }

    /// <summary><c>J_b = A_b I + B_b x_b x_b^T</c> for every row b, written into <paramref name="result"/>.</summary>
    [AiDotNet.Attributes.ScalarKernel("writes the [batch, n, n] result once, row by row, instead of materialising tiled full-batch "
        + "operands for an outer product and an identity")]
    private void FillJacobians(Tensor<T> vectors, Tensor<T> diagonal, Tensor<T> outer, Tensor<T> result, int batch, int n)
    {
        var x = vectors.AsSpan();
        var a = diagonal.AsSpan();
        var b = outer.AsSpan();
        var cells = result.AsWritableSpan();
        for (int row = 0; row < batch; row++)
        {
            int xBase = row * n;
            int jBase = row * n * n;
            for (int i = 0; i < n; i++)
            {
                T scaled = NumOps.Multiply(b[row], x[xBase + i]);
                int cellRow = jBase + i * n;
                for (int j = 0; j < n; j++)
                {
                    cells[cellRow + j] = NumOps.Multiply(scaled, x[xBase + j]);
                }

                cells[cellRow + i] = NumOps.Add(cells[cellRow + i], a[row]);
            }
        }
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
        var outputShape = new int[input.Shape.Length + 1];
        for (int i = 0; i < input.Shape.Length; i++)
        {
            outputShape[i] = input.Shape[i];
        }
        outputShape[outputShape.Length - 1] = vectorLength;

        // An empty batch (or empty capsules) has an empty Jacobian of the same leading shape; the batch size below
        // would divide by a zero vector length, and the kernel has no row to fill.
        if (input.Length == 0)
        {
            return new Tensor<T>(outputShape);
        }

        int batchElements = input.Length / vectorLength;
        var flatOutput = SquashJacobians(Engine.Reshape(input, new[] { batchElements, vectorLength }));
        return Engine.Reshape(flatOutput, outputShape);
    }

}
