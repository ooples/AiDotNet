using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// Tape-visible building blocks shared by the self-supervised objectives.
/// </summary>
/// <remarks>
/// <para>
/// Every operation here goes through <see cref="IEngine"/> so the result carries tape history and
/// the objective can actually produce gradients. These objectives previously assembled their loss
/// from host loops over <c>Tensor</c> indexers and returned a bare scalar, which cannot be
/// differentiated — the loss could be measured but never trained on.
/// </para>
/// <para>
/// One audited implementation, shared. Hand-rolling a contrastive objective per model is where the
/// classic bugs live: reducing the wrong axis, dropping the symmetric term, or exponentiating raw
/// logits and overflowing at a small temperature.
/// </para>
/// </remarks>
internal static class ObjectiveOps
{
    /// <summary>
    /// L2-normalizes each row, differentiably.
    /// </summary>
    /// <remarks>
    /// <paramref name="epsilon"/> is added inside the square root so a zero row yields a finite
    /// gradient instead of a division by zero.
    /// </remarks>
    internal static Tensor<T> L2NormalizeRows<T>(Tensor<T> x, double epsilon = 1e-12)
    {
        var engine = AiDotNetEngine.Current;
        var squared = engine.TensorMultiply(x, x);
        var sumSq = engine.ReduceSum(squared, new[] { x.Shape.Length - 1 }, keepDims: true);
        var safe = engine.TensorAddScalar(sumSq, MathHelper.GetNumericOperations<T>().FromDouble(epsilon));
        var norm = engine.TensorSqrt(safe);
        // norm is [rows, 1] against x's [rows, cols]; this engine does not broadcast implicitly.
        return engine.TensorBroadcastDivide(x, norm);
    }

    /// <summary>
    /// Numerically stable log-softmax along <paramref name="axis"/>, differentiably.
    /// </summary>
    /// <remarks>
    /// Computed as <c>x - (max + log(sum(exp(x - max))))</c>. The max subtraction is what keeps
    /// <c>exp</c> from overflowing; a naive <c>log(softmax(x))</c> overflows at the small
    /// temperatures these objectives use (MoCo's default is 0.07, so logits are scaled by ~14x).
    /// </remarks>
    internal static Tensor<T> LogSoftmax<T>(Tensor<T> logits, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var max = engine.ReduceMax(logits, new[] { axis }, keepDims: true);
        var shifted = engine.TensorBroadcastSubtract(logits, max);
        var sumExp = engine.ReduceSum(engine.TensorExp(shifted), new[] { axis }, keepDims: true);
        return engine.TensorBroadcastSubtract(shifted, engine.TensorLog(sumExp));
    }

    /// <summary>
    /// Mean of the diagonal of a square <c>[n, n]</c> tensor, differentiably.
    /// </summary>
    /// <remarks>
    /// Extracted by multiplying with a constant identity mask and reducing, rather than indexing.
    /// Indexing would read values off the tape and sever the gradient — the exact mistake that made
    /// these objectives untrainable.
    /// </remarks>
    internal static Tensor<T> MeanDiagonal<T>(Tensor<T> square)
    {
        var engine = AiDotNetEngine.Current;
        int n = square.Shape[0];
        var identity = Identity<T>(n);
        var picked = engine.TensorMultiply(square, identity);
        var summed = engine.ReduceSum(picked, null, keepDims: false);
        return engine.TensorDivideScalar(summed, MathHelper.GetNumericOperations<T>().FromDouble(n));
    }

    /// <summary>
    /// Constant <c>[n, n]</c> identity. Not a parameter and never on the tape as a leaf to learn.
    /// </summary>
    internal static Tensor<T> Identity<T>(int n)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var identity = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++) identity[i, i] = ops.One;
        return identity;
    }

    /// <summary>
    /// Scaled similarity matrix <c>a @ b^T / temperature</c>, optionally L2-normalized first.
    /// </summary>
    internal static Tensor<T> SimilarityMatrix<T>(Tensor<T> a, Tensor<T> b, double temperature, bool normalize)
    {
        var engine = AiDotNetEngine.Current;
        var left = normalize ? L2NormalizeRows(a) : a;
        var right = normalize ? L2NormalizeRows(b) : b;
        var logits = engine.TensorMatMul(left, engine.TensorTranspose(right));
        return engine.TensorMultiplyScalar(
            logits, MathHelper.GetNumericOperations<T>().FromDouble(1.0 / temperature));
    }
}
