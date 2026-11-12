using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Shared helper methods for distillation strategies.
/// </summary>
internal static class DistillationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private const double Epsilon = 1e-10;

    /// <summary>
    /// Applies temperature-scaled softmax to logits.
    /// </summary>
    public static Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        // Max-shifted softmax of logits / temperature
        return Engine.Softmax(
            Engine.TensorDivideScalar(Tensor<T>.FromVector(logits), NumOps.FromDouble(temperature)), -1).ToVector();
    }

    /// <summary>
    /// Computes KL divergence: KL(p || q) = sum(p * log(p / q)).
    /// </summary>
    public static T KLDivergence(Vector<T> p, Vector<T> q)
    {
        Vector<T> pClamped = ClampBelow(p, Epsilon);
        Vector<T> qClamped = ClampBelow(q, Epsilon);
        return Engine.Sum(Engine.Multiply(pClamped, Engine.Log(Engine.Divide(pClamped, qClamped))));
    }

    /// <summary>
    /// Computes cross-entropy: H(p, q) = -sum(p * log(q)).
    /// </summary>
    public static T CrossEntropy(Vector<T> p, Vector<T> q)
    {
        return NumOps.Negate(Engine.Sum(Engine.Multiply(p, Engine.Log(ClampBelow(q, Epsilon)))));
    }

    /// <summary>
    /// Applies temperature-scaled softmax to each row of a logits matrix.
    /// </summary>
    public static Matrix<T> SoftmaxRows(Matrix<T> logits, double temperature)
    {
        return Engine.Softmax(
            Engine.TensorDivideScalar(Tensor<T>.FromMatrix(logits), NumOps.FromDouble(temperature)), 1).ToMatrix();
    }

    /// <summary>
    /// Computes KL(p[r] || q[r]) for every row r (same clamping as <see cref="KLDivergence"/>).
    /// </summary>
    public static Vector<T> KLDivergenceRows(Matrix<T> p, Matrix<T> q)
    {
        Tensor<T> pClamped = ClampBelow(Tensor<T>.FromMatrix(p), Epsilon);
        Tensor<T> qClamped = ClampBelow(Tensor<T>.FromMatrix(q), Epsilon);
        return RowSums(Engine.TensorMultiply(pClamped, Engine.TensorLog(Engine.TensorDivide(pClamped, qClamped))));
    }

    /// <summary>
    /// Computes H(p[r], q[r]) for every row r (same clamping as <see cref="CrossEntropy"/>).
    /// </summary>
    public static Vector<T> CrossEntropyRows(Matrix<T> p, Matrix<T> q)
    {
        Tensor<T> terms = Engine.TensorMultiply(
            Tensor<T>.FromMatrix(p), Engine.TensorLog(ClampBelow(Tensor<T>.FromMatrix(q), Epsilon)));
        return Engine.Negate(RowSums(terms));
    }

    /// <summary>
    /// Sums each row of a [rows, cols] tensor into a vector of length rows.
    /// </summary>
    public static Vector<T> RowSums(Tensor<T> matrix)
    {
        return Engine.ReduceSum(matrix, [1], keepDims: false).ToVector();
    }

    /// <summary>
    /// Elementwise max(value, floor); NaN entries stay NaN (like Math.Max).
    /// </summary>
    private static Vector<T> ClampBelow(Vector<T> values, double floor)
    {
        return Engine.Max(Vector<T>.CreateDefault(values.Length, NumOps.FromDouble(floor)), values);
    }

    /// <summary>
    /// Elementwise max(value, floor); NaN entries stay NaN (like Math.Max).
    /// </summary>
    private static Tensor<T> ClampBelow(Tensor<T> values, double floor)
    {
        return Engine.TensorMax(Tensor<T>.CreateDefault(values.Shape.ToArray(), NumOps.FromDouble(floor)), values);
    }
}
