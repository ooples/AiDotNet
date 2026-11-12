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
    /// Per-sample response-distillation loss at a fixed temperature: T² KL(teacher_soft || student_soft),
    /// blended as α × CE(student_probs, labels) + (1 - α) × soft when labels are provided.
    /// </summary>
    public static Vector<T> StandardSampleLosses(
        Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch, double temperature, double alpha)
    {
        Matrix<T> studentSoft = SoftmaxRows(studentBatchOutput, temperature);
        Matrix<T> teacherSoft = SoftmaxRows(teacherBatchOutput, temperature);
        Vector<T> losses = Engine.Multiply(
            KLDivergenceRows(teacherSoft, studentSoft), NumOps.FromDouble(temperature * temperature));

        if (trueLabelsBatch != null)
        {
            Matrix<T> studentProbs = SoftmaxRows(studentBatchOutput, 1.0);
            Vector<T> hardLosses = CrossEntropyRows(studentProbs, trueLabelsBatch);
            losses = Engine.Add(
                Engine.Multiply(hardLosses, NumOps.FromDouble(alpha)),
                Engine.Multiply(losses, NumOps.FromDouble(1.0 - alpha)));
        }

        return losses;
    }

    /// <summary>
    /// Per-sample (not batch-averaged) gradient of <see cref="StandardSampleLosses"/> w.r.t. student logits:
    /// (student_soft - teacher_soft) × T, blended as α × (student_probs - labels) + (1 - α) × soft.
    /// </summary>
    public static Matrix<T> StandardSampleGradients(
        Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch, double temperature, double alpha)
    {
        Matrix<T> studentSoft = SoftmaxRows(studentBatchOutput, temperature);
        Matrix<T> teacherSoft = SoftmaxRows(teacherBatchOutput, temperature);
        Matrix<T> gradient = Engine.MatrixMultiplyScalar(
            Engine.MatrixSubtract(studentSoft, teacherSoft), NumOps.FromDouble(temperature));

        if (trueLabelsBatch != null)
        {
            Matrix<T> studentProbs = SoftmaxRows(studentBatchOutput, 1.0);
            gradient = Engine.MatrixAdd(
                Engine.MatrixMultiplyScalar(Engine.MatrixSubtract(studentProbs, trueLabelsBatch), NumOps.FromDouble(alpha)),
                Engine.MatrixMultiplyScalar(gradient, NumOps.FromDouble(1.0 - alpha)));
        }

        return gradient;
    }

    /// <summary>
    /// Index of the largest element, matching a left-to-right scan that only moves on a strictly
    /// greater value: ties keep the first index, NaN entries never win, and a NaN first element
    /// yields 0. Throws for an empty vector, like reading its first element.
    /// </summary>
    public static int ArgMax(Vector<T> vector)
    {
        T first = vector[0];
        if (NumOps.IsNaN(first))
        {
            return 0;
        }

        int n = vector.Length;
        Tensor<T> values = Tensor<T>.FromVector(vector);
        // NaN entries can never win a strict '>' comparison; standing in the first element keeps them out
        Tensor<T> cleaned = Engine.TensorWhere(
            Engine.TensorNotEquals(values, values), Tensor<T>.CreateDefault([n], first), values);
        T max = Engine.TensorMaxValue(cleaned);
        Tensor<T> positions = Engine.TensorArange<T>(0, n, 1);
        T firstIndex = Engine.TensorMinValue(Engine.TensorWhere(
            Engine.TensorEquals(cleaned, max), positions, Tensor<T>.CreateDefault([n], NumOps.FromDouble(n))));
        return (int)Math.Round(NumOps.ToDouble(firstIndex));
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
