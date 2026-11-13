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
        return Engine.Softmax(
            Engine.TensorDivideScalar(Tensor<T>.FromVector(logits).Cast<double>(), temperature), -1).Cast<T>().ToVector();
    }

    /// <summary>
    /// Computes KL divergence: KL(p || q) = sum(p * log(p / q)).
    /// </summary>
    public static T KLDivergence(Vector<T> p, Vector<T> q)
    {
        var pClamped = ClampBelow(Tensor<T>.FromVector(p).Cast<double>(), Epsilon);
        var qClamped = ClampBelow(Tensor<T>.FromVector(q).Cast<double>(), Epsilon);
        return NumOps.FromDouble(Engine.TensorSum(Engine.TensorMultiply(pClamped,
            Engine.TensorLog(Engine.TensorDivide(pClamped, qClamped)))));
    }

    /// <summary>
    /// Computes cross-entropy: H(p, q) = -sum(p * log(q)).
    /// </summary>
    public static T CrossEntropy(Vector<T> p, Vector<T> q)
    {
        var labels = Tensor<T>.FromVector(p).Cast<double>();
        var predictions = ClampBelow(Tensor<T>.FromVector(q).Cast<double>(), Epsilon);
        return NumOps.FromDouble(-Engine.TensorSum(Engine.TensorMultiply(labels, Engine.TensorLog(predictions))));
    }

    /// <summary>
    /// Applies temperature-scaled softmax to each row of a logits matrix.
    /// </summary>
    public static Matrix<T> SoftmaxRows(Matrix<T> logits, double temperature)
    {
        return Engine.Softmax(
            Engine.TensorDivideScalar(Tensor<T>.FromMatrix(logits).Cast<double>(), temperature), 1).Cast<T>().ToMatrix();
    }

    /// <summary>
    /// Computes KL(p[r] || q[r]) for every row r (same clamping as <see cref="KLDivergence"/>).
    /// </summary>
    public static Vector<T> KLDivergenceRows(Matrix<T> p, Matrix<T> q)
    {
        var pClamped = ClampBelow(Tensor<T>.FromMatrix(p).Cast<double>(), Epsilon);
        var qClamped = ClampBelow(Tensor<T>.FromMatrix(q).Cast<double>(), Epsilon);
        var terms = Engine.TensorMultiply(pClamped, Engine.TensorLog(Engine.TensorDivide(pClamped, qClamped)));
        return Engine.ReduceSum(terms, [1], keepDims: false).Cast<T>().ToVector();
    }

    /// <summary>
    /// Computes H(p[r], q[r]) for every row r (same clamping as <see cref="CrossEntropy"/>).
    /// </summary>
    public static Vector<T> CrossEntropyRows(Matrix<T> p, Matrix<T> q)
    {
        var labels = Tensor<T>.FromMatrix(p).Cast<double>();
        var predictions = ClampBelow(Tensor<T>.FromMatrix(q).Cast<double>(), Epsilon);
        var terms = Engine.TensorNegate(Engine.TensorMultiply(labels, Engine.TensorLog(predictions)));
        return Engine.ReduceSum(terms, [1], keepDims: false).Cast<T>().ToVector();
    }

    /// <summary>
    /// Per-sample response-distillation loss at a fixed temperature: T² KL(teacher_soft || student_soft),
    /// blended as α × CE(student_probs, labels) + (1 - α) × soft when labels are provided.
    /// </summary>
    public static Vector<T> StandardSampleLosses(
        Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch, double temperature, double alpha)
    {
        if (typeof(T) != typeof(double))
            return Tensor<double>.FromVector(DistillationHelper<double>.StandardSampleLosses(
                Tensor<T>.FromMatrix(studentBatchOutput).Cast<double>().ToMatrix(),
                Tensor<T>.FromMatrix(teacherBatchOutput).Cast<double>().ToMatrix(),
                trueLabelsBatch is null ? null : Tensor<T>.FromMatrix(trueLabelsBatch).Cast<double>().ToMatrix(),
                temperature, alpha)).Cast<T>().ToVector();

        Matrix<T> studentSoft = SoftmaxRows(studentBatchOutput, temperature);
        Matrix<T> teacherSoft = SoftmaxRows(teacherBatchOutput, temperature);
        Vector<T> losses = Engine.Multiply(
            KLDivergenceRows(teacherSoft, studentSoft), NumOps.FromDouble(temperature * temperature));

        if (trueLabelsBatch != null)
        {
            Matrix<T> studentProbs = SoftmaxRows(studentBatchOutput, 1.0);
            Vector<T> hardLosses = CrossEntropyRows(trueLabelsBatch, studentProbs);
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
        if (typeof(T) != typeof(double))
            return Tensor<double>.FromMatrix(DistillationHelper<double>.StandardSampleGradients(
                Tensor<T>.FromMatrix(studentBatchOutput).Cast<double>().ToMatrix(),
                Tensor<T>.FromMatrix(teacherBatchOutput).Cast<double>().ToMatrix(),
                trueLabelsBatch is null ? null : Tensor<T>.FromMatrix(trueLabelsBatch).Cast<double>().ToMatrix(),
                temperature, alpha)).Cast<T>().ToMatrix();

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
        if (NumOps.IsNaN(first)) return 0;
        int n = vector.Length;
        var values = Tensor<T>.FromVector(vector);
        var cleaned = Engine.TensorWhere(
            Engine.TensorNotEquals(values, values), Tensor<T>.CreateDefault([n], first), values);
        T max = Engine.TensorMaxValue(cleaned);
        var positions = Engine.TensorArange<int>(0, n, 1);
        return Engine.TensorMinValue(Engine.TensorWhere(
            Engine.TensorEquals(cleaned, max).Cast<int>(), positions, Tensor<int>.CreateDefault([n], n)));
    }

    /// <summary>
    /// Cosine similarity of every row of <paramref name="a"/> with every row of <paramref name="b"/>
    /// ([n, d] and [m, d] -> [n, m]) with VectorHelper.CosineSimilarity semantics: clamped to [-1, 1],
    /// and 0 where the norm product is at most 1e-10.
    /// </summary>
    public static Tensor<T> CosineSimilarityMatrix(Tensor<T> a, Tensor<T> b)
    {
        var wideA = a.Cast<double>();
        var wideB = b.Cast<double>();
        var dot = Engine.TensorMatMul(wideA, Engine.TensorTranspose(wideB));
        var normsA = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(wideA, wideA), [1], keepDims: true));
        var normsB = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(wideB, wideB), [1], keepDims: true));
        var denom = Engine.TensorMatMul(normsA, Engine.TensorTranspose(normsB));
        var positive = Engine.TensorGreaterThan(denom, 1e-10);
        var safeDenom = Engine.TensorWhere(positive, denom, Tensor<double>.CreateDefault(denom.Shape.ToArray(), 1.0));
        var cosine = Engine.TensorClamp(Engine.TensorDivide(dot, safeDenom), -1.0, 1.0);
        return Engine.TensorWhere(positive, cosine, new Tensor<double>(dot.Shape.ToArray())).Cast<T>();
    }

    /// <summary>
    /// Stacks equally sized vectors as the rows of an [n, d] tensor.
    /// </summary>
    public static Tensor<T> StackRows(Vector<T>[] rows)
    {
        return Engine.TensorStack(rows.Select(row => Tensor<T>.FromVector(row)).ToArray(), 0);
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
    private static Tensor<double> ClampBelow(Tensor<double> values, double floor)
    {
        return Engine.TensorMax(Tensor<double>.CreateDefault(values.Shape.ToArray(), floor), values);
    }

    // Probability floors are applied in double before any conversion back to T.
}
