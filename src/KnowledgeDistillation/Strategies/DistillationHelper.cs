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
        => CosineSimilarityMatrixWide(a, b).Cast<T>();

    /// <summary>
    /// <see cref="CosineSimilarityMatrix"/> kept in double, for callers that exponentiate the result.
    /// </summary>
    public static Tensor<double> CosineSimilarityMatrixWide(Tensor<T> a, Tensor<T> b)
    {
        var wideA = a.Cast<double>();
        var wideB = b.Cast<double>();
        var dot = Engine.TensorMatMul(wideA, Engine.TensorTranspose(wideB));
        var normsA = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(wideA, wideA), [1], keepDims: true));
        var normsB = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(wideB, wideB), [1], keepDims: true));
        return GuardedCosine(dot, Engine.TensorMatMul(normsA, Engine.TensorTranspose(normsB)));
    }

    /// <summary>
    /// Cosine similarity of each row of <paramref name="a"/> with the same row of <paramref name="b"/>
    /// ([n, d] and [n, d] -> [n]), in double, with the same clamping and degenerate-norm rule as
    /// <see cref="CosineSimilarityMatrix"/>.
    /// </summary>
    public static Tensor<double> RowwiseCosineSimilarityWide(Tensor<T> a, Tensor<T> b)
    {
        var wideA = a.Cast<double>();
        var wideB = b.Cast<double>();
        var dot = Engine.ReduceSum(Engine.TensorMultiply(wideA, wideB), [1], keepDims: false);
        var normsA = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(wideA, wideA), [1], keepDims: false));
        var normsB = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(wideB, wideB), [1], keepDims: false));
        return GuardedCosine(dot, Engine.TensorMultiply(normsA, normsB));
    }

    // Norms are multiplied after their square roots so large finite rows cannot overflow the product.
    private static Tensor<double> GuardedCosine(Tensor<double> dot, Tensor<double> denom)
    {
        var positive = Engine.TensorGreaterThan(denom, 1e-10);
        var safeDenom = Engine.TensorWhere(positive, denom, Tensor<double>.CreateDefault(denom.Shape.ToArray(), 1.0));
        var cosine = Engine.TensorClamp(Engine.TensorDivide(dot, safeDenom), -1.0, 1.0);
        return Engine.TensorWhere(positive, cosine, new Tensor<double>(dot.Shape.ToArray()));
    }

    /// <summary>
    /// Computes log(Σ_j exp(x_ij) + epsilon) for each row of an [n, m] logit tensor without overflow:
    /// each row is shifted by its maximum, and epsilon is folded in with a stable log-add-exp.
    /// Entries equal to negative infinity contribute nothing; every row needs one finite entry.
    /// </summary>
    public static Tensor<double> RowLogSumExp(Tensor<double> logits, double epsilon)
    {
        int n = logits.Shape[0], m = logits.Shape[1];
        var rowMax = Engine.ReduceMax(logits, [1], keepDims: true);
        var shifted = Engine.TensorExp(Engine.TensorSubtract(logits, Engine.TensorBroadcastTo(rowMax, [n, m])));
        var logSum = Engine.Reshape(
            Engine.TensorAdd(Engine.TensorLog(Engine.ReduceSum(shifted, [1], keepDims: true)), rowMax), [n]);
        if (epsilon <= 0.0) return logSum;

        // log(e^a + e^b) = max(a, b) + log(1 + e^-|a - b|), with b = log(epsilon)
        var logEpsilon = Tensor<double>.CreateDefault([n], Math.Log(epsilon));
        var gap = Engine.TensorNegate(Engine.TensorAbs(Engine.TensorSubtract(logSum, logEpsilon)));
        return Engine.TensorAdd(
            Engine.TensorMax(logSum, logEpsilon),
            Engine.TensorLog(Engine.TensorAddScalar(Engine.TensorExp(gap), 1.0)));
    }

    /// <summary>
    /// Stacks equally sized vectors as the rows of an [n, d] tensor.
    /// </summary>
    public static Tensor<T> StackRows(Vector<T>[] rows)
    {
        return Engine.TensorStack(rows.Select(row => Tensor<T>.FromVector(row)).ToArray(), 0);
    }

    /// <summary>
    /// The rows of <paramref name="matrix"/> as separate vectors.
    /// </summary>
    public static Vector<T>[] RowsOf(Matrix<T> matrix)
    {
        return Enumerable.Range(0, matrix.Rows).Select(r => matrix.GetRow(r)).ToArray();
    }

    /// <summary>
    /// [n, n] 1/0 mask of the pairs i &lt; j (<paramref name="upperOnly"/>) or of all pairs i != j.
    /// </summary>
    public static Tensor<T> PairMask(int n, bool upperOnly)
    {
        var rows = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, n, 1), [n, 1]), [n, n]);
        var cols = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorArange<T>(0, n, 1), [1, n]), [n, n]);
        return upperOnly ? Engine.TensorLessThan(rows, cols) : Engine.TensorNotEquals(rows, cols);
    }

    /// <summary>
    /// [n, n] mask that is 1 off the diagonal (pairs i != j) and 0 on it.
    /// </summary>
    public static Tensor<T> OffDiagonalMask(int n) => PairMask(n, upperOnly: false);

    /// <summary>
    /// Maps a gradient w.r.t. softmax(z / T) probabilities to a gradient w.r.t. the logits z:
    /// ∂L/∂z_k = p_k * (∂L/∂p_k - Σ_m (∂L/∂p_m) p_m) / T, row by row.
    /// </summary>
    public static Tensor<T> SoftmaxJacobianProduct(Tensor<T> probabilities, Tensor<T> probabilityGradient, double temperature)
    {
        var weightedSum = Engine.ReduceSum(Engine.TensorMultiply(probabilityGradient, probabilities), [1], keepDims: true);
        var centered = Engine.TensorSubtract(
            probabilityGradient, Engine.TensorBroadcastTo(weightedSum, probabilities.Shape.ToArray()));
        return Engine.TensorDivideScalar(Engine.TensorMultiply(probabilities, centered), NumOps.FromDouble(temperature));
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
