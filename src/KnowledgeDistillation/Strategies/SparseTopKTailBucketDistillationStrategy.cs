using System;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Sparse Top-K knowledge distillation with an exact <c>(K + 1)</c>-th remainder tail bucket that prevents
/// pathological tail starvation on large-vocabulary language, speech, and multimodal models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (for example, <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// When distilling over a large vocabulary <c>V</c> (such as <c>32K–256K</c> tokens) from only the teacher's
/// top-<c>K</c> logits per token, naively renormalizing the teacher's top-<c>K</c> probabilities to sum to <c>1</c>
/// (<c>q_T(tail) = 0</c>) while normalizing the student over all <c>V</c> classes forces <c>V - K</c> tail logits
/// toward <c>-infinity</c>, severely degrading <c>T = 1</c> perplexity whenever a held-out token falls outside the
/// teacher's top-<c>K</c> set.
/// </para>
/// <para>
/// This strategy evaluates the exact coarse-grained <c>(K + 1)</c>-outcome Kullback-Leibler divergence over
/// <c>{top_1, ..., top_K, Tail}</c> using vectorized <see cref="IEngine"/> tensor operations, preserving the teacher's
/// un-renormalized full-vocabulary top-<c>K</c> probabilities <c>q_{T,j} = softmax_V(z_T / T)[idx_j]</c> together with
/// the exact remainder tail mass <c>q_{T,tail} = 1 - sum_{j=1}^K q_{T,j}</c> matched against
/// <c>p_{S,tail} = 1 - sum_{j=1}^K p_S(idx_j)</c>. By the data-processing inequality,
/// <c>KL_{K+1}(q_T || p_S) &gt;= 0</c> with <c>KL_{K+1} = 0</c> at <c>z_S = z_T + c</c>.
/// </para>
/// <example>
/// <code>
/// var strategy = new SparseTopKTailBucketDistillationStrategy&lt;float&gt;(topK: 64, temperature: 2.0, alpha: 0.5);
/// float loss = strategy.ComputeLoss(studentLogits, teacherLogits, trueLabels);
/// Matrix&lt;float&gt; grad = strategy.ComputeGradient(studentLogits, teacherLogits, trueLabels);
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class SparseTopKTailBucketDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private const double ProbabilityFloor = 1e-12;
    private readonly int _topK;

    /// <summary>
    /// Gets the maximum number of top teacher classes retained explicitly before aggregating the remainder into the tail bucket.
    /// </summary>
    public int TopK => _topK;

    /// <summary>
    /// Initializes a new instance of the <see cref="SparseTopKTailBucketDistillationStrategy{T}"/> class.
    /// </summary>
    /// <param name="topK">Number of top teacher classes per sample to retain explicitly (must be positive).</param>
    /// <param name="temperature">Softmax temperature <c>T &gt; 0</c> (default <c>2.0</c>).</param>
    /// <param name="alpha">Blend weight in <c>[0, 1]</c> on hard-label cross-entropy when labels are supplied (default <c>0.5</c>).</param>
    public SparseTopKTailBucketDistillationStrategy(
        int topK = 64,
        double temperature = 2.0,
        double alpha = 0.5)
        : base(temperature, alpha)
    {
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), topK, "TopK must be at least 1.");
        _topK = topK;
    }

    /// <inheritdoc />
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        ExtractTopKTargets(teacherBatchOutput, out Tensor<int> topIndices, out Matrix<T> topProbabilities);
        return ComputeLossFromSparseTargets(studentBatchOutput, topIndices, topProbabilities, trueLabelsBatch);
    }

    /// <inheritdoc />
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        ExtractTopKTargets(teacherBatchOutput, out Tensor<int> topIndices, out Matrix<T> topProbabilities);
        return ComputeGradientFromSparseTargets(studentBatchOutput, topIndices, topProbabilities, trueLabelsBatch);
    }

    /// <summary>
    /// Extracts the un-renormalized full-vocabulary teacher top-<c>K</c> indices and probabilities at <see cref="DistillationStrategyBase{T}.Temperature"/>.
    /// </summary>
    /// <param name="teacherBatchOutput">Teacher logits of shape <c>[batchSize, numClasses]</c>.</param>
    /// <param name="topIndices">Output top-<c>K</c> class indices tensor of shape <c>[batchSize, effectiveK]</c>.</param>
    /// <param name="topProbabilities">Output un-renormalized teacher probabilities of shape <c>[batchSize, effectiveK]</c>.</param>
    public void ExtractTopKTargets(Matrix<T> teacherBatchOutput, out Tensor<int> topIndices, out Matrix<T> topProbabilities)
    {
        if (teacherBatchOutput is null) throw new ArgumentNullException(nameof(teacherBatchOutput));
        int rows = teacherBatchOutput.Rows;
        int cols = teacherBatchOutput.Columns;
        if (rows <= 0 || cols <= 0)
            throw new ArgumentException("Teacher output matrix must have positive dimensions.", nameof(teacherBatchOutput));

        int k = Math.Min(_topK, cols);
        var teacherProbs = Engine.Softmax(
            Engine.TensorDivideScalar(Tensor<T>.FromMatrix(teacherBatchOutput).Cast<double>(), Temperature), 1);
        var topVals = Engine.TensorTopK(teacherProbs, k, 1, out topIndices);
        topProbabilities = topVals.Cast<T>().ToMatrix();
    }

    /// <summary>
    /// Computes the batch-averaged <c>(K + 1)</c>-outcome tail-bucket sparse distillation loss directly from pre-extracted top-<c>K</c> teacher targets.
    /// </summary>
    public T ComputeLossFromSparseTargets(
        Matrix<T> studentBatchOutput,
        Tensor<int> topIndices,
        Matrix<T> topProbabilities,
        Matrix<T>? trueLabelsBatch = null)
    {
        ValidateSparseTargets(studentBatchOutput, topIndices, topProbabilities);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int rows = studentBatchOutput.Rows;
        int cols = studentBatchOutput.Columns;
        int k = topProbabilities.Columns;

        var studentDouble = Tensor<T>.FromMatrix(studentBatchOutput).Cast<double>();
        var stuSoft = Engine.Softmax(Engine.TensorDivideScalar(studentDouble, Temperature), 1);
        var qTop = Tensor<T>.FromMatrix(topProbabilities).Cast<double>();

        var rowOffsets = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<int>(0, rows * cols, cols), [rows, 1]),
            [rows, k]);
        var flatIdx = Engine.Reshape(Engine.TensorAdd(rowOffsets, topIndices), [rows * k]);
        var stuTop = Engine.Reshape(
            Engine.TensorGather(Engine.Reshape(stuSoft, [rows * cols]), flatIdx, 0),
            [rows, k]);

        var mTeacher = Engine.ReduceSum(qTop, [1], keepDims: true);
        var mStudent = Engine.ReduceSum(stuTop, [1], keepDims: true);

        var zeroRow = Tensor<double>.CreateDefault([rows, 1], 0.0);
        var floorRow = Tensor<double>.CreateDefault([rows, 1], ProbabilityFloor);
        var floorTop = Tensor<double>.CreateDefault([rows, k], ProbabilityFloor);

        var qTail = Engine.TensorMax(Engine.TensorAddScalar(Engine.TensorNegate(mTeacher), 1.0), zeroRow);
        var rawTailStu = Engine.TensorAddScalar(Engine.TensorNegate(mStudent), 1.0);
        var pTailSafe = Engine.TensorMax(rawTailStu, floorRow);
        var qTailSafe = Engine.TensorMax(qTail, floorRow);

        var qTopSafe = Engine.TensorMax(qTop, floorTop);
        var pTopSafe = Engine.TensorMax(stuTop, floorTop);

        var topKlRaw = Engine.TensorMultiply(qTop, Engine.TensorSubtract(Engine.TensorLog(qTopSafe), Engine.TensorLog(pTopSafe)));
        var topKl = Engine.TensorMultiply(Engine.TensorGreaterThan(qTop, floorTop), topKlRaw);

        var tailKlRaw = Engine.TensorMultiply(qTail, Engine.TensorSubtract(Engine.TensorLog(qTailSafe), Engine.TensorLog(pTailSafe)));
        var tailKl = Engine.TensorMultiply(Engine.TensorGreaterThan(qTail, floorRow), tailKlRaw);

        Tensor<double> rowSoftLoss = Engine.TensorMultiplyScalar<double>(
            Engine.TensorAdd(Engine.ReduceSum(topKl, [1], keepDims: true), tailKl),
            Temperature * Temperature);

        Tensor<double> rowTotalLoss;
        if (trueLabelsBatch is not null)
        {
            var labelsDouble = Tensor<T>.FromMatrix(trueLabelsBatch).Cast<double>();
            var stuHard = Engine.Softmax(studentDouble, 1);
            var stuHardSafe = Engine.TensorMax(stuHard, Tensor<double>.CreateDefault([rows, cols], ProbabilityFloor));
            var hardCeTerms = Engine.TensorNegate(Engine.TensorMultiply(labelsDouble, Engine.TensorLog(stuHardSafe)));
            var rowHardLoss = Engine.ReduceSum(hardCeTerms, [1], keepDims: true);
            rowTotalLoss = Engine.TensorAdd(
                Engine.TensorMultiplyScalar<double>(rowHardLoss, Alpha),
                Engine.TensorMultiplyScalar<double>(rowSoftLoss, 1.0 - Alpha));
        }
        else
        {
            rowTotalLoss = rowSoftLoss;
        }

        double totalSum = Engine.TensorSum(rowTotalLoss);
        return NumOps.FromDouble(totalSum / rows);
    }

    /// <summary>
    /// Computes the batch-averaged gradient w.r.t. <paramref name="studentBatchOutput"/> of <see cref="ComputeLossFromSparseTargets"/>.
    /// </summary>
    public Matrix<T> ComputeGradientFromSparseTargets(
        Matrix<T> studentBatchOutput,
        Tensor<int> topIndices,
        Matrix<T> topProbabilities,
        Matrix<T>? trueLabelsBatch = null)
    {
        ValidateSparseTargets(studentBatchOutput, topIndices, topProbabilities);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int rows = studentBatchOutput.Rows;
        int cols = studentBatchOutput.Columns;
        int k = topProbabilities.Columns;

        var studentDouble = Tensor<T>.FromMatrix(studentBatchOutput).Cast<double>();
        var stuSoft = Engine.Softmax(Engine.TensorDivideScalar(studentDouble, Temperature), 1);
        var qTop = Tensor<T>.FromMatrix(topProbabilities).Cast<double>();

        var oneHot = Engine.Reshape(
            Engine.TensorOneHot<double>(Engine.Reshape(topIndices, [rows * k]), cols),
            [rows, k, cols]);
        var isTopMask = Engine.ReduceSum(oneHot, [1], keepDims: false);
        var qTopFull = Engine.ReduceSum(
            Engine.TensorMultiply(oneHot, Engine.TensorBroadcastTo(Engine.Reshape(qTop, [rows, k, 1]), [rows, k, cols])),
            [1], keepDims: false);

        var rowOffsets = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<int>(0, rows * cols, cols), [rows, 1]),
            [rows, k]);
        var flatIdx = Engine.Reshape(Engine.TensorAdd(rowOffsets, topIndices), [rows * k]);
        var stuTop = Engine.Reshape(
            Engine.TensorGather(Engine.Reshape(stuSoft, [rows * cols]), flatIdx, 0),
            [rows, k]);

        var mTeacher = Engine.ReduceSum(qTop, [1], keepDims: true);
        var mStudent = Engine.ReduceSum(stuTop, [1], keepDims: true);

        var zeroRow = Tensor<double>.CreateDefault([rows, 1], 0.0);
        var floorRow = Tensor<double>.CreateDefault([rows, 1], ProbabilityFloor);
        var qTail = Engine.TensorMax(Engine.TensorAddScalar(Engine.TensorNegate(mTeacher), 1.0), zeroRow);
        var rawTailStu = Engine.TensorAddScalar(Engine.TensorNegate(mStudent), 1.0);
        var pTailSafe = Engine.TensorMax(rawTailStu, floorRow);

        var bothPositive = Engine.TensorMultiply(
            Engine.TensorGreaterThan(qTail, floorRow),
            Engine.TensorGreaterThan(rawTailStu, floorRow));

        var tailFactor = BlendByMask(
            bothPositive,
            Engine.TensorDivide(Engine.TensorSubtract(mTeacher, mStudent), pTailSafe),
            mTeacher);

        var mTeacherFull = Engine.TensorBroadcastTo(mTeacher, [rows, cols]);
        var bothPositiveFull = Engine.TensorBroadcastTo(bothPositive, [rows, cols]);
        var topGradUnits = BlendByMask(
            bothPositiveFull,
            Engine.TensorSubtract(stuSoft, qTopFull),
            Engine.TensorSubtract(Engine.TensorMultiply(stuSoft, mTeacherFull), qTopFull));

        var tailGradUnits = Engine.TensorMultiply(stuSoft, Engine.TensorBroadcastTo(tailFactor, [rows, cols]));
        var isTopMaskBinary = Engine.TensorGreaterThan(isTopMask, Tensor<double>.CreateDefault([rows, cols], 0.5));
        Tensor<double> softGrad = Engine.TensorMultiplyScalar<double>(
            BlendByMask(isTopMaskBinary, topGradUnits, tailGradUnits),
            Temperature);

        Tensor<double> totalGrad;
        if (trueLabelsBatch is not null)
        {
            var labelsDouble = Tensor<T>.FromMatrix(trueLabelsBatch).Cast<double>();
            var stuHard = Engine.Softmax(studentDouble, 1);
            var hardGrad = Engine.TensorSubtract(stuHard, labelsDouble);
            totalGrad = Engine.TensorAdd(
                Engine.TensorMultiplyScalar<double>(hardGrad, Alpha),
                Engine.TensorMultiplyScalar<double>(softGrad, 1.0 - Alpha));
        }
        else
        {
            totalGrad = softGrad;
        }

        return Engine.TensorDivideScalar(totalGrad, rows).Cast<T>().ToMatrix();
    }

    private static Tensor<double> BlendByMask(Tensor<double> unitMask, Tensor<double> whenOne, Tensor<double> whenZero)
    {
        var oneMinusMask = Engine.TensorAddScalar(Engine.TensorNegate(unitMask), 1.0);
        return Engine.TensorAdd(
            Engine.TensorMultiply(unitMask, whenOne),
            Engine.TensorMultiply(oneMinusMask, whenZero));
    }

    private void ValidateSparseTargets(Matrix<T> studentBatchOutput, Tensor<int> topIndices, Matrix<T> topProbabilities)
    {
        if (studentBatchOutput is null) throw new ArgumentNullException(nameof(studentBatchOutput));
        if (topIndices is null) throw new ArgumentNullException(nameof(topIndices));
        if (topProbabilities is null) throw new ArgumentNullException(nameof(topProbabilities));
        int rows = studentBatchOutput.Rows;
        int cols = studentBatchOutput.Columns;
        if (rows <= 0 || cols <= 0)
            throw new ArgumentException("Student output must have positive dimensions.", nameof(studentBatchOutput));
        if (topIndices.Rank != 2 || topIndices.Shape[0] != rows || topProbabilities.Rows != rows)
            throw new ArgumentException("Sparse Top-K target batch size must match student batch size.");
        int k = topProbabilities.Columns;
        if (k <= 0 || k > cols || topIndices.Shape[1] != k)
            throw new ArgumentException("Sparse Top-K width must be in [1, numClasses] and match topIndices.");
        int minIdx = Engine.TensorMinValue(topIndices);
        int maxIdx = Engine.TensorMaxValue(topIndices);
        if (minIdx < 0 || maxIdx >= cols)
            throw new ArgumentOutOfRangeException(nameof(topIndices), $"Top-K class indices [{minIdx}, {maxIdx}] must lie within [0, {cols}).");
    }
}
