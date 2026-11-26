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
/// <c>{top_1, ..., top_K, Tail}</c> using vectorized <see cref="IEngine"/> log-domain tensor operations, preserving
/// the teacher's un-renormalized full-vocabulary top-<c>K</c> probabilities <c>q_{T,j} = softmax_V(z_T / T)[idx_j]</c>
/// together with the exact remainder tail mass <c>q_{T,tail} = 1 - sum_{j=1}^K q_{T,j}</c> matched against the
/// log-sum-exp of unselected student log-probabilities. By the data-processing inequality,
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
        var (qTop, _, isTopMaskBinary) = ValidateAndPrepareSparseTargets(studentBatchOutput, topIndices, topProbabilities);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int rows = studentBatchOutput.Rows;
        int cols = studentBatchOutput.Columns;
        int k = topProbabilities.Columns;

        var studentDouble = Tensor<T>.FromMatrix(studentBatchOutput).Cast<double>();
        var lsm = Engine.TensorLogSoftmax(Engine.TensorDivideScalar(studentDouble, Temperature), 1);

        var rowOffsets = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<int>(0, rows * cols, cols), [rows, 1]),
            [rows, k]);
        var flatIdx = Engine.Reshape(Engine.TensorAdd(rowOffsets, topIndices), [rows * k]);
        var stuTopLog = Engine.Reshape(
            Engine.TensorGather(Engine.Reshape(lsm, [rows * cols]), flatIdx, 0),
            [rows, k]);

        var zeroTop = Tensor<double>.CreateDefault([rows, k], 0.0);
        var oneTop = Tensor<double>.CreateDefault([rows, k], 1.0);
        var posTopMask = Engine.TensorGreaterThan(qTop, zeroTop);
        var qTopForLog = BlendByMask(posTopMask, qTop, oneTop);
        var topKl = Engine.TensorMultiply(qTop, Engine.TensorSubtract(Engine.TensorLog(qTopForLog), stuTopLog));
        var sumTopKl = Engine.ReduceSum(topKl, [1], keepDims: true);

        Tensor<double> rowKl;
        if (k == cols)
        {
            rowKl = sumTopKl;
        }
        else
        {
            var mTeacher = Engine.ReduceSum(qTop, [1], keepDims: true);
            var zeroRow = Tensor<double>.CreateDefault([rows, 1], 0.0);
            var oneRow = Tensor<double>.CreateDefault([rows, 1], 1.0);
            var qTail = Engine.TensorMax(Engine.TensorAddScalar(Engine.TensorNegate(mTeacher), 1.0), zeroRow);
            var (logTailStu, _) = ComputeUnselectedTailDistribution(lsm, isTopMaskBinary, rows, cols);

            var posTailMask = Engine.TensorGreaterThan(qTail, zeroRow);
            var qTailForLog = BlendByMask(posTailMask, qTail, oneRow);
            var tailKl = Engine.TensorMultiply(qTail, Engine.TensorSubtract(Engine.TensorLog(qTailForLog), logTailStu));
            rowKl = Engine.TensorAdd(sumTopKl, tailKl);
        }

        Tensor<double> rowSoftLoss = Engine.TensorMultiplyScalar<double>(rowKl, Temperature * Temperature);

        Tensor<double> rowTotalLoss;
        if (trueLabelsBatch is not null)
        {
            var labelsDouble = Tensor<T>.FromMatrix(trueLabelsBatch).Cast<double>();
            var stuHardLog = Engine.TensorLogSoftmax(studentDouble, 1);
            var hardCeTerms = Engine.TensorNegate(Engine.TensorMultiply(labelsDouble, stuHardLog));
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
        var (qTop, qTopFull, isTopMaskBinary) = ValidateAndPrepareSparseTargets(studentBatchOutput, topIndices, topProbabilities);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int rows = studentBatchOutput.Rows;
        int cols = studentBatchOutput.Columns;
        int k = topProbabilities.Columns;

        var studentDouble = Tensor<T>.FromMatrix(studentBatchOutput).Cast<double>();
        var lsm = Engine.TensorLogSoftmax(Engine.TensorDivideScalar(studentDouble, Temperature), 1);
        var stuSoft = Engine.TensorExp(lsm);

        var mTeacher = Engine.ReduceSum(qTop, [1], keepDims: true);

        Tensor<double> gradUnits;
        if (k == cols)
        {
            var scaledStu = Engine.TensorMultiply(stuSoft, Engine.TensorBroadcastTo(mTeacher, [rows, cols]));
            gradUnits = Engine.TensorSubtract(scaledStu, qTopFull);
        }
        else
        {
            var zeroRow = Tensor<double>.CreateDefault([rows, 1], 0.0);
            var qTail = Engine.TensorMax(Engine.TensorAddScalar(Engine.TensorNegate(mTeacher), 1.0), zeroRow);
            var totalMass = Engine.TensorAdd(mTeacher, qTail);
            var (_, condTailProb) = ComputeUnselectedTailDistribution(lsm, isTopMaskBinary, rows, cols);

            var tailTargetFull = Engine.TensorMultiply(Engine.TensorBroadcastTo(qTail, [rows, cols]), condTailProb);
            var totalTargetFull = Engine.TensorAdd(qTopFull, tailTargetFull);
            var scaledStu = Engine.TensorMultiply(stuSoft, Engine.TensorBroadcastTo(totalMass, [rows, cols]));
            gradUnits = Engine.TensorSubtract(scaledStu, totalTargetFull);
        }

        Tensor<double> softGrad = Engine.TensorMultiplyScalar<double>(gradUnits, Temperature);

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

    private static (Tensor<double> LogTailStu, Tensor<double> CondTailProb) ComputeUnselectedTailDistribution(
        Tensor<double> lsm,
        Tensor<double> isTopMaskBinary,
        int rows,
        int cols)
    {
        var tailMask = Engine.TensorAddScalar(Engine.TensorNegate(isTopMaskBinary), 1.0);
        var minLsm = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(lsm), [1], keepDims: true));
        var belowMin = Engine.TensorBroadcastTo(Engine.TensorAddScalar(minLsm, -1.0), [rows, cols]);
        var lsmForMax = BlendByMask(isTopMaskBinary, belowMin, lsm);
        var maxTailLog = Engine.ReduceMax(lsmForMax, [1], keepDims: true);

        var shiftedForExp = Engine.TensorSubtract(lsmForMax, Engine.TensorBroadcastTo(maxTailLog, [rows, cols]));
        var tailExp = Engine.TensorMultiply(tailMask, Engine.TensorExp(shiftedForExp));
        var sumTailExp = Engine.ReduceSum(tailExp, [1], keepDims: true);

        var logTailStu = Engine.TensorAdd(maxTailLog, Engine.TensorLog(sumTailExp));
        var condTailProb = Engine.TensorDivide(tailExp, Engine.TensorBroadcastTo(sumTailExp, [rows, cols]));
        return (logTailStu, condTailProb);
    }

    private static Tensor<double> BlendByMask(Tensor<double> unitMask, Tensor<double> whenOne, Tensor<double> whenZero)
    {
        var oneMinusMask = Engine.TensorAddScalar(Engine.TensorNegate(unitMask), 1.0);
        return Engine.TensorAdd(
            Engine.TensorMultiply(unitMask, whenOne),
            Engine.TensorMultiply(oneMinusMask, whenZero));
    }

    private (Tensor<double> QTop, Tensor<double> QTopFull, Tensor<double> IsTopMaskBinary) ValidateAndPrepareSparseTargets(
        Matrix<T> studentBatchOutput,
        Tensor<int> topIndices,
        Matrix<T> topProbabilities)
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

        // Index-based [rows, cols] scatter instead of a dense [rows, k, cols] one-hot expansion:
        // memory stays O(rows * cols) regardless of K, which matters for large vocabularies.
        var rowOffsets = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<int>(0, rows * cols, cols), [rows, 1]),
            [rows, k]);
        var flatIdx = Engine.Reshape(Engine.TensorAdd(rowOffsets, topIndices), [rows * k]);
        var isTopCount = Engine.Reshape(
            Engine.TensorScatterAdd(
                Tensor<double>.CreateDefault([rows * cols], 0.0),
                flatIdx,
                Tensor<double>.CreateDefault([rows * k], 1.0),
                0),
            [rows, cols]);
        if (Engine.TensorMaxValue(isTopCount) > 1.5)
            throw new ArgumentException("Each row of topIndices must contain distinct class indices.", nameof(topIndices));

        var qTop = Tensor<T>.FromMatrix(topProbabilities).Cast<double>();
        double qAbsSum = Engine.TensorSum(Engine.TensorAbs(qTop));
        if (double.IsNaN(qAbsSum) || double.IsInfinity(qAbsSum))
            throw new ArgumentException("Sparse Top-K probabilities must be finite values in [0, 1].", nameof(topProbabilities));

        double minProb = Engine.TensorMinValue(qTop);
        double maxProb = Engine.TensorMaxValue(qTop);
        if (minProb < 0.0 || maxProb > 1.0)
            throw new ArgumentException("Sparse Top-K probabilities must lie within [0, 1].", nameof(topProbabilities));

        var rowSums = Engine.ReduceSum(qTop, [1], keepDims: false);
        double maxRowSum = Engine.TensorMaxValue(rowSums);
        double rowSumTolerance = typeof(T) == typeof(double)
            ? Math.Max(1e-9, k * 1e-14)
            : Math.Max(1e-5, k * 1e-6);
        if (maxRowSum > 1.0 + rowSumTolerance)
            throw new ArgumentException(
                $"Each row of topProbabilities must sum to at most 1 (max row sum was {maxRowSum:G9}).",
                nameof(topProbabilities));

        // When K covers every class there is no tail bucket to absorb missing mass, so each row
        // must be a complete distribution; sub-unit selected mass is only valid when k < cols.
        if (k == cols)
        {
            double minRowSum = Engine.TensorMinValue(rowSums);
            if (minRowSum < 1.0 - rowSumTolerance)
                throw new ArgumentException(
                    $"Each row of topProbabilities must sum to 1 when K covers all {cols} classes (min row sum was {minRowSum:G9}).",
                    nameof(topProbabilities));
        }

        // Scatter the Top-K probabilities into their [rows, cols] slots (indices are distinct per row).
        var qTopFull = Engine.Reshape(
            Engine.TensorScatterAdd(
                Tensor<double>.CreateDefault([rows * cols], 0.0),
                flatIdx,
                Engine.Reshape(qTop, [rows * k]),
                0),
            [rows, cols]);
        var isTopMaskBinary = Engine.TensorGreaterThan(isTopCount, Tensor<double>.CreateDefault([rows, cols], 0.5));
        return (qTop, qTopFull, isTopMaskBinary);
    }
}
