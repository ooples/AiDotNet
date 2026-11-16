using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Runs a default self-supervised pretraining loop over unlabeled data with representation-collapse
/// monitoring and an optional linear-probe quality estimate.
/// </summary>
internal static class SelfSupervisedLearningPretrainer
{
    /// <summary>
    /// Pretrains <paramref name="method"/> on <paramref name="data"/> for the given epochs, watching for
    /// representation collapse and (when <paramref name="targets"/> is supplied) probing representation
    /// quality with a ridge linear model.
    /// </summary>
    public static SelfSupervisedLearningPretrainingResult<T> Run<T>(
        ISelfSupervisedLearningMethod<T> method,
        Tensor<T> data,
        Vector<T>? targets,
        int epochs,
        int batchSize,
        double collapseThreshold = 1e-3)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = data.Shape[0];
        int d = data.Length / n;
        int b = Math.Max(1, Math.Min(batchSize, n));

        var epochLosses = new List<double>(epochs);
        int steps = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double epochLoss = 0;
            int batches = 0;
            for (int start = 0; start < n; start += b)
            {
                int count = Math.Min(b, n - start);
                var batch = Slice(data, start, count, d, numOps);
                var stepResult = method.TrainStep(batch);
                epochLoss += numOps.ToDouble(stepResult.Loss);
                batches++;
                steps++;
            }

            epochLosses.Add(batches > 0 ? epochLoss / batches : 0.0);
        }

        // Collapse check: spread of the encoder's representations across a batch.
        double repStd = RepresentationStdDev(method, Slice(data, 0, Math.Min(b, n), d, numOps), numOps);
        bool collapsed = repStd < collapseThreshold;

        // Linear probe: how well the learned representations linearly predict the targets (in-sample).
        double? probeR2 = targets is null ? null : LinearProbeR2(method, data, targets, numOps);

        return new SelfSupervisedLearningPretrainingResult<T>
        {
            MethodName = method.Name,
            EpochsRun = epochs,
            StepsRun = steps,
            FinalLoss = epochLosses.Count > 0 ? epochLosses[^1] : 0.0,
            EpochLosses = epochLosses,
            CollapseDetected = collapsed,
            RepresentationStdDev = repStd,
            LinearProbeR2 = probeR2,
        };
    }

    private static Tensor<T> Slice<T>(Tensor<T> data, int start, int count, int d, INumericOperations<T> numOps)
    {
        // Rows [start, start + count) of the data viewed as [n, d]
        var batch = new Tensor<T>(new[] { count, d });
        data.Contiguous().AsSpan().Slice(start * d, count * d).CopyTo(batch.AsWritableSpan());

        return batch;
    }

    private static double RepresentationStdDev<T>(ISelfSupervisedLearningMethod<T> method, Tensor<T> batch, INumericOperations<T> numOps)
    {
        var repr = method.Encode(batch);
        int rows = repr.Shape[0];
        int k = repr.Length / rows;
        if (rows < 2 || k == 0)
        {
            return 0.0;
        }

        return FeatureStdDev(repr.Contiguous().Reshape(rows, k));
    }

    /// <summary>
    /// Mean over features of the unbiased per-feature standard deviation of a [rows, k] matrix, computed in
    /// double as master did: in Half the squared deviations overflow past 65504.
    /// </summary>
    internal static double FeatureStdDev<T>(Tensor<T> matrix)
    {
        int rows = matrix.Shape[0];
        var engine = AiDotNetEngine.Current;
        var values = matrix.Cast<double>();
        var centered = engine.TensorSubtract(values, engine.ReduceMean(values, new[] { 0 }, keepDims: true));
        var variance = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: false), (double)(rows - 1));

        return engine.TensorMean(engine.TensorSqrt(variance));
    }

    private static double LinearProbeR2<T>(ISelfSupervisedLearningMethod<T> method, Tensor<T> data, Vector<T> targets, INumericOperations<T> numOps)
    {
        int n = data.Shape[0];
        int d = data.Length / n;
        var features = method.Encode(Slice(data, 0, n, d, numOps));
        int k = features.Length / n;
        if (k == 0 || n < 2)
        {
            return 0.0;
        }

        return RidgeR2(features.Contiguous().Reshape(n, k), targets.AsTensor().Reshape(targets.Length, 1));
    }

    /// <summary>
    /// R² of a ridge fit (λ = 1e-3, with a bias column) of the first n targets on [n, k] features. Fitted
    /// in double as master did: in Half the normal equations FᵀF overflow past 65504.
    /// </summary>
    internal static double RidgeR2<T>(Tensor<T> features, Tensor<T> targets)
    {
        int n = features.Shape[0];
        int k = features.Shape[1];
        var engine = AiDotNetEngine.Current;

        // Design matrix with a bias column: F[n, k+1].
        int p = k + 1;
        var ones = engine.TensorAddScalar(new Tensor<double>(new[] { n, 1 }), 1.0);
        var f = engine.TensorConcatenate(new[] { features.Cast<double>(), ones }, 1);
        var y = engine.TensorSlice(targets.Cast<double>(), new[] { 0, 0 }, new[] { n, 1 });

        // Ridge normal equations: (FᵀF + λI) w = Fᵀy, solved by LU with partial pivoting.
        const double lambda = 1e-3;
        var fT = engine.TensorTranspose(f);
        var ata = engine.TensorAdd(
            engine.TensorMatMul(fT, f), engine.TensorMultiplyScalar(engine.TensorEye<double>(p), lambda));
        var aty = engine.TensorMatMul(fT, y);

        var (w, info) = LinalgDispatch<double>.SolveEx(ata, aty);
        if (engine.TensorCountNonzero(info) != 0)
        {
            return 0.0;
        }

        double meanY = engine.TensorMean(y);
        var residual = engine.TensorSubtract(y, engine.TensorMatMul(f, w));
        var deviation = engine.TensorAddScalar(y, -meanY);
        double ssRes = engine.TensorSum(engine.TensorMultiply(residual, residual));
        double ssTot = engine.TensorSum(engine.TensorMultiply(deviation, deviation));

        return ssTot > 0 ? 1.0 - (ssRes / ssTot) : 0.0;
    }
}
