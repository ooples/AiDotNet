using AiDotNet.Helpers;
using AiDotNet.Tensors;
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
        var batch = new Tensor<T>(new[] { count, d });
        for (int i = 0; i < count; i++)
        {
            int baseIdx = (start + i) * d;
            for (int j = 0; j < d; j++)
            {
                batch[(i * d) + j] = data[baseIdx + j];
            }
        }

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

        double total = 0;
        for (int j = 0; j < k; j++)
        {
            double mean = 0;
            for (int i = 0; i < rows; i++) mean += numOps.ToDouble(repr[(i * k) + j]);
            mean /= rows;

            double var = 0;
            for (int i = 0; i < rows; i++)
            {
                double v = numOps.ToDouble(repr[(i * k) + j]) - mean;
                var += v * v;
            }

            total += Math.Sqrt(var / (rows - 1));
        }

        return total / k;
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

        // Design matrix with a bias column: F[n, k+1].
        int p = k + 1;
        var f = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++) f[i, j] = numOps.ToDouble(features[(i * k) + j]);
            f[i, k] = 1.0;
        }

        var y = new double[n];
        for (int i = 0; i < n; i++) y[i] = numOps.ToDouble(targets[i]);

        // Ridge normal equations: (FᵀF + λI) w = Fᵀy.
        const double lambda = 1e-3;
        var ata = new double[p, p];
        var aty = new double[p];
        for (int r = 0; r < p; r++)
        {
            for (int c = 0; c < p; c++)
            {
                double s = 0;
                for (int i = 0; i < n; i++) s += f[i, r] * f[i, c];
                ata[r, c] = s + (r == c ? lambda : 0.0);
            }

            double sy = 0;
            for (int i = 0; i < n; i++) sy += f[i, r] * y[i];
            aty[r] = sy;
        }

        var w = SolveSymmetric(ata, aty, p);
        if (w is null)
        {
            return 0.0;
        }

        double meanY = 0;
        for (int i = 0; i < n; i++) meanY += y[i];
        meanY /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            for (int j = 0; j < p; j++) pred += f[i, j] * w[j];
            ssRes += (y[i] - pred) * (y[i] - pred);
            ssTot += (y[i] - meanY) * (y[i] - meanY);
        }

        return ssTot > 0 ? 1.0 - (ssRes / ssTot) : 0.0;
    }

    /// <summary>Solves A w = b for a small symmetric positive-definite A via Gaussian elimination.</summary>
    private static double[]? SolveSymmetric(double[,] a, double[] b, int p)
    {
        var m = new double[p, p + 1];
        for (int r = 0; r < p; r++)
        {
            for (int c = 0; c < p; c++) m[r, c] = a[r, c];
            m[r, p] = b[r];
        }

        for (int col = 0; col < p; col++)
        {
            int pivot = col;
            double best = Math.Abs(m[col, col]);
            for (int r = col + 1; r < p; r++)
            {
                if (Math.Abs(m[r, col]) > best) { best = Math.Abs(m[r, col]); pivot = r; }
            }

            if (best < 1e-12)
            {
                return null;
            }

            if (pivot != col)
            {
                for (int c = 0; c <= p; c++) (m[col, c], m[pivot, c]) = (m[pivot, c], m[col, c]);
            }

            for (int r = 0; r < p; r++)
            {
                if (r == col) continue;
                double factor = m[r, col] / m[col, col];
                for (int c = col; c <= p; c++) m[r, c] -= factor * m[col, c];
            }
        }

        var w = new double[p];
        for (int r = 0; r < p; r++) w[r] = m[r, p] / m[r, r];
        return w;
    }
}
