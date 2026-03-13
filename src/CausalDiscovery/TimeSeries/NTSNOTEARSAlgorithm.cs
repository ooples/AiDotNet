using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// NTS-NOTEARS — Nonstationary Time Series NOTEARS.
/// </summary>
/// <remarks>
/// <para>
/// NTS-NOTEARS extends DYNOTEARS to handle nonstationary time series where the causal
/// structure may change over time. It partitions the data into segments using a variance-based
/// change-point detector, learns a separate NOTEARS-style DAG for each segment, and produces
/// a summary graph by taking the maximum absolute edge weight across segments, preserving
/// regime-specific causal effects.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Partition time series into K segments using variance-based change-point detection</item>
/// <item>For each segment, learn a contemporaneous DAG via NOTEARS with DYNOTEARS-style
///   lagged terms and the augmented Lagrangian acyclicity constraint</item>
/// <item>Aggregate segment-level DAGs into a summary graph: edge weight = max |w_k| over
///   segments where that edge appears, preserving regime-specific causal effects</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular time series methods assume the causal relationships stay
/// the same forever. NTS-NOTEARS can detect when relationships change — for example,
/// a market regime shift where the causes of stock prices change.
/// </para>
/// <para>
/// Reference: Sun et al. (2021), "NTS-NOTEARS: Learning Nonparametric DBN Structure
/// from Nonstationary Time Series".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("NTS-NOTEARS: Learning Nonparametric DBN Structure from Nonstationary Time Series", "https://doi.org/10.48550/arXiv.2109.04286", Year = 2021, Authors = "Xiangyu Sun, Guiliang Liu, Pascal Poupart, Oliver Schulte")]
public class NTSNOTEARSAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NTS-NOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    private readonly double _lambda1;
    private readonly double _wThreshold;
    private readonly int _maxSegments;
    private readonly int _maxOuterIterations;
    private readonly int _maxInnerSteps;

    public NTSNOTEARSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _lambda1 = options?.SparsityPenalty ?? 0.1;
        _wThreshold = options?.EdgeThreshold ?? 0.3;
        // MaxSegments is a separate concept from optimizer iterations
        _maxSegments = Math.Max(2, Math.Min(options?.MaxParents ?? 3, 5));
        // MaxIterations controls the actual optimization convergence
        _maxOuterIterations = Math.Max(1, options?.MaxIterations ?? 20);
        _maxInnerSteps = 100;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - MaxLag;

        if (d < 2)
            throw new ArgumentException($"NTS-NOTEARS requires at least 2 variables, got {d}.");
        if (effectiveN < 2 * d + 3)
            throw new ArgumentException($"NTS-NOTEARS requires at least {2 * d + 3 + MaxLag} samples for {d} variables with lag {MaxLag}, got {n}.");

        // Phase 1: Change-point detection using rolling variance ratio
        var changePoints = DetectChangePoints(data, n, d);

        // Phase 2: Learn structure for each segment
        var segmentGraphs = new List<Matrix<T>>();
        var segmentLengths = new List<int>();

        int prevEnd = 0;
        for (int seg = 0; seg <= changePoints.Count; seg++)
        {
            int segEnd = seg < changePoints.Count ? changePoints[seg] : n;
            int segLen = segEnd - prevEnd;

            if (segLen >= MaxLag + d + 3)
            {
                // Extract segment data
                var segData = new Matrix<T>(segLen, d);
                for (int t = 0; t < segLen; t++)
                    for (int j = 0; j < d; j++)
                        segData[t, j] = data[prevEnd + t, j];

                // Learn structure for this segment using DYNOTEARS
                var segGraph = LearnSegmentStructure(segData, segLen, d);
                segmentGraphs.Add(segGraph);
                segmentLengths.Add(segLen);
            }

            prevEnd = segEnd;
        }

        if (segmentGraphs.Count == 0)
            return new Matrix<T>(d, d);

        // Phase 3: Aggregate — take max |w_k| across segments
        var result = new Matrix<T>(d, d);
        for (int sg = 0; sg < segmentGraphs.Count; sg++)
        {
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    T absNew = NumOps.Abs(segmentGraphs[sg][i, j]);
                    T absCur = NumOps.Abs(result[i, j]);
                    if (NumOps.GreaterThan(absNew, absCur))
                        result[i, j] = segmentGraphs[sg][i, j];
                }
            }
        }

        return result;
    }

    private List<int> DetectChangePoints(Matrix<T> data, int n, int d)
    {
        var changePoints = new List<int>();
        int windowSize = Math.Max(n / (_maxSegments * 2), MaxLag + d + 3);
        if (windowSize >= n / 2) return changePoints;

        // Compute rolling variance ratio using Engine-accelerated dot products
        var varianceRatios = new List<double>();
        for (int t = windowSize; t <= n - windowSize; t++)
        {
            T totalRatio = NumOps.Zero;
            for (int col = 0; col < d; col++)
            {
                // Left window variance
                var leftVec = new Vector<T>(windowSize);
                T leftSum = NumOps.Zero;
                for (int s = 0; s < windowSize; s++)
                {
                    leftVec[s] = data[t - windowSize + s, col];
                    leftSum = NumOps.Add(leftSum, leftVec[s]);
                }
                T leftMean = NumOps.Divide(leftSum, NumOps.FromDouble(windowSize));
                for (int s = 0; s < windowSize; s++)
                    leftVec[s] = NumOps.Subtract(leftVec[s], leftMean);
                T leftVar = Engine.DotProduct(leftVec, leftVec);

                // Right window variance
                var rightVec = new Vector<T>(windowSize);
                T rightSum = NumOps.Zero;
                for (int s = 0; s < windowSize; s++)
                {
                    rightVec[s] = data[t + s, col];
                    rightSum = NumOps.Add(rightSum, rightVec[s]);
                }
                T rightMean = NumOps.Divide(rightSum, NumOps.FromDouble(windowSize));
                for (int s = 0; s < windowSize; s++)
                    rightVec[s] = NumOps.Subtract(rightVec[s], rightMean);
                T rightVar = Engine.DotProduct(rightVec, rightVec);

                // Variance ratio (log ratio to handle scale differences)
                double dLeft = Math.Max(NumOps.ToDouble(leftVar), 1e-15);
                double dRight = Math.Max(NumOps.ToDouble(rightVar), 1e-15);
                totalRatio = NumOps.Add(totalRatio, NumOps.FromDouble(Math.Abs(Math.Log(dRight / dLeft))));
            }
            varianceRatios.Add(NumOps.ToDouble(totalRatio));
        }

        if (varianceRatios.Count == 0) return changePoints;

        // Find peaks in variance ratio that exceed threshold
        double mean = varianceRatios.Average();
        double std = Math.Sqrt(varianceRatios.Sum(v => (v - mean) * (v - mean)) / varianceRatios.Count);
        double peakThreshold = mean + 2.0 * std;

        int minGap = windowSize;
        int lastCP = -minGap;
        for (int i = 1; i < varianceRatios.Count - 1 && changePoints.Count < _maxSegments - 1; i++)
        {
            int absT = windowSize + i;
            if (varianceRatios[i] > peakThreshold &&
                varianceRatios[i] > varianceRatios[i - 1] &&
                varianceRatios[i] > varianceRatios[i + 1] &&
                absT - lastCP >= minGap)
            {
                changePoints.Add(absT);
                lastCP = absT;
            }
        }

        return changePoints;
    }

    private Matrix<T> LearnSegmentStructure(Matrix<T> segData, int n, int d)
    {
        // Standardize segment
        T nT = NumOps.FromDouble(n);
        var stdData = new Matrix<T>(n, d);
        for (int j = 0; j < d; j++)
        {
            T mean = NumOps.Zero;
            for (int t = 0; t < n; t++)
                mean = NumOps.Add(mean, segData[t, j]);
            mean = NumOps.Divide(mean, nT);

            T variance = NumOps.Zero;
            for (int t = 0; t < n; t++)
            {
                T diff = NumOps.Subtract(segData[t, j], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            T std = NumOps.Sqrt(NumOps.Add(NumOps.Divide(variance, nT), NumOps.FromDouble(1e-15)));

            for (int t = 0; t < n; t++)
                stdData[t, j] = NumOps.Divide(NumOps.Subtract(segData[t, j], mean), std);
        }

        int effectiveN = n - MaxLag;
        if (effectiveN < d + 3) return new Matrix<T>(d, d);

        // Build contemporaneous and lagged matrices
        var Xt = new Matrix<T>(effectiveN, d);
        for (int t = 0; t < effectiveN; t++)
            for (int j = 0; j < d; j++)
                Xt[t, j] = stdData[t + MaxLag, j];

        int lagDim = d * MaxLag;
        var Z = new Matrix<T>(effectiveN, lagDim);
        for (int t = 0; t < effectiveN; t++)
            for (int lag = 1; lag <= MaxLag; lag++)
            {
                int colOffset = (lag - 1) * d;
                for (int j = 0; j < d; j++)
                    Z[t, colOffset + j] = stdData[t + MaxLag - lag, j];
            }

        // DYNOTEARS-style: optimize contemporaneous W and lagged A matrices
        // Loss = 1/(2n) * ||Xt - Xt*W - Z*A||_F^2 + lambda*||W||_1
        // subject to h(W) = tr(exp(W∘W)) - d = 0
        var W = new Matrix<T>(d, d);
        var A = new Matrix<T>(lagDim, d); // Lagged coefficients
        double rho = 1.0, alpha = 0.0;
        T lr = NumOps.FromDouble(1e-3);
        T lambda = NumOps.FromDouble(_lambda1);
        T invN = NumOps.FromDouble(1.0 / effectiveN);

        for (int outer = 0; outer < _maxOuterIterations; outer++)
        {
            // Inner gradient descent
            for (int step = 0; step < _maxInnerSteps; step++)
            {
                var gradW = new Matrix<T>(d, d);
                var gradA = new Matrix<T>(lagDim, d);

                // Compute residual and gradients for each target column j
                for (int j = 0; j < d; j++)
                {
                    // Residual: r_j = Xt[:,j] - Xt*W[:,j] - Z*A[:,j]
                    var residCol = new Vector<T>(effectiveN);
                    for (int t = 0; t < effectiveN; t++)
                    {
                        T pred = NumOps.Zero;
                        // Contemporaneous: Xt * W[:,j]
                        for (int k = 0; k < d; k++)
                            pred = NumOps.Add(pred, NumOps.Multiply(Xt[t, k], W[k, j]));
                        // Lagged: Z * A[:,j]
                        for (int l = 0; l < lagDim; l++)
                            pred = NumOps.Add(pred, NumOps.Multiply(Z[t, l], A[l, j]));
                        residCol[t] = NumOps.Subtract(Xt[t, j], pred);
                    }

                    // Gradient for W: -1/n * Xt' * residual
                    for (int i = 0; i < d; i++)
                    {
                        var xiCol = new Vector<T>(effectiveN);
                        for (int t = 0; t < effectiveN; t++)
                            xiCol[t] = Xt[t, i];
                        T dot = Engine.DotProduct(xiCol, residCol);
                        gradW[i, j] = NumOps.Negate(NumOps.Multiply(invN, dot));
                    }

                    // Gradient for A: -1/n * Z' * residual
                    for (int l = 0; l < lagDim; l++)
                    {
                        var zCol = new Vector<T>(effectiveN);
                        for (int t = 0; t < effectiveN; t++)
                            zCol[t] = Z[t, l];
                        T dot = Engine.DotProduct(zCol, residCol);
                        gradA[l, j] = NumOps.Negate(NumOps.Multiply(invN, dot));
                    }
                }

                // Add acyclicity gradient for W: (alpha + rho*h) * [exp(W∘W)^T ∘ 2W]
                double h = ComputeTraceExpWoW(W, d);
                T constraintMult = NumOps.FromDouble(alpha + rho * h);
                T two = NumOps.FromDouble(2.0);
                var expWoW = ComputeExpWoW(W, d);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                    {
                        // Use transposed expWoW: expWoW[j,i] not expWoW[i,j]
                        T acGrad = NumOps.Multiply(constraintMult,
                            NumOps.Multiply(two, NumOps.Multiply(expWoW[j, i], W[i, j])));
                        T l1Grad = NumOps.Multiply(lambda, NumOps.FromDouble(Math.Sign(NumOps.ToDouble(W[i, j]))));
                        gradW[i, j] = NumOps.Add(gradW[i, j], NumOps.Add(acGrad, l1Grad));
                    }

                // Update W and A
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++)
                        W[i, j] = NumOps.Subtract(W[i, j], NumOps.Multiply(lr, gradW[i, j]));
                for (int l = 0; l < lagDim; l++)
                    for (int j = 0; j < d; j++)
                        A[l, j] = NumOps.Subtract(A[l, j], NumOps.Multiply(lr, gradA[l, j]));

                // Zero diagonal of W
                for (int i = 0; i < d; i++)
                    W[i, i] = NumOps.Zero;
            }

            double hVal = ComputeTraceExpWoW(W, d);
            if (hVal < 1e-8) break;
            alpha += rho * hVal;
            rho = Math.Min(rho * 10.0, 1e16);
        }

        // Threshold and return
        var result = new Matrix<T>(d, d);
        T thresh = NumOps.FromDouble(_wThreshold);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (NumOps.GreaterThan(NumOps.Abs(W[i, j]), thresh))
                    result[i, j] = W[i, j];

        return result;
    }

    private double ComputeTraceExpWoW(Matrix<T> W, int d)
    {
        // tr(e^{W∘W}) - d via Taylor series with MatMul acceleration
        var WoW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                WoW[i, j] = NumOps.Multiply(W[i, j], W[i, j]);

        double trace = 0;
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++) power[i, i] = NumOps.One;

        double factorial = 1.0;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            factorial *= k;
            power = MatMul(power, WoW);
            double tr = 0;
            for (int i = 0; i < d; i++) tr += NumOps.ToDouble(power[i, i]);
            trace += tr / factorial;
        }

        return trace;
    }

    private Matrix<T> ComputeExpWoW(Matrix<T> W, int d)
    {
        var WoW = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                WoW[i, j] = NumOps.Multiply(W[i, j], W[i, j]);

        var result = new Matrix<T>(d, d);
        var power = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
        {
            result[i, i] = NumOps.One;
            power[i, i] = NumOps.One;
        }

        double factorial = 1.0;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            factorial *= k;
            T invFact = NumOps.FromDouble(1.0 / factorial);
            power = MatMul(power, WoW);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    result[i, j] = NumOps.Add(result[i, j], NumOps.Multiply(power[i, j], invFact));
        }

        return result;
    }
}
