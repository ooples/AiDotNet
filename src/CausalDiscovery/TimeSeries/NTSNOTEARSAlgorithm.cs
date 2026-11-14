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
[ResearchPaper("NTS-NOTEARS: Learning Nonparametric DBN Structure from Nonstationary Time Series", "https://doi.org/10.48550/arXiv.2109.04286", Year = 2021, Authors = "Xiangyu Sun, Guiliang Liu, Pascal Poupart, Oliver Schulte")]
public class NTSNOTEARSAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NTS-NOTEARS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    private const int DefaultMaxSegments = 3;
    private const int MinSegments = 2;
    private const int MaxSegmentsCap = 10;
    private const int DefaultInnerSteps = 100;

    private readonly double _lambda1;
    private readonly double _wThreshold;
    private readonly int _maxSegments;
    private readonly int _maxOuterIterations;
    private readonly int _maxInnerSteps;

    public NTSNOTEARSAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        _lambda1 = options?.SparsityPenalty ?? 0.1;
        _wThreshold = options?.EdgeThreshold ?? 0.1;
        _maxSegments = Math.Max(MinSegments, Math.Min(options?.MaxSegments ?? DefaultMaxSegments, MaxSegmentsCap));
        _maxOuterIterations = Math.Max(1, options?.MaxIterations ?? 20);
        _maxInnerSteps = options?.InnerIterations ?? DefaultInnerSteps;
        if (double.IsNaN(_lambda1) || double.IsInfinity(_lambda1) || _lambda1 < 0)
            throw new ArgumentException("SparsityPenalty must be a non-negative finite value.");
        if (double.IsNaN(_wThreshold) || double.IsInfinity(_wThreshold) || _wThreshold < 0)
            throw new ArgumentException("EdgeThreshold must be a non-negative finite value.");
        if (_maxInnerSteps < 1)
            throw new ArgumentException("InnerIterations must be at least 1.");
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
        var x = Tensor<T>.FromMatrix(data);
        var segmentGraphs = new List<Tensor<T>>();

        int prevEnd = 0;
        for (int seg = 0; seg <= changePoints.Count; seg++)
        {
            int segEnd = seg < changePoints.Count ? changePoints[seg] : n;
            int segLen = segEnd - prevEnd;

            if (segLen >= MaxLag + d + 3)
            {
                // Extract segment data
                var segData = Engine.TensorSlice(x, new[] { prevEnd, 0 }, new[] { segLen, d });

                // Learn structure for this segment using DYNOTEARS
                segmentGraphs.Add(LearnSegmentStructure(segData, segLen, d));
            }

            prevEnd = segEnd;
        }

        if (segmentGraphs.Count == 0)
            return new Matrix<T>(d, d);

        // Phase 3: Aggregate — take the entry with the largest |w_k| across segments (earliest segment on ties)
        var result = new Tensor<T>(new[] { d, d });
        foreach (var segmentGraph in segmentGraphs)
        {
            var larger = Engine.TensorGreaterThan(Engine.TensorAbs(segmentGraph), Engine.TensorAbs(result));
            result = Engine.TensorWhere(larger, segmentGraph, result);
        }

        return result.ToMatrix();
    }

    private List<int> DetectChangePoints(Matrix<T> data, int n, int d)
    {
        var changePoints = new List<int>();
        int windowSize = Math.Max(n / (_maxSegments * 2), MaxLag + d + 3);
        if (windowSize >= n / 2) return changePoints;

        // Rolling variance ratio: for each split t, Σ_col |log(max(SS_right, 1e-15) / max(SS_left, 1e-15))| where
        // SS is the centered sum of squares of the windowSize rows before / from t.
        var x = Tensor<T>.FromMatrix(data);
        T floor = NumOps.FromDouble(1e-15);
        var ratios = new List<Tensor<T>>();
        for (int t = windowSize; t <= n - windowSize; t++)
        {
            var leftVar = WindowSumOfSquares(x, t - windowSize, windowSize, floor);
            var rightVar = WindowSumOfSquares(x, t, windowSize, floor);
            var logRatio = Engine.TensorAbs(Engine.TensorLog(Engine.TensorDivide(rightVar, leftVar)));
            ratios.Add(Engine.ReduceSum(logRatio, new[] { 0 }, true));
        }

        if (ratios.Count == 0) return changePoints;
        var varianceRatios = Engine.TensorConcatenate(ratios.ToArray(), 0);
        int count = ratios.Count;
        if (count < 3) return changePoints;

        // Find peaks in variance ratio that exceed mean + 2 * std (population std)
        T mean = Engine.TensorMean(varianceRatios);
        var deviation = Engine.TensorSubtractScalar(varianceRatios, mean);
        T std = NumOps.Sqrt(Engine.TensorMean(Engine.TensorMultiply(deviation, deviation)));
        T peakThreshold = NumOps.Add(mean, NumOps.Multiply(NumOps.FromDouble(2.0), std));

        var middle = Engine.TensorSlice(varianceRatios, new[] { 1 }, new[] { count - 2 });
        var previous = Engine.TensorSlice(varianceRatios, new[] { 0 }, new[] { count - 2 });
        var next = Engine.TensorSlice(varianceRatios, new[] { 2 }, new[] { count - 2 });
        var isPeak = Engine.TensorMultiply(Engine.TensorMultiply(
            Engine.TensorGreaterThan(middle, peakThreshold), Engine.TensorGreaterThan(middle, previous)),
            Engine.TensorGreaterThan(middle, next));

        int minGap = windowSize;
        int lastCP = -minGap;
        foreach (int peak in Engine.TensorNonzero(isPeak).ToArray())
        {
            if (changePoints.Count >= _maxSegments - 1) break;
            int absT = windowSize + peak + 1;
            if (absT - lastCP >= minGap)
            {
                changePoints.Add(absT);
                lastCP = absT;
            }
        }

        return changePoints;
    }

    /// <summary>
    /// Per-column centered sum of squares of rows start..start+length-1, floored at <paramref name="floor"/>. Returns [d].
    /// </summary>
    private Tensor<T> WindowSumOfSquares(Tensor<T> x, int start, int length, T floor)
    {
        var window = CenterColumns(Engine.TensorSlice(x, new[] { start, 0 }, new[] { length, x.Shape[1] }));
        return Engine.TensorMax(Engine.ReduceSum(Engine.TensorMultiply(window, window), new[] { 0 }, false), floor);
    }

    private Tensor<T> LearnSegmentStructure(Tensor<T> segData, int n, int d)
    {
        // Standardize segment: (x - mean) / sqrt(var + 1e-15), population variance
        var centered = CenterColumns(segData);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(
            Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 0 }, true), NumOps.FromDouble(1e-15)));
        var stdData = Engine.TensorDivide(centered, Engine.TensorBroadcastTo(std, new[] { n, d }));

        int effectiveN = n - MaxLag;
        if (effectiveN < d + 3) return new Tensor<T>(new[] { d, d });

        // Build contemporaneous and lagged matrices: Z[:, (lag - 1) * d + j] = x_j(t - lag)
        var Xt = Engine.TensorSlice(stdData, new[] { MaxLag, 0 }, new[] { effectiveN, d });
        int lagDim = d * MaxLag;
        var Z = Engine.TensorConcatenate(Enumerable.Range(1, MaxLag)
            .Select(lag => Engine.TensorSlice(stdData, new[] { MaxLag - lag, 0 }, new[] { effectiveN, d }))
            .ToArray(), 1);
        var XtT = Engine.TensorTranspose(Xt);
        var ZT = Engine.TensorTranspose(Z);

        // DYNOTEARS-style: optimize contemporaneous W and lagged A matrices
        // Loss = 1/(2n) * ||Xt - Xt*W - Z*A||_F^2 + lambda*||W||_1
        // subject to h(W) = tr(exp(W∘W)) - d = 0
        var W = new Tensor<T>(new[] { d, d });
        var A = new Tensor<T>(new[] { lagDim, d }); // Lagged coefficients
        // Augmented-Lagrangian state as [1] tensors: rho = 1, alpha = 0
        var rho = Engine.TensorOnesLike(new Tensor<T>(new[] { 1 }));
        var alpha = new Tensor<T>(new[] { 1 });
        // For small problems, use higher learning rate
        T lr = NumOps.FromDouble(d <= 10 ? 0.01 : 1e-3);
        T lambda = NumOps.FromDouble(_lambda1);
        T negInvN = NumOps.FromDouble(-1.0 / effectiveN);
        T two = NumOps.FromDouble(2.0);
        T ten = NumOps.FromDouble(10.0);
        T rhoMax = NumOps.FromDouble(1e16);
        T converged = NumOps.FromDouble(1e-8);
        T notNaNBound = NumOps.FromDouble(-1.0);
        var offDiagonal = OffDiagonalMask(d);
        var zeros = new Tensor<T>(new[] { d, d });

        for (int outer = 0; outer < _maxOuterIterations; outer++)
        {
            // Inner gradient descent
            for (int step = 0; step < _maxInnerSteps; step++)
            {
                // Residual R = Xt - Xt*W - Z*A; gradients -1/n * Xt' R and -1/n * Z' R
                var residual = Engine.TensorSubtract(Engine.TensorSubtract(Xt, Engine.TensorMatMul(Xt, W)), Engine.TensorMatMul(Z, A));
                var gradW = Engine.TensorMultiplyScalar(Engine.TensorMatMul(XtT, residual), negInvN);
                var gradA = Engine.TensorMultiplyScalar(Engine.TensorMatMul(ZT, residual), negInvN);

                // sign(W) is undefined for NaN (Math.Sign throws), so a diverged W fails loudly.
                if (Engine.TensorCountNonzero(Engine.TensorGreaterThan(Engine.TensorAbs(W), notNaNBound)) < d * d)
                    throw new ArithmeticException("Function does not accept floating point Not-a-Number values.");

                // Add acyclicity gradient for W: (alpha + rho*h) * [exp(W∘W)^T ∘ 2W]
                var (expWoW, h) = ExpHadamardSquare(W, d);
                var constraintMult = Engine.TensorAdd(alpha, Engine.TensorMultiply(rho, h));
                var acGrad = Engine.TensorMultiply(
                    Engine.TensorBroadcastTo(Engine.TensorMultiplyScalar(constraintMult, two).Reshape(new[] { 1, 1 }), new[] { d, d }),
                    Engine.TensorMultiply(Engine.TensorTranspose(expWoW), W));
                var l1Grad = Engine.TensorMultiplyScalar(Engine.TensorSign(W), lambda);
                gradW = Engine.TensorAdd(gradW, Engine.TensorAdd(acGrad, l1Grad));

                // Update W and A, then zero the diagonal of W
                W = Engine.TensorSubtract(W, Engine.TensorMultiplyScalar(gradW, lr));
                A = Engine.TensorSubtract(A, Engine.TensorMultiplyScalar(gradA, lr));
                W = Engine.TensorWhere(offDiagonal, W, zeros);
            }

            var hVal = ExpHadamardSquare(W, d).Trace;
            if (CausalTensorHelpers.AnyBelow(hVal, converged)) break;
            alpha = Engine.TensorAdd(alpha, Engine.TensorMultiply(rho, hVal));
            rho = Engine.TensorMin(Engine.TensorMultiplyScalar(rho, ten), rhoMax);
        }

        // Threshold and return, with covariance fallback
        T thresh = NumOps.FromDouble(_wThreshold);
        var strong = Engine.TensorGreaterThan(Engine.TensorAbs(W), thresh);
        if (Engine.TensorCountNonzero(strong) > 0)
            return Engine.TensorWhere(strong, W, zeros);

        // Covariance-based fallback for deterministic data: weight cov_ij / var_i when var_i ≥ 1e-10 and
        // |weight| ≥ threshold, kept when it beats the reverse |cov_ij / var_j| (0 when var_j ≤ 1e-10),
        // ties going to i < j.
        var cov = Tensor<T>.FromMatrix(ComputeCovarianceMatrix(stdData.ToMatrix()));
        var variance = Engine.TensorDiagonal(cov);
        var shape = new[] { d, d };
        var varianceRows = Engine.TensorBroadcastTo(variance.Reshape(new[] { d, 1 }), shape);
        var varianceCols = Engine.TensorBroadcastTo(variance.Reshape(new[] { 1, d }), shape);
        T varianceFloor = NumOps.FromDouble(1e-10);
        var ones = Engine.TensorOnesLike(cov);
        var weight = Engine.TensorDivide(cov, varianceRows);
        var absWeight = Engine.TensorAbs(weight);
        var reverseW = Engine.TensorWhere(Engine.TensorGreaterThan(varianceCols, varianceFloor),
            Engine.TensorAbs(Engine.TensorDivide(cov, varianceCols)), zeros);
        var usable = Engine.TensorMultiply(Engine.TensorMultiply(offDiagonal,
            Engine.TensorSubtract(ones, Engine.TensorLessThan(varianceRows, varianceFloor))),
            Engine.TensorSubtract(ones, Engine.TensorLessThan(absWeight, thresh)));
        var wins = Engine.TensorAdd(Engine.TensorGreaterThan(absWeight, reverseW),
            Engine.TensorMultiply(Engine.TensorEquals(absWeight, reverseW), Engine.TensorTriu(ones, 1)));
        return Engine.TensorWhere(Engine.TensorMultiply(usable, wins), weight, zeros);
    }

    /// <summary>
    /// exp(W∘W) by its Taylor series through min(d, 20) terms, and h = tr(exp(W∘W)) − d (the trace of
    /// every term after the identity) as a [1] tensor.
    /// </summary>
    private (Tensor<T> Exp, Tensor<T> Trace) ExpHadamardSquare(Tensor<T> w, int d)
    {
        var wow = Engine.TensorMultiply(w, w);
        var identity = Engine.TensorEye<T>(d);
        var series = new Tensor<T>(new[] { d, d });
        var power = identity;

        // runningFactorial holds k! as a [1, 1] tensor, broadcast over the power matrix.
        var one = Engine.TensorOnesLike(new Tensor<T>(new[] { 1, 1 }));
        var counter = Engine.TensorZerosLike(one);
        var runningFactorial = one;
        for (int k = 1; k <= Math.Min(d, 20); k++)
        {
            power = Engine.TensorMatMul(power, wow);
            counter = Engine.TensorAdd(counter, one);
            runningFactorial = Engine.TensorMultiply(runningFactorial, counter);
            series = Engine.TensorAdd(series, Engine.TensorDivide(power, Engine.TensorBroadcastTo(runningFactorial, new[] { d, d })));
        }

        var trace = Engine.ReduceSum(Engine.TensorDiagonal(series), new[] { 0 }, true);
        return (Engine.TensorAdd(identity, series), trace);
    }
}
