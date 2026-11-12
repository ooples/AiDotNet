using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// Transfer Entropy — information-theoretic measure of directed information flow.
/// </summary>
/// <remarks>
/// <para>
/// Transfer entropy quantifies the amount of directed information transfer from one
/// process to another. It measures the reduction in uncertainty of Y's future given
/// the past of both X and Y, compared to only Y's past. It is a nonlinear generalization
/// of Granger causality.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each pair (X→Y), construct lagged variables: Y_past (lag 1..L of Y), X_past (lag 1..L of X)</item>
/// <item>Compute TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)</item>
/// <item>Using Gaussian approximation: TE = 0.5 * log(var(Y_future|Y_past) / var(Y_future|Y_past,X_past))</item>
/// <item>Apply score threshold: only keep edges where TE exceeds threshold</item>
/// <item>Direction is inherent: TE(X→Y) ≠ TE(Y→X) in general</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Transfer Entropy is like Granger causality but works for nonlinear
/// relationships too. It asks: "Does knowing X's past reduce my uncertainty about Y's future,
/// beyond what Y's own past already tells me?" If yes, X transfers information to Y.
/// </para>
/// <para>
/// Reference: Schreiber (2000), "Measuring Information Transfer", Physical Review Letters.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Measuring Information Transfer", "https://doi.org/10.1103/PhysRevLett.85.461", Year = 2000, Authors = "Thomas Schreiber")]
public class TransferEntropyAlgorithm<T> : InfoTheoreticBase<T>
{
    /// <inheritdoc/>
    public override string Name => "TransferEntropy";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    private readonly int _maxLag;
    private readonly double _threshold;

    public TransferEntropyAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyInfoOptions(options);
        _maxLag = options?.MaxLag ?? 2;
        _threshold = options?.EdgeThreshold ?? 0.05;
        if (_maxLag < 1)
            throw new ArgumentException("MaxLag must be at least 1.");
        if (double.IsNaN(_threshold) || double.IsInfinity(_threshold) || _threshold < 0)
            throw new ArgumentException("EdgeThreshold must be non-negative and finite.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - _maxLag;

        if (d < 2)
            throw new ArgumentException($"TransferEntropy requires at least 2 variables, got {d}.");
        if (effectiveN < 2 * _maxLag + 3)
            throw new ArgumentException($"Insufficient samples for TransferEntropy: need at least {3 * _maxLag + 3} rows for maxLag={_maxLag}, got {n}.");

        var result = new Matrix<T>(d, d);

        for (int source = 0; source < d; source++)
        {
            for (int target = 0; target < d; target++)
            {
                if (source == target) continue;

                // TE(source→target) = H(Y_future|Y_past) - H(Y_future|Y_past,X_past)
                // Gaussian approximation:
                // TE = 0.5 * log(var_restricted / var_unrestricted)
                double te = ComputeTransferEntropy(data, source, target, n);

                if (te > _threshold)
                    result[source, target] = NumOps.FromDouble(te);
            }
        }

        return result;
    }

    private double ComputeTransferEntropy(Matrix<T> data, int source, int target, int n)
    {
        int effectiveN = n - _maxLag;

        var x = Tensor<T>.FromMatrix(data);
        var targetColumn = Engine.TensorSlice(x, new[] { 0, target }, new[] { n, 1 });
        var sourceColumn = Engine.TensorSlice(x, new[] { 0, source }, new[] { n, 1 });

        // Build target future vector: Y[t]
        var yFuture = Engine.TensorSlice(targetColumn, new[] { _maxLag, 0 }, new[] { effectiveN, 1 });

        // Build restricted design matrix: Y_past only (lags 1..L of target), column lag - 1 = Y[t - lag]
        // Build unrestricted design matrix: Y_past + X_past
        int fullDim = 2 * _maxLag;
        var targetLags = new Tensor<T>[_maxLag];
        var sourceLags = new Tensor<T>[_maxLag];
        for (int lag = 1; lag <= _maxLag; lag++)
        {
            targetLags[lag - 1] = Engine.TensorSlice(targetColumn, new[] { _maxLag - lag, 0 }, new[] { effectiveN, 1 });
            sourceLags[lag - 1] = Engine.TensorSlice(sourceColumn, new[] { _maxLag - lag, 0 }, new[] { effectiveN, 1 });
        }
        var yPast = Engine.TensorConcatenate(targetLags, 1);
        var fullPast = Engine.TensorConcatenate(new[] { yPast, Engine.TensorConcatenate(sourceLags, 1) }, 1);

        // Compute residual variance for restricted model (Y_past only)
        double rssRestricted = ComputeOLSResidualVariance(yPast, yFuture, effectiveN, _maxLag);

        // Compute residual variance for unrestricted model (Y_past + X_past)
        double rssUnrestricted = ComputeOLSResidualVariance(fullPast, yFuture, effectiveN, fullDim);

        // TE = 0.5 * log(var_restricted / var_unrestricted)
        if (rssRestricted < 1e-15 && rssUnrestricted < 1e-15)
        {
            // Both models fit perfectly — no meaningful difference in explanatory power.
            return 0;
        }

        if (rssRestricted < 1e-15)
        {
            // Restricted model already fits perfectly — source adds no information.
            return 0;
        }

        // Use a relative cutoff based on restricted RSS to handle different data scales
        double minRss = rssRestricted * 1e-12;
        if (minRss < double.Epsilon) minRss = double.Epsilon;

        if (rssUnrestricted < minRss)
        {
            // Unrestricted model fits nearly perfectly compared to restricted.
            // Cap at a large finite value (log ratio is +infinity when denominator → 0).
            return 0.5 * Math.Log(rssRestricted / minRss);
        }

        double te = 0.5 * Math.Log(rssRestricted / rssUnrestricted);
        return Math.Max(te, 0);
    }

    private double ComputeOLSResidualVariance(Tensor<T> X, Tensor<T> y, int n, int p)
    {
        if (p == 0 || n <= p)
        {
            // Just compute variance of y
            var centered = Engine.TensorSubtractScalar(y, Engine.TensorMean(y));
            return NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(centered))) / n;
        }

        // Build normal equations: X'X * beta = X'y (X'X mirrored so it is exactly symmetric)
        var Xt = Engine.TensorTranspose(X);
        var XtX = CausalTensorHelpers.Symmetrize(Engine.TensorMatMul(Xt, X));

        // Ridge regularization for stability
        XtX = Engine.TensorAdd(XtX, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), NumOps.FromDouble(1e-10)));
        var Xty = Engine.TensorMatMul(Xt, y).Reshape(new[] { p }).ToVector();

        // Solve for beta
        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX.ToMatrix(), Xty, MatrixDecompositionType.Lu);

        // Compute residuals and their variance
        var residuals = Engine.TensorSubtract(y, Engine.TensorMatMul(X, beta.AsTensor().Reshape(new[] { p, 1 })));
        return NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(residuals))) / n;
    }

    /// <summary>
    /// Fallback for deterministic data where OLS residuals are near-zero.
    /// Uses lagged Pearson correlation as a proxy for transfer entropy.
    /// </summary>
    private double ComputeCorrelationFallback(Matrix<T> data, int source, int target, int n)
    {
        int effectiveN = n - 1;
        if (effectiveN < 3) return 0;

        // x = source[t - 1], y = target[t] for t = 1..n-1, with raw-moment covariance and variances
        var values = Tensor<T>.FromMatrix(data);
        var x = Engine.TensorSlice(Engine.TensorSliceAxis(values, 1, source), new[] { 0 }, new[] { effectiveN });
        var y = Engine.TensorSlice(Engine.TensorSliceAxis(values, 1, target), new[] { 1 }, new[] { effectiveN });
        T nT = NumOps.FromDouble(effectiveN);
        T meanX = NumOps.Divide(Engine.TensorSum(x), nT);
        T meanY = NumOps.Divide(Engine.TensorSum(y), nT);
        T covXY = NumOps.Subtract(NumOps.Divide(Engine.TensorSum(Engine.TensorMultiply(x, y)), nT), NumOps.Multiply(meanX, meanY));
        T varX = NumOps.Subtract(NumOps.Divide(Engine.TensorSum(Engine.TensorSquare(x)), nT), NumOps.Multiply(meanX, meanX));
        T varY = NumOps.Subtract(NumOps.Divide(Engine.TensorSum(Engine.TensorSquare(y)), nT), NumOps.Multiply(meanY, meanY));

        T tiny = NumOps.FromDouble(1e-15);
        if (NumOps.LessThan(varX, tiny) || NumOps.LessThan(varY, tiny)) return 0;

        T r = NumOps.Abs(NumOps.Divide(covXY, NumOps.Sqrt(NumOps.Multiply(varX, varY))));
        if (!NumOps.GreaterThan(r, NumOps.FromDouble(_threshold))) return 0;
        return NumOps.ToDouble(NumOps.Multiply(NumOps.FromDouble(-0.5),
            NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, NumOps.Multiply(r, r)), tiny))));
    }
}
