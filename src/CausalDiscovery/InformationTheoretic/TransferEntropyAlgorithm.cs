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
/// <item>Apply significance threshold: only keep edges where TE exceeds threshold</item>
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
[ModelPaper("Measuring Information Transfer", "https://doi.org/10.1103/PhysRevLett.85.461", Year = 2000, Authors = "Thomas Schreiber")]
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
        _maxLag = options?.MaxIterations ?? 2;
        _threshold = options?.EdgeThreshold ?? 0.05;
        if (_maxLag < 1)
            throw new ArgumentException("MaxIterations (lag) must be at least 1.");
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
            throw new ArgumentException($"TransferEntropy requires at least {2 * _maxLag + 3 + _maxLag} samples for lag={_maxLag}, got {n}.");

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

        // Build target future vector: Y[t]
        var yFuture = new Vector<T>(effectiveN);
        for (int t = 0; t < effectiveN; t++)
            yFuture[t] = data[t + _maxLag, target];

        // Build restricted design matrix: Y_past only (lags 1..L of target)
        var yPast = new Matrix<T>(effectiveN, _maxLag);
        for (int t = 0; t < effectiveN; t++)
            for (int lag = 1; lag <= _maxLag; lag++)
                yPast[t, lag - 1] = data[t + _maxLag - lag, target];

        // Build unrestricted design matrix: Y_past + X_past
        int fullDim = 2 * _maxLag;
        var fullPast = new Matrix<T>(effectiveN, fullDim);
        for (int t = 0; t < effectiveN; t++)
        {
            for (int lag = 1; lag <= _maxLag; lag++)
            {
                fullPast[t, lag - 1] = data[t + _maxLag - lag, target];
                fullPast[t, _maxLag + lag - 1] = data[t + _maxLag - lag, source];
            }
        }

        // Compute residual variance for restricted model (Y_past only)
        double rssRestricted = ComputeOLSResidualVariance(yPast, yFuture, effectiveN, _maxLag);

        // Compute residual variance for unrestricted model (Y_past + X_past)
        double rssUnrestricted = ComputeOLSResidualVariance(fullPast, yFuture, effectiveN, fullDim);

        // TE = 0.5 * log(var_restricted / var_unrestricted)
        if (rssUnrestricted < 1e-15 || rssRestricted < 1e-15) return 0;

        double te = 0.5 * Math.Log(rssRestricted / rssUnrestricted);
        return Math.Max(te, 0);
    }

    private double ComputeOLSResidualVariance(Matrix<T> X, Vector<T> y, int n, int p)
    {
        if (p == 0 || n <= p)
        {
            // Just compute variance of y
            T mean = NumOps.Zero;
            T nT = NumOps.FromDouble(n);
            for (int t = 0; t < n; t++)
                mean = NumOps.Add(mean, y[t]);
            mean = NumOps.Divide(mean, nT);

            var centered = new Vector<T>(n);
            for (int t = 0; t < n; t++)
                centered[t] = NumOps.Subtract(y[t], mean);

            return NumOps.ToDouble(Engine.DotProduct(centered, centered)) / n;
        }

        // Build normal equations: X'X * beta = X'y
        var XtX = new Matrix<T>(p, p);
        var Xty = new Vector<T>(p);

        // Use Engine.DotProduct for column-wise computation
        for (int a = 0; a < p; a++)
        {
            var colA = new Vector<T>(n);
            for (int t = 0; t < n; t++) colA[t] = X[t, a];

            Xty[a] = Engine.DotProduct(colA, y);

            for (int b = a; b < p; b++)
            {
                var colB = new Vector<T>(n);
                for (int t = 0; t < n; t++) colB[t] = X[t, b];

                T dot = Engine.DotProduct(colA, colB);
                XtX[a, b] = dot;
                XtX[b, a] = dot;
            }
        }

        // Ridge regularization for stability
        T ridge = NumOps.FromDouble(1e-10);
        for (int a = 0; a < p; a++)
            XtX[a, a] = NumOps.Add(XtX[a, a], ridge);

        // Solve for beta
        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX, Xty, MatrixDecompositionType.Lu);

        // Compute residuals and their variance using Engine.DotProduct
        var residuals = new Vector<T>(n);
        for (int t = 0; t < n; t++)
        {
            T pred = NumOps.Zero;
            for (int c = 0; c < p; c++)
                pred = NumOps.Add(pred, NumOps.Multiply(X[t, c], beta[c]));
            residuals[t] = NumOps.Subtract(y[t], pred);
        }

        return NumOps.ToDouble(Engine.DotProduct(residuals, residuals)) / n;
    }
}
