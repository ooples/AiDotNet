using AiDotNet.Attributes;
using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// TiMINo — Time series Models with Independent Noise (TiMINo causality, linear model class).
/// </summary>
/// <remarks>
/// <para>
/// TiMINo assumes each series follows a restricted structural equation model whose noise is
/// independent of the past of all series and of the present of its parents:
/// x_k[t] = f_k(x_pa[t-p..t], x_k[t-p..t-1]) + n_k[t]. This class fits f_k by least squares with an
/// intercept (the linear TiMINo model class) and decides independence with the HSIC test.
/// </para>
/// <para>
/// <b>Algorithm (Peters, Janzing and Schölkopf 2013, Algorithm 1):</b>
/// <list type="number">
/// <item>S := all variables. For every k in S, regress x_k[t] on x_k[t-1..t-p] and on x_i[t..t-p] for
/// every other i in S, and test whether the residuals are independent of each regressor series.</item>
/// <item>The k with the weakest dependence (the largest p-value) is a sink of S: remove it from S and
/// make the variables left in S its parents. Repeat until one variable is left.</item>
/// <item>Remove unnecessary parents: drop parent i of k when the residuals of the model without i are
/// still independent of every series of the model with i (p-value of at least the significance
/// level).</item>
/// </list>
/// The output has an edge i → k (weight 1) for every remaining parent i of k: TiMINo returns parent
/// sets, not effect sizes.
/// </para>
/// <para>
/// <b>Independence test:</b> the HSIC test of Gretton et al. (2008) with Gaussian kernels whose width
/// is the median heuristic (σ² = median of the positive squared distances / 2) and the gamma
/// approximation of the null distribution. A regression is tested against each of its regressor
/// series (the lags 0..p of the other variables and the lags 1..p of the target); the smallest
/// p-value is Bonferroni corrected by the number of tests.
/// </para>
/// <para>
/// <b>Bad model fit:</b> when even the best sink candidate rejects independence at the significance
/// level, the paper stops with "I do not know". This implementation still completes the order search
/// and pruning (the ordering by weakest dependence is the best available guess) and reports the
/// rejection through <see cref="ModelFitRejected"/>, so a caller can discard the graph.
/// </para>
/// <para>
/// <b>Options:</b> <see cref="CausalDiscoveryOptions.MaxLag"/> is the model order p (default 3) and
/// <see cref="CausalDiscoveryOptions.SignificanceLevel"/> the test level α (default 0.05, as in the
/// paper). The kernel matrices make each test O(m²) in the m = n - p usable time steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> TiMINo checks if the "leftover noise" after predicting one variable
/// from the others is truly unrelated to everything used for the prediction. Only the true causal
/// direction leaves such independent noise, so the variable whose noise looks most independent is
/// placed last in the causal order, and the search repeats on the rest.
/// </para>
/// <para>
/// Reference: Peters, Janzing and Schölkopf (2013), "Causal Inference on Time Series using Restricted
/// Structural Equation Models", NeurIPS. HSIC test: Gretton et al. (2008), "A Kernel Statistical Test
/// of Independence", NeurIPS.
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
[ResearchPaper("Causal Inference on Time Series using Restricted Structural Equation Models", "https://arxiv.org/abs/1207.5136", Year = 2013, Authors = "Jonas Peters, Dominik Janzing, Bernhard Scholkopf")]
public class TiMINoAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <summary>
    /// Default significance level of the independence tests (the paper's α = 0.05).
    /// </summary>
    private const double DefaultSignificanceLevel = 0.05;

    /// <summary>
    /// Smallest number of usable time steps for the HSIC variance estimate, whose factor
    /// (m-4)(m-5) / (m(m-1)(m-2)(m-3)) is only positive from m = 6 on.
    /// </summary>
    private const int MinimumTestSamples = 6;

    private readonly double _alpha = DefaultSignificanceLevel;

    /// <inheritdoc/>
    public override string Name => "TiMINo";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// True when the last <c>DiscoverStructure</c> call
    /// selected a sink whose residuals were still dependent at the significance level: the paper's
    /// "I do not know - bad model fit" outcome, meaning the linear TiMINo model class does not fit.
    /// </summary>
    public bool ModelFitRejected { get; private set; }

    public TiMINoAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
        if (options?.SignificanceLevel is double alpha)
        {
            if (double.IsNaN(alpha) || alpha <= 0 || alpha >= 1)
                throw new ArgumentOutOfRangeException(nameof(options), "SignificanceLevel must be in (0, 1).");
            _alpha = alpha;
        }
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int p = MaxLag;
        int m = n - p;
        ModelFitRejected = false;
        if (d < 2 || m < Math.Max(2 * p + 3, MinimumTestSamples)) return new Matrix<T>(d, d);

        // Column h*d + v of the lag matrix is x_v[t - h], t = p..n-1, h = 0..p.
        var lagged = BuildLagMatrix(Tensor<T>.FromMatrix(data), p);
        int columns = d * (p + 1);
        var kernels = new GaussianKernel[columns];
        for (int c = 0; c < columns; c++)
            kernels[c] = BuildGaussianKernel(Engine.TensorSliceAxis(lagged, 1, c));

        // Order search: repeatedly take the sink of S (weakest residual dependence).
        var remaining = Enumerable.Range(0, d).ToList();
        var parents = new List<int>[d];
        while (remaining.Count > 1)
        {
            int sink = remaining[0];
            double sinkPValue = -1;
            foreach (int k in remaining)
            {
                var regressors = RegressorColumns(k, remaining, d, p);
                double pValue = ResidualIndependencePValue(lagged, k, regressors, regressors, kernels);
                if (pValue > sinkPValue)
                {
                    sinkPValue = pValue;
                    sink = k;
                }
            }

            if (sinkPValue < _alpha) ModelFitRejected = true;
            remaining.Remove(sink);
            parents[sink] = new List<int>(remaining);
        }

        parents[remaining[0]] = new List<int>();

        // Remove unnecessary parents: i is not needed when the model without it still leaves residuals
        // independent of every series of the model with it.
        for (int k = 0; k < d; k++)
        {
            foreach (int parent in parents[k].ToList())
            {
                var withParent = new List<int>(parents[k]) { k };
                var withoutParent = withParent.Where(v => v != parent).ToList();
                double pValue = ResidualIndependencePValue(lagged, k,
                    RegressorColumns(k, withoutParent, d, p), RegressorColumns(k, withParent, d, p), kernels);
                if (pValue >= _alpha) parents[k].Remove(parent);
            }
        }

        var adjacency = new int[d * d];
        for (int k = 0; k < d; k++)
            foreach (int parent in parents[k])
                adjacency[parent * d + k] = 1;

        return new Tensor<int>(adjacency, new[] { d, d }).Cast<T>().ToMatrix();
    }

    /// <summary>
    /// The [n - p, d(p + 1)] matrix whose column h*d + v holds x_v[t - h] for t = p..n-1.
    /// </summary>
    private Tensor<T> BuildLagMatrix(Tensor<T> x, int p)
    {
        int m = x.Shape[0] - p;
        int d = x.Shape[1];
        var blocks = new Tensor<T>[p + 1];
        for (int h = 0; h <= p; h++)
            blocks[h] = Engine.TensorSlice(x, new[] { p - h, 0 }, new[] { m, d });
        return Engine.TensorConcatenate(blocks, 1);
    }

    /// <summary>
    /// Lag-matrix columns regressing x_k[t]: its own lags 1..p and the lags 0..p of every other
    /// variable in <paramref name="variables"/>, in ascending column order.
    /// </summary>
    private static int[] RegressorColumns(int k, IReadOnlyCollection<int> variables, int d, int p)
    {
        var columns = new List<int>();
        for (int h = 0; h <= p; h++)
            foreach (int v in variables)
                if (v != k || h > 0) columns.Add(h * d + v);
        columns.Sort();
        return columns.ToArray();
    }

    /// <summary>
    /// Bonferroni-corrected p-value of "the residuals of x_k[t] on <paramref name="predictors"/> are
    /// independent of every series in <paramref name="testColumns"/>": the smallest HSIC p-value times
    /// the number of tests, capped at 1.
    /// </summary>
    private double ResidualIndependencePValue(Tensor<T> lagged, int k, int[] predictors, int[] testColumns,
        GaussianKernel[] kernels)
    {
        var (_, residual) = FitCenteredRegression(lagged, k, predictors);
        var residualKernel = BuildGaussianKernel(residual);
        double smallest = 1.0;
        foreach (int c in testColumns)
            smallest = Math.Min(smallest, HsicGammaPValue(residualKernel, kernels[c]));
        return Math.Min(1.0, smallest * testColumns.Length);
    }

    /// <summary>
    /// Gaussian kernel matrix of one series and the quantities the HSIC gamma test needs from it.
    /// </summary>
    internal sealed class GaussianKernel
    {
        public GaussianKernel(int samples, T squaredWidth, Tensor<T>? centered, T offDiagonalMean)
        {
            Samples = samples;
            SquaredWidth = squaredWidth;
            Centered = centered;
            OffDiagonalMean = offDiagonalMean;
        }

        /// <summary>Number of samples m.</summary>
        public int Samples { get; }

        /// <summary>2σ², the median of the positive squared pairwise distances (zero for a constant series).</summary>
        public T SquaredWidth { get; }

        /// <summary>HKH with H = I - 11ᵀ/m, or null for a constant series (no positive distance).</summary>
        public Tensor<T>? Centered { get; }

        /// <summary>Mean of the off-diagonal kernel entries, (1ᵀK1 - m) / (m(m-1)).</summary>
        public T OffDiagonalMean { get; }
    }

    /// <summary>
    /// K[a, b] = exp(-(x_a - x_b)² / (2σ²)) with the median heuristic 2σ² = median of the positive
    /// squared distances (Gretton et al. 2008), returned centered together with its off-diagonal mean.
    /// </summary>
    internal GaussianKernel BuildGaussianKernel(Tensor<T> series)
    {
        int m = series.Length;
        var square = new[] { m, m };
        var difference = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(Engine.Reshape(series, new[] { m, 1 }), square),
            Engine.TensorBroadcastTo(Engine.Reshape(series, new[] { 1, m }), square));
        var squaredDistance = Engine.TensorMultiply(difference, difference);

        // The positive entries come in symmetric pairs, so their median equals the median over the
        // strict lower triangle; after an ascending sort they are the last `positives` entries.
        int positives = (int)Engine.TensorCountNonzero(Engine.TensorGreaterThan(squaredDistance, NumOps.Zero));
        if (positives == 0) return new GaussianKernel(m, NumOps.Zero, null, NumOps.One);

        var (sorted, _) = Engine.TensorSort(Engine.Reshape(squaredDistance, new[] { m * m }), 0, false);
        int lowerMiddle = m * m - positives + positives / 2 - 1;
        T squaredWidth = NumOps.Divide(
            Engine.TensorSum(Engine.TensorSlice(sorted, new[] { lowerMiddle }, new[] { 2 })), NumOps.FromDouble(2));

        var gram = Engine.TensorExp(Engine.TensorMultiplyScalar(squaredDistance,
            NumOps.Negate(NumOps.Divide(NumOps.One, squaredWidth))));
        var rowMean = Engine.TensorBroadcastTo(Engine.ReduceMean(gram, new[] { 1 }, true), square);
        var columnMean = Engine.TensorBroadcastTo(Engine.ReduceMean(gram, new[] { 0 }, true), square);
        var centered = Engine.TensorAddScalar(
            Engine.TensorSubtract(Engine.TensorSubtract(gram, rowMean), columnMean), Engine.TensorMean(gram));

        // The diagonal is exp(0) = 1, so the off-diagonal sum is the total minus m.
        T offDiagonalMean = NumOps.Divide(
            NumOps.Subtract(Engine.TensorSum(gram), NumOps.FromDouble(m)), NumOps.FromDouble((double)m * (m - 1)));
        return new GaussianKernel(m, squaredWidth, centered, offDiagonalMean);
    }

    /// <summary>
    /// HSIC test statistic m·HSIC_b = tr(K̃L̃)/m and its p-value under the gamma approximation of the
    /// null distribution (Gretton et al. 2008): mean (1 + μ_K μ_L - μ_K - μ_L)/m, variance
    /// 72(m-4)(m-5)/(m(m-1)(m-2)(m-3)) · Σ_{a≠b}((K̃∘L̃)_ab/6)² / (m(m-1)), shape mean²/variance and
    /// scale m·variance/mean. A constant series is independent of everything (p-value 1).
    /// </summary>
    internal (T Statistic, double PValue) HsicGammaTest(GaussianKernel x, GaussianKernel y)
    {
        if (x.Centered is not Tensor<T> kx || y.Centered is not Tensor<T> ky) return (NumOps.Zero, 1.0);

        int m = x.Samples;
        var product = Engine.TensorMultiply(kx, ky);
        T statistic = NumOps.Divide(Engine.TensorSum(product), NumOps.FromDouble(m));

        var scaledSquare = Engine.TensorSquare(Engine.TensorMultiplyScalar(product, NumOps.FromDouble(1.0 / 6)));
        T offDiagonal = NumOps.Subtract(Engine.TensorSum(scaledSquare), Engine.TensorTrace(scaledSquare));
        double md = m;
        double varianceFactor = 72.0 * (md - 4) * (md - 5) / (md * (md - 1) * (md - 2) * (md - 3)) / (md * (md - 1));
        T variance = NumOps.Multiply(offDiagonal, NumOps.FromDouble(varianceFactor));

        T muX = x.OffDiagonalMean;
        T muY = y.OffDiagonalMean;
        T mean = NumOps.Divide(
            NumOps.Subtract(NumOps.Add(NumOps.One, NumOps.Multiply(muX, muY)), NumOps.Add(muX, muY)),
            NumOps.FromDouble(m));
        if (!NumOps.GreaterThan(variance, NumOps.Zero) || !NumOps.GreaterThan(mean, NumOps.Zero))
            return (statistic, 1.0);

        T shape = NumOps.Divide(NumOps.Multiply(mean, mean), variance);
        T rate = NumOps.Divide(mean, NumOps.Multiply(variance, NumOps.FromDouble(m)));
        var nullDistribution = new GammaDistribution<T>(shape, rate);
        return (statistic, 1.0 - NumOps.ToDouble(nullDistribution.Cdf(statistic)));
    }

    private double HsicGammaPValue(GaussianKernel x, GaussianKernel y) => HsicGammaTest(x, y).PValue;
}
