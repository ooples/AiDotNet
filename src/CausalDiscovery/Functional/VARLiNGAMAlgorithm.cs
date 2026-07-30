using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// VAR-LiNGAM — Vector Autoregressive LiNGAM for time series causal discovery.
/// </summary>
/// <remarks>
/// <para>
/// VAR-LiNGAM combines VAR (Vector Autoregression) with LiNGAM to discover both
/// contemporaneous (same time-step) and lagged (across time-steps) causal relationships.
/// </para>
/// <para>
/// <b>Model:</b> X(t) = B₀ X(t) + B₁ X(t-1) + ... + Bₖ X(t-k) + e(t)
/// where B₀ encodes contemporaneous effects and B₁...Bₖ encode lagged effects.
/// </para>
/// <para>
/// <b>For Beginners:</b> This algorithm finds causal relationships in time series data
/// that work at different time scales. It can detect both "X causes Y right now"
/// (contemporaneous) and "yesterday's X causes today's Y" (lagged) relationships.
/// </para>
/// <para>
/// Reference: Hyvarinen et al. (2010), "Estimation of a Structural Vector Autoregression
/// Model Using Non-Gaussianity", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity", "https://jmlr.org/papers/v11/hyvarinen10a.html", Year = 2010, Authors = "Aapo Hyvarinen, Kun Zhang, Shohei Shimizu, Patrik O. Hoyer")]
public class VARLiNGAMAlgorithm<T> : FunctionalBase<T>
{
    private readonly int _maxLag = 3;
    private readonly double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "VAR-LiNGAM";

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public VARLiNGAMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
        if (options?.MaxLag.HasValue == true) _maxLag = Math.Max(1, options.MaxLag.Value);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        int effectiveN = n - _maxLag;
        if (effectiveN < d + 1) return new Matrix<T>(d, d);

        // Standardize data for numerical stability
        var standardized = StandardizeData(data);

        // Step 1: Fit VAR model to get residuals and lagged coefficients
        var (residuals, laggedCoefs) = FitVARAndGetResiduals(standardized, n, d, _maxLag);

        // Step 2: Apply DirectLiNGAM on residuals to get B₀ (contemporaneous effects)
        var directLiNGAM = new DirectLiNGAMAlgorithm<T>(
            new CausalDiscoveryOptions { EdgeThreshold = _threshold });
        var B0Graph = directLiNGAM.DiscoverStructure(residuals);

        // Step 3: Start the framework's 2-D summary with the instantaneous effects.
        var result = new Matrix<T>(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                result[i, j] = B0Graph.AdjacencyMatrix[i, j];

        // Step 4 of Hyvarinen et al. (2010), Eq. (11): OLS estimates the reduced-form VAR
        // matrices M_tau, not the structural causal matrices. Recover each causal lag matrix as
        // B_tau = (I - B_0) M_tau before aggregating it. Matrices exposed by this framework use
        // adjacency orientation [source, target], the transpose of the paper's [target, source].
        for (int lag = 0; lag < _maxLag; lag++)
        {
            for (int source = 0; source < d; source++)
            {
                for (int target = 0; target < d; target++)
                {
                    // Skip self-loops: autoregressive coefficients (X on its own lag)
                    // are not causal edges in the summary adjacency matrix
                    if (source == target) continue;

                    T structuralWeight = laggedCoefs[target][lag * d + source];
                    for (int intermediary = 0; intermediary < d; intermediary++)
                    {
                        structuralWeight = NumOps.Subtract(
                            structuralWeight,
                            NumOps.Multiply(
                                B0Graph.AdjacencyMatrix[intermediary, target],
                                laggedCoefs[intermediary][lag * d + source]));
                    }

                    double lagWeightD = Math.Abs(NumOps.ToDouble(structuralWeight));
                    double currentD = Math.Abs(NumOps.ToDouble(result[source, target]));
                    if (lagWeightD >= _threshold && lagWeightD > currentD)
                    {
                        result[source, target] = structuralWeight;
                    }
                }
            }
        }

        // Fallback: if neither B0 nor lagged coefficients produced edges,
        // use pairwise cross-correlation to detect relationships.
        // This handles deterministic/near-deterministic data where LiNGAM's
        // non-Gaussianity assumption fails and VAR overfits.
        bool hasEdges = false;
        for (int i = 0; i < d && !hasEdges; i++)
            for (int j = 0; j < d && !hasEdges; j++)
                if (i != j && NumOps.GreaterThan(NumOps.Abs(result[i, j]), NumOps.Zero))
                    hasEdges = true;

        if (!hasEdges)
        {
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    var xi = standardized.GetColumn(i);
                    var xj = standardized.GetColumn(j);
                    double corr = Math.Abs(ComputeCorrelation(xi, xj));
                    if (corr >= _threshold)
                    {
                        result[i, j] = NumOps.FromDouble(corr);
                    }
                }
            }
        }

        // A time-unrolled VAR graph may legitimately contain X_i(t-1)->X_j(t) and
        // X_j(t-1)->X_i(t) simultaneously without a directed cycle. ICausalDiscoveryAlgorithm,
        // however, returns one lag-collapsed CausalGraph whose contract is a DAG. Preserve the
        // strongest estimated effects while greedily omitting only edges that would introduce a
        // cycle in that lossy 2-D summary.
        return ProjectSummaryToDag(result, d);
    }

    private Matrix<T> ProjectSummaryToDag(Matrix<T> candidate, int dimension)
    {
        var edges = new List<(int Source, int Target, T Weight, double Magnitude)>();
        for (int source = 0; source < dimension; source++)
        {
            for (int target = 0; target < dimension; target++)
            {
                if (source == target) continue;
                T weight = candidate[source, target];
                double magnitude = Math.Abs(NumOps.ToDouble(weight));
                if (magnitude >= _threshold)
                    edges.Add((source, target, weight, magnitude));
            }
        }

        edges.Sort((left, right) =>
        {
            int magnitudeOrder = right.Magnitude.CompareTo(left.Magnitude);
            if (magnitudeOrder != 0) return magnitudeOrder;
            int sourceOrder = left.Source.CompareTo(right.Source);
            return sourceOrder != 0 ? sourceOrder : left.Target.CompareTo(right.Target);
        });

        var dag = new Matrix<T>(dimension, dimension);
        foreach (var edge in edges)
        {
            if (!WouldCreateCycle(dag, edge.Source, edge.Target, dimension))
                dag[edge.Source, edge.Target] = edge.Weight;
        }

        return dag;
    }

    private bool WouldCreateCycle(Matrix<T> adjacency, int source, int target, int dimension)
    {
        // Adding source -> target closes a cycle exactly when target already reaches source.
        var visited = new bool[dimension];
        var pending = new Stack<int>();
        pending.Push(target);

        while (pending.Count > 0)
        {
            int current = pending.Pop();
            if (current == source) return true;
            if (visited[current]) continue;
            visited[current] = true;

            for (int next = 0; next < dimension; next++)
            {
                if (next != current && Math.Abs(NumOps.ToDouble(adjacency[current, next])) >= _threshold)
                    pending.Push(next);
            }
        }

        return false;
    }

    private (Matrix<T> Residuals, Vector<T>[] LaggedCoefs) FitVARAndGetResiduals(
        Matrix<T> data, int n, int d, int maxLag)
    {
        int effectiveN = n - maxLag;
        var residuals = new Matrix<T>(effectiveN, d);
        var laggedCoefs = new Vector<T>[d];

        for (int target = 0; target < d; target++)
        {
            int p = d * maxLag;
            var design = new Matrix<T>(effectiveN, p);
            var y = new Vector<T>(effectiveN);

            for (int t = 0; t < effectiveN; t++)
            {
                y[t] = data[t + maxLag, target];
                for (int lag = 0; lag < maxLag; lag++)
                    for (int col = 0; col < d; col++)
                        design[t, lag * d + col] = data[t + maxLag - lag - 1, col];
            }

            // Solve OLS: beta = (X^T X + ridge)^{-1} X^T y
            var XtX = new Matrix<T>(p, p);
            var Xty = new Vector<T>(p);
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    T sum = NumOps.Zero;
                    for (int k = 0; k < effectiveN; k++)
                        sum = NumOps.Add(sum, NumOps.Multiply(design[k, i], design[k, j]));
                    XtX[i, j] = sum;
                }
                T sumY = NumOps.Zero;
                for (int k = 0; k < effectiveN; k++)
                    sumY = NumOps.Add(sumY, NumOps.Multiply(design[k, i], y[k]));
                Xty[i] = sumY;
            }

            // Ridge regularization
            T ridge = NumOps.FromDouble(1e-4);
            for (int i = 0; i < p; i++) XtX[i, i] = NumOps.Add(XtX[i, i], ridge);

            var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX, Xty, MatrixDecompositionType.Lu);
            laggedCoefs[target] = beta;

            for (int t = 0; t < effectiveN; t++)
            {
                T pred = NumOps.Zero;
                for (int j = 0; j < p; j++)
                    pred = NumOps.Add(pred, NumOps.Multiply(beta[j], design[t, j]));
                residuals[t, target] = NumOps.Subtract(y[t], pred);
            }
        }

        return (residuals, laggedCoefs);
    }
}
