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

        // The (I - B_0) M_tau product below uses the UNTHRESHOLDED coefficients. DirectLiNGAM zeroes
        // anything under its edge threshold in the graph it returns, and doing matrix algebra with
        // that treats every small-but-real coefficient as an exact zero, which biases B_tau. The
        // reported graph stays thresholded; only the algebra sees the raw values.
        var b0Coefficients = directLiNGAM.RawCoefficients ?? B0Graph.AdjacencyMatrix;

        // Step 3: Start the framework's 2-D summary with the instantaneous effects.
        var result = Tensor<T>.FromMatrix(B0Graph.AdjacencyMatrix);
        var offDiagonal = OffDiagonalMask(d);
        var b0 = Tensor<T>.FromMatrix(b0Coefficients);
        var identityMinusB0 = Engine.TensorSubtract(Engine.TensorEye<T>(d), b0);
        T threshold = NumOps.FromDouble(_threshold);
        var thresholdTensor = Engine.TensorFullLike(result, threshold);

        // Step 4 of Hyvarinen et al. (2010), Eq. (11): OLS estimates the reduced-form VAR
        // matrices M_tau, not the structural causal matrices. Recover each causal lag matrix as
        // B_tau = (I - B_0) M_tau before aggregating it. Matrices exposed by this framework use
        // adjacency orientation [source, target], the transpose of the paper's [target, source]:
        // structural[source, target] = M[source, target] - sum_k M[source, k] * B0[k, target].
        for (int lag = 0; lag < _maxLag; lag++)
        {
            var reducedForm = Engine.TensorSlice(laggedCoefs, new[] { lag * d, 0 }, new[] { d, d });
            var structural = Engine.TensorMatMul(reducedForm, identityMinusB0);
            var magnitude = Engine.TensorAbs(structural);

            // Skip self-loops: autoregressive coefficients (X on its own lag)
            // are not causal edges in the summary adjacency matrix
            var replace = Engine.TensorMultiply(offDiagonal, Engine.TensorMultiply(
                Engine.TensorGreaterEqual(magnitude, thresholdTensor),
                Engine.TensorGreaterThan(magnitude, Engine.TensorAbs(result))));
            result = Engine.TensorWhere(replace, structural, result);
        }

        // Fallback: if neither B0 nor lagged coefficients produced edges,
        // use pairwise cross-correlation to detect relationships.
        // This handles deterministic/near-deterministic data where LiNGAM's
        // non-Gaussianity assumption fails and VAR overfits.
        bool hasEdges = CausalTensorHelpers.AnyAbove(
            Engine.TensorWhere(offDiagonal, Engine.TensorAbs(result), Engine.TensorZerosLike(result)), NumOps.Zero);

        if (!hasEdges)
        {
            var correlations = new Dictionary<int, double>();
            for (int i = 0; i < d; i++)
            {
                for (int j = i + 1; j < d; j++)
                {
                    var xi = standardized.GetColumn(i);
                    var xj = standardized.GetColumn(j);
                    double corr = Math.Abs(ComputeCorrelation(xi, xj));
                    if (corr >= _threshold)
                    {
                        correlations[i * d + j] = corr;
                    }
                }
            }

            result = CausalTensorHelpers.OverwriteStatistics(result, correlations);
        }

        // A time-unrolled VAR graph may legitimately contain X_i(t-1)->X_j(t) and
        // X_j(t-1)->X_i(t) simultaneously without a directed cycle. ICausalDiscoveryAlgorithm,
        // however, returns one lag-collapsed CausalGraph whose contract is a DAG. Preserve the
        // strongest estimated effects while greedily omitting only edges that would introduce a
        // cycle in that lossy 2-D summary.
        return ProjectSummaryToDag(result.ToMatrix(), d);
    }

    private Matrix<T> ProjectSummaryToDag(Matrix<T> candidate, int dimension)
    {
        // Candidate edges: off-diagonal entries with |weight| >= threshold, in row-major (source, target) order.
        var weights = Tensor<T>.FromMatrix(candidate);
        var magnitude = Engine.TensorAbs(weights);
        var isCandidate = Engine.TensorMultiply(OffDiagonalMask(dimension),
            Engine.TensorGreaterEqual(magnitude, Engine.TensorFullLike(magnitude, NumOps.FromDouble(_threshold))));
        int[] positions = Engine.TensorNonzero(isCandidate.Reshape(new[] { dimension * dimension })).ToArray();
        if (positions.Length == 0)
            return new Matrix<T>(dimension, dimension);

        // Strongest first; equal magnitudes keep (source, target) order.
        var candidateMagnitudes = Engine.TensorIndexSelect(magnitude.Reshape(new[] { dimension * dimension }),
            new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);
        int[] order = CausalTensorHelpers.StableArgsort(candidateMagnitudes, descending: true);

        // Every kept edge has |weight| >= threshold, so the kept edge lists are the dag's adjacency.
        var children = new List<int>[dimension];
        for (int node = 0; node < dimension; node++) children[node] = new List<int>();
        var kept = new List<int>();
        foreach (int rank in order)
        {
            int position = positions[rank];
            int source = position / dimension, target = position % dimension;
            if (!WouldCreateCycle(children, source, target, dimension))
            {
                children[source].Add(target);
                kept.Add(position);
            }
        }

        return CausalTensorHelpers.KeepPositions(weights, kept);
    }

    private static bool WouldCreateCycle(List<int>[] children, int source, int target, int dimension)
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

            foreach (int next in children[current])
                pending.Push(next);
        }

        return false;
    }

    /// <summary>
    /// OLS fit of every variable on all lags; returns the residuals [effectiveN x d] and the
    /// coefficients [d * maxLag x d] whose row lag * d + source, column target is the coefficient of
    /// source at lag + 1 for target.
    /// </summary>
    private (Matrix<T> Residuals, Tensor<T> LaggedCoefs) FitVARAndGetResiduals(
        Matrix<T> data, int n, int d, int maxLag)
    {
        int effectiveN = n - maxLag;
        int p = d * maxLag;
        var x = Tensor<T>.FromMatrix(data);

        // design[t, lag * d + col] = data[t + maxLag - lag - 1, col]; y[t, target] = data[t + maxLag, target]
        var lagBlocks = new Tensor<T>[maxLag];
        for (int lag = 0; lag < maxLag; lag++)
            lagBlocks[lag] = Engine.TensorSlice(x, new[] { maxLag - lag - 1, 0 }, new[] { effectiveN, d });
        var design = Engine.TensorConcatenate(lagBlocks, 1);
        var y = Engine.TensorSlice(x, new[] { maxLag, 0 }, new[] { effectiveN, d });

        // Solve OLS: beta = (X^T X + ridge)^{-1} X^T y, the Gram matrix mirrored so it is exactly symmetric
        var designT = Engine.TensorTranspose(design);
        var gram = Engine.TensorMatMul(designT, design);
        gram = Engine.TensorAdd(Engine.TensorTriu(gram, 0), Engine.TensorTranspose(Engine.TensorTriu(gram, 1)));
        var XtX = Engine.TensorAdd(gram, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), NumOps.FromDouble(1e-4)))
            .ToMatrix();
        var Xty = Engine.TensorMatMul(designT, y);

        var betas = new Tensor<T>[d];
        for (int target = 0; target < d; target++)
        {
            var beta = MatrixSolutionHelper.SolveLinearSystem<T>(XtX,
                Engine.TensorSliceAxis(Xty, 1, target).ToVector(), MatrixDecompositionType.Lu);
            betas[target] = beta.AsTensor();
        }

        var laggedCoefs = Engine.TensorStack(betas, 1);
        var residuals = Engine.TensorSubtract(y, Engine.TensorMatMul(design, laggedCoefs));
        return (residuals.ToMatrix(), laggedCoefs);
    }
}
