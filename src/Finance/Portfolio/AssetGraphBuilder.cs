using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// Builds the filtered asset graph for <see cref="GraphAttentionPortfolio{T}"/>: distance correlation
/// over return-volatility series, filtered by the Triangulated Maximally Filtered Graph (TMFG)
/// method (Korangi, Mues and Bravo, arXiv:2407.15532).
/// </summary>
/// <remarks>
/// <para>
/// Two choices distinguish this from the usual correlation network, and both are load-bearing.
/// </para>
/// <para>
/// <b>Distance correlation, not Pearson.</b> Pearson captures only LINEAR pairwise dependence, and
/// is zero for plenty of strongly dependent pairs. Distance correlation is zero if and only if the
/// two series are independent, so it detects non-linear co-movement. It also tolerates short,
/// irregular or partly missing history — which is what lets the paper keep every firm, including
/// those that later default, instead of dropping them and introducing selection bias.
/// </para>
/// <para>
/// <b>Volatility series, not returns.</b> Correlations are computed between return-VOLATILITY series
/// (rolling standard deviation of returns), following Diebold and Yilmaz's connectedness work.
/// Volatility series co-move sharply during risk-off periods and only weakly in benign conditions,
/// which is the structure the graph is meant to expose.
/// </para>
/// <para>
/// <b>TMFG, not a minimum spanning tree.</b> TMFG keeps a near-maximal-weight PLANAR subgraph,
/// reducing edges from <c>|V|(|V|-1)/2</c> to at most <c>3(|V| - 2)</c> while retaining far more
/// information than an MST, which forbids cycles entirely. It is also parallelizable, unlike PMFG,
/// which is what makes it usable at the paper's ~5,000-firm scale.
/// </para>
/// <para><b>For Beginners:</b> This works out which companies genuinely move together, then throws
/// away all but the most informative connections so the network is small enough to learn from.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class AssetGraphBuilder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's volatility lookback in trading days.</summary>
    public const int PaperVolatilityLookback = 30;

    private readonly int _volatilityLookback;

    /// <summary>Gets the volatility lookback in days.</summary>
    public int VolatilityLookback => _volatilityLookback;

    /// <summary>Creates a graph builder.</summary>
    /// <param name="volatilityLookback">Days per volatility estimate. Paper: 30.</param>
    public AssetGraphBuilder(int volatilityLookback = PaperVolatilityLookback)
    {
        if (volatilityLookback < 2)
            throw new ArgumentOutOfRangeException(nameof(volatilityLookback), volatilityLookback,
                "A standard deviation needs at least 2 observations.");
        _volatilityLookback = volatilityLookback;
    }

    /// <summary>
    /// Converts a return panel into rolling return-volatility series.
    /// </summary>
    /// <param name="returns">Daily returns, shaped <c>[steps, assets]</c>.</param>
    /// <returns>
    /// Volatility series shaped <c>[steps - lookback + 1, assets]</c>: each row is the standard
    /// deviation of the trailing <see cref="VolatilityLookback"/> returns.
    /// </returns>
    public Tensor<T> VolatilitySeries(Tensor<T> returns)
    {
        if (returns == null) throw new ArgumentNullException(nameof(returns));
        if (returns.Shape.Length != 2)
            throw new ArgumentException(
                $"returns must be [steps, assets]; got rank {returns.Shape.Length}.", nameof(returns));

        int steps = returns.Shape[0];
        int assets = returns.Shape[1];
        if (steps < _volatilityLookback)
            throw new ArgumentException(
                $"Need at least {_volatilityLookback} steps for a volatility estimate; got {steps}.",
                nameof(returns));

        int outSteps = steps - _volatilityLookback + 1;

        // Window k of every output step is rows [k, k + outSteps) of the panel. The two-pass population
        // standard deviation accumulates those shifted [outSteps, assets] views one at a time, so memory
        // stays at a few panels instead of lookback copies of it.
        var engine = AiDotNetEngine.Current;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var count = NumOps.FromDouble(_volatilityLookback);
        var sum = engine.TensorNarrow(returns, 0, 0, outSteps).Contiguous();
        for (int k = 1; k < _volatilityLookback; k++)
            sum = engine.TensorAdd(sum, engine.TensorNarrow(returns, 0, k, outSteps).Contiguous());
        var mean = engine.TensorDivideScalar(sum, count);

        Tensor<T>? squares = null;
        for (int k = 0; k < _volatilityLookback; k++)
        {
            var centred = engine.TensorSubtract(engine.TensorNarrow(returns, 0, k, outSteps).Contiguous(), mean);
            var square = engine.TensorMultiply(centred, centred);
            squares = squares is null ? square : engine.TensorAdd(squares, square);
        }

        return engine.TensorSqrt(engine.TensorDivideScalar(squares ?? engine.TensorZerosLike(mean), count));
    }

    /// <summary>
    /// Distance correlation between two series, in [0, 1].
    /// </summary>
    /// <remarks>
    /// <para>
    /// Builds the pairwise absolute-difference matrix for each series, DOUBLE-CENTRES both (subtract
    /// the row mean and the column mean, add the grand mean), and takes the mean of their elementwise
    /// product as the distance covariance. Double-centring is what makes the statistic vanish exactly
    /// under independence; skipping it leaves a quantity that is positive for independent series and
    /// so cannot distinguish dependence from none.
    /// </para>
    /// <para>
    /// Unlike a Pearson correlation this is 0 if and only if the two series are INDEPENDENT, and it is
    /// never negative — it measures strength of dependence, not direction.
    /// </para>
    /// </remarks>
    public T DistanceCorrelation(Vector<T> first, Vector<T> second)
    {
        if (first == null) throw new ArgumentNullException(nameof(first));
        if (second == null) throw new ArgumentNullException(nameof(second));
        if (first.Length != second.Length)
            throw new ArgumentException(
                $"Series must have equal length; got {first.Length} and {second.Length}.", nameof(second));
        if (first.Length < 2)
            throw new ArgumentException("At least 2 observations are required.", nameof(first));
        int n = first.Length;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var columns = engine.TensorStack(new[] { Tensor<T>.FromVector(first), Tensor<T>.FromVector(second) }, 0);
        var centred = DoubleCentredDistances(columns, n);                       // [2, n * n]
        var dcov2 = MeanProducts(centred, centred, n);                          // [2, 2]
        var dvar = DiagonalOf(dcov2, 2);
        return DistanceCorrelationFromCentred(engine.TensorNarrow(engine.TensorNarrow(dcov2, 0, 0, 1), 1, 1, 1),
            engine.TensorNarrow(dvar, 0, 0, 1), engine.TensorNarrow(dvar, 0, 1, 1)).AsSpan()[0];
    }

    /// <summary>
    /// Distance correlation from the squared distance covariances: dcor = sqrt(max(0, dcov2)) / sqrt(sqrt(dvarX2) *
    /// sqrt(dvarY2)), clamped to [0, 1], and 0 where the denominator is not positive.
    /// </summary>
    /// <param name="dcov2">Squared distance covariances [rows, cols].</param>
    /// <param name="dvarRows">Squared distance variances of the row series, [rows] or [rows, 1].</param>
    /// <param name="dvarCols">Squared distance variances of the column series, [cols] or [1, cols].</param>
    private static Tensor<T> DistanceCorrelationFromCentred(Tensor<T> dcov2, Tensor<T> dvarRows, Tensor<T> dvarCols)
    {
        var engine = AiDotNetEngine.Current;
        int rows = dcov2.Shape[0];
        int cols = dcov2.Shape[1];
        var rowScale = engine.Reshape(engine.TensorSqrt(dvarRows), new[] { rows, 1 });
        var colScale = engine.Reshape(engine.TensorSqrt(dvarCols), new[] { 1, cols });
        var denominator = engine.TensorSqrt(engine.TensorMultiply(rowScale, colScale));

        // A constant series has zero distance variance and no dependence structure to report, so the
        // correlation is 0 rather than a division by zero.
        var degenerate = engine.TensorLessEqual(denominator, engine.TensorZerosLike(denominator));
        var safeDenominator = engine.TensorWhere(degenerate, engine.TensorFullLike(denominator, NumOps.One), denominator);
        var dcor = engine.TensorDivide(engine.TensorSqrt(engine.TensorMax(dcov2, NumOps.Zero)), safeDenominator);

        // Clamp against floating-point drift a hair above 1, so callers can rely on the [0,1] range.
        dcor = engine.TensorClamp(dcor, NumOps.Zero, NumOps.One);
        return engine.TensorWhere(degenerate, engine.TensorZerosLike(dcor), dcor);
    }

    /// <summary>
    /// Double-centred pairwise distance matrices of a block of series, flattened row-major: for each series x,
    /// A[i, j] = |x_i - x_j| - rowMean[i] - colMean[j] + grandMean.
    /// </summary>
    /// <param name="series">The series, one per row: [count, n].</param>
    /// <param name="n">Observations per series.</param>
    /// <returns>[count, n * n].</returns>
    private static Tensor<T> DoubleCentredDistances(Tensor<T> series, int n)
    {
        var engine = AiDotNetEngine.Current;
        int count = series.Shape[0];
        var distances = engine.TensorAbs(engine.TensorSubtract(
            engine.Reshape(series, new[] { count, n, 1 }), engine.Reshape(series, new[] { count, 1, n })));
        var rowMean = engine.ReduceMean(distances, new[] { 2 }, true);
        var colMean = engine.ReduceMean(distances, new[] { 1 }, true);
        var grand = engine.ReduceMean(distances, new[] { 1, 2 }, true);
        var centred = engine.TensorAdd(
            engine.TensorSubtract(engine.TensorSubtract(distances, rowMean), colMean), grand);
        return engine.Reshape(centred, new[] { count, n * n });
    }

    /// <summary>Mean element-wise products of every row of <paramref name="a"/> with every row of <paramref name="b"/>.</summary>
    private static Tensor<T> MeanProducts(Tensor<T> a, Tensor<T> b, int n)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivideScalar(
            engine.TensorMatMul(a, engine.TensorTranspose(b)), NumOps.FromDouble((double)n * n));
    }

    /// <summary>The diagonal of a square [size, size] tensor, as [size].</summary>
    private static Tensor<T> DiagonalOf(Tensor<T> square, int size)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(engine.TensorMultiply(square, engine.TensorEye<T>(size)), new[] { 1 }, false);
    }

    /// <summary>
    /// Dense distance-correlation matrix over a panel of series.
    /// </summary>
    /// <param name="panel">Series shaped <c>[steps, assets]</c> — volatility series in the paper.</param>
    /// <returns>A symmetric <c>[assets, assets]</c> matrix with a unit diagonal.</returns>
    public Tensor<T> DistanceCorrelationMatrix(Tensor<T> panel)
    {
        if (panel == null) throw new ArgumentNullException(nameof(panel));
        if (panel.Shape.Length != 2)
            throw new ArgumentException(
                $"panel must be [steps, assets]; got rank {panel.Shape.Length}.", nameof(panel));

        int steps = panel.Shape[0];
        int assets = panel.Shape[1];
        if (steps < 2)
            throw new ArgumentException("At least 2 observations are required.", nameof(panel));

        // Columns are double-centred per block rather than once per partner. That loop called
        // DistanceCorrelation for every pair, and each of those calls centred BOTH of its arguments,
        // so every column paid the O(steps^2) centring and its n x n allocation (assets - 1) times;
        // here a column block is centred once for each row block at or before it.
        //
        // BOUNDED, because the cache is the trade: holding every column's centred matrix costs
        // assets * steps^2 * 8 bytes, which at 500 assets over a 250-step panel is about 250 MB and
        // grows with the paper's ~5,000-firm scale. Columns are processed in blocks sized to a fixed
        // memory budget, so only a row block and a column block of centred matrices are resident at once;
        // each block pair's covariances come from one matrix product.
        const long CentringBudgetBytes = 64L * 1024 * 1024;
        long perColumnBytes = (long)steps * steps * sizeof(double);
        int blockSize = perColumnBytes > 0
            ? (int)Math.Max(1, Math.Min(assets, CentringBudgetBytes / perColumnBytes))
            : assets;

        var engine = AiDotNetEngine.Current;
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var series = engine.TensorTranspose(panel);                              // [assets, steps]

        // The squared distance covariance of every block pair on or above the diagonal, a block of rows
        // against a block of columns at a time, so at most two blocks of centred matrices are resident.
        // Each column's squared distance variance is the diagonal of its own diagonal block.
        var starts = new List<int>();
        for (int start = 0; start < assets; start += blockSize)
            starts.Add(start);
        int blocks = starts.Count;
        int CountOf(int b) => Math.Min(assets, starts[b] + blockSize) - starts[b];
        var covariances = new Tensor<T>[blocks, blocks];
        var variances = new Tensor<T>[blocks];
        for (int r = 0; r < blocks; r++)
        {
            var rowCentred = DoubleCentredDistances(engine.TensorNarrow(series, 0, starts[r], CountOf(r)), steps);
            for (int c = r; c < blocks; c++)
            {
                var colCentred = c == r
                    ? rowCentred
                    : DoubleCentredDistances(engine.TensorNarrow(series, 0, starts[c], CountOf(c)), steps);
                covariances[r, c] = MeanProducts(rowCentred, colCentred, steps);
            }

            variances[r] = DiagonalOf(covariances[r, r], CountOf(r));
        }

        // Correlate the upper blocks and mirror each into the lower triangle with a transpose, so the
        // matrix is exactly symmetric.
        var correlations = new Tensor<T>[blocks, blocks];
        for (int r = 0; r < blocks; r++)
            for (int c = r; c < blocks; c++)
                correlations[r, c] = DistanceCorrelationFromCentred(covariances[r, c], variances[r], variances[c]);

        var rowBlocks = new Tensor<T>[blocks];
        for (int r = 0; r < blocks; r++)
        {
            var columnBlocks = new Tensor<T>[blocks];
            for (int c = 0; c < blocks; c++)
                columnBlocks[c] = c >= r ? correlations[r, c] : engine.TensorTranspose(correlations[c, r]).Contiguous();
            rowBlocks[r] = blocks == 1 ? columnBlocks[0] : engine.TensorConcatenate(columnBlocks, 1);
        }

        // A series is perfectly dependent on itself.
        var dcor = blocks == 1 ? rowBlocks[0] : engine.TensorConcatenate(rowBlocks, 0);
        var identity = engine.TensorEye<T>(assets);
        return engine.TensorWhere(identity, identity, dcor);
    }

    /// <summary>
    /// An undirected edge of the filtered graph.
    /// </summary>
    /// <param name="Source">Lower node index.</param>
    /// <param name="Target">Higher node index.</param>
    /// <param name="Weight">Edge weight, the distance correlation.</param>
    public readonly record struct GraphEdge(int Source, int Target, double Weight);

    /// <summary>
    /// Filters a dense dependency matrix to a near-maximal-weight PLANAR subgraph using TMFG.
    /// </summary>
    /// <param name="dependencies">Symmetric <c>[assets, assets]</c> weights.</param>
    /// <returns>The retained edges: exactly <c>3(n - 2)</c> of them once <c>n &gt;= 3</c>.</returns>
    /// <remarks>
    /// <para>
    /// The algorithm seeds a tetrahedron from the four most strongly connected nodes, then repeatedly
    /// inserts the remaining node whose total weight to some triangular face is largest, splitting
    /// that face into three. Every insertion adds 3 edges and 2 faces, so the result is a maximal
    /// planar graph: <c>3(n - 2)</c> edges for <c>n</c> nodes.
    /// </para>
    /// <para>
    /// Greedy by construction — the paper describes TMFG as finding a NEAR-maximal planar subgraph,
    /// not a provably optimal one. That is the trade being made for the ability to run at thousands of
    /// nodes, where an exact planar-maximum-weight search is intractable.
    /// </para>
    /// </remarks>
    public IReadOnlyList<GraphEdge> FilterTmfg(Tensor<T> dependencies)
    {
        if (dependencies == null) throw new ArgumentNullException(nameof(dependencies));
        if (dependencies.Shape.Length != 2 || dependencies.Shape[0] != dependencies.Shape[1])
            throw new ArgumentException("dependencies must be a square [assets, assets] matrix.",
                nameof(dependencies));

        int n = dependencies.Shape[0];
        if (n < 3)
            throw new ArgumentException(
                $"TMFG needs at least 3 nodes to form a face; got {n}.", nameof(dependencies));

        double W(int i, int j) => NumOps.ToDouble(dependencies[(i * n) + j]);

        // Seed: the four nodes with the largest total connection strength. Starting from the most
        // connected core is what keeps the greedy insertions near-maximal.
        var strength = new double[n];
        for (int i = 0; i < n; i++)
        {
            double s = 0.0;
            for (int j = 0; j < n; j++) if (i != j) s += W(i, j);
            strength[i] = s;
        }

        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        // Sort a copy: Array.Sort reorders its key array too, and `strength` is the caller's
        // computed result. The previous `strength.Clone() as double[] ?? strength` documented the
        // opposite intent -- the cast cannot fail for a double[], so the fallback was dead code that
        // read as permission to sort the original in place.
        var sortKeys = (double[])strength.Clone();
        Array.Sort(sortKeys, order);
        Array.Reverse(order);

        int seedCount = Math.Min(4, n);
        var inGraph = new List<int>();
        var edges = new List<GraphEdge>();
        var faces = new List<(int A, int B, int C)>();

        for (int i = 0; i < seedCount; i++) inGraph.Add(order[i]);

        for (int i = 0; i < seedCount; i++)
            for (int j = i + 1; j < seedCount; j++)
                edges.Add(MakeEdge(inGraph[i], inGraph[j], W(inGraph[i], inGraph[j])));

        if (seedCount == 4)
        {
            // The four triangular faces of a tetrahedron.
            faces.Add((inGraph[0], inGraph[1], inGraph[2]));
            faces.Add((inGraph[0], inGraph[1], inGraph[3]));
            faces.Add((inGraph[0], inGraph[2], inGraph[3]));
            faces.Add((inGraph[1], inGraph[2], inGraph[3]));
        }
        else
        {
            faces.Add((inGraph[0], inGraph[1], inGraph[2]));
        }

        var remaining = new List<int>();
        for (int i = seedCount; i < n; i++) remaining.Add(order[i]);

        while (remaining.Count > 0)
        {
            int bestNode = -1, bestFace = -1;
            double bestGain = double.NegativeInfinity;

            for (int r = 0; r < remaining.Count; r++)
            {
                int node = remaining[r];
                for (int f = 0; f < faces.Count; f++)
                {
                    var (fa, fb, fc) = faces[f];
                    double gain = W(node, fa) + W(node, fb) + W(node, fc);
                    if (gain > bestGain)
                    {
                        bestGain = gain;
                        bestNode = node;
                        bestFace = f;
                    }
                }
            }

            // bestFace stays -1 when every gain comparison is false, which is what a NaN weight
            // produces: NaN > bestGain is false for every face. faces[-1] then threw
            // ArgumentOutOfRangeException from deep inside the loop with nothing to act on.
            // FilterTmfg is public and takes a caller-supplied matrix, so this is reachable input,
            // not an internal invariant.
            if (bestFace < 0 || bestNode < 0)
            {
                throw new ArgumentException(
                    "No triangular face could be selected. The dependency matrix contains no finite "
                    + "gain for any remaining asset, which happens when it holds NaN entries.",
                    nameof(dependencies));
            }

            var (a, b, c) = faces[bestFace];
            edges.Add(MakeEdge(bestNode, a, W(bestNode, a)));
            edges.Add(MakeEdge(bestNode, b, W(bestNode, b)));
            edges.Add(MakeEdge(bestNode, c, W(bestNode, c)));

            // Replace the filled face with the three it splits into: +3 edges, +2 faces per node,
            // which is what keeps the running graph maximal planar at every step.
            faces.RemoveAt(bestFace);
            faces.Add((bestNode, a, b));
            faces.Add((bestNode, a, c));
            faces.Add((bestNode, b, c));

            remaining.Remove(bestNode);
        }

        return edges;
    }

    private static GraphEdge MakeEdge(int u, int v, double weight) =>
        u <= v ? new GraphEdge(u, v, weight) : new GraphEdge(v, u, weight);

    /// <summary>
    /// The maximum edge count of a maximal planar graph on <paramref name="nodeCount"/> nodes:
    /// <c>3(n - 2)</c>.
    /// </summary>
    public static int MaxPlanarEdges(int nodeCount)
    {
        if (nodeCount < 3)
            throw new ArgumentOutOfRangeException(nameof(nodeCount), nodeCount, "Need at least 3 nodes.");
        return 3 * (nodeCount - 2);
    }

    /// <summary>
    /// Builds a symmetric adjacency mask from an edge list, for use as the GAT's neighbourhood.
    /// </summary>
    /// <remarks>
    /// The diagonal is set, because a GAT node attends to ITSELF as well as its neighbours; omitting
    /// self-loops would discard each asset's own features from its updated representation.
    /// </remarks>
    public Tensor<T> AdjacencyMask(IReadOnlyList<GraphEdge> edges, int nodeCount)
    {
        if (edges == null) throw new ArgumentNullException(nameof(edges));
        if (nodeCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(nodeCount), nodeCount, "nodeCount must be positive.");

        var rows = new int[edges.Count * 2];
        var cols = new int[edges.Count * 2];
        int written = 0;
        foreach (var e in edges)
        {
            if (e.Source < 0 || e.Source >= nodeCount || e.Target < 0 || e.Target >= nodeCount)
                throw new ArgumentException(
                    $"Edge ({e.Source}, {e.Target}) is outside a {nodeCount}-node graph.", nameof(edges));

            rows[written] = e.Source;
            cols[written] = e.Target;
            written++;
            rows[written] = e.Target;
            cols[written] = e.Source;
            written++;
        }

        // Self-loops on the diagonal plus both directions of every edge.
        var engine = AiDotNetEngine.Current;
        var mask = engine.TensorEye<T>(nodeCount);
        if (written == 0)
            return mask;

        var rowIndex = new Tensor<int>(new[] { written }, new Vector<int>(rows));
        var colIndex = new Tensor<int>(new[] { written }, new Vector<int>(cols));
        var ones = engine.TensorFullLike(new Tensor<T>(new[] { written }), NumOps.One);
        return engine.TensorIndexPut(mask, new[] { rowIndex, colIndex }, ones, false);
    }
}
