using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.InformationTheoretic;

/// <summary>
/// Kraskov MI — Mutual Information estimation using k-nearest neighbors (KSG estimator).
/// </summary>
/// <remarks>
/// <para>
/// The Kraskov-Stoegbauer-Grassberger (KSG) estimator computes mutual information
/// using nearest-neighbor distances in the joint and marginal spaces. It's non-parametric
/// and works well for both linear and nonlinear dependencies.
/// </para>
/// <para>
/// <b>Algorithm (KSG Algorithm 1):</b>
/// <list type="number">
/// <item>For each point, find its k-th nearest neighbor in the joint (X,Y) space using Chebyshev distance</item>
/// <item>Let epsilon_i = distance to k-th neighbor in joint space</item>
/// <item>Count n_x(i) = number of points with |x_j - x_i| &lt; epsilon_i</item>
/// <item>Count n_y(i) = number of points with |y_j - y_i| &lt; epsilon_i</item>
/// <item>MI = psi(k) - &lt;psi(n_x + 1) + psi(n_y + 1)&gt; + psi(N)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Most MI estimators assume the data follows a specific distribution
/// (like Gaussian). The Kraskov method doesn't make this assumption — it works by looking
/// at how close data points are to each other in different ways. This makes it more reliable
/// for complex, real-world data.
/// </para>
/// <para>
/// Reference: Kraskov et al. (2004), "Estimating Mutual Information", Physical Review E.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Estimating Mutual Information", "https://doi.org/10.1103/PhysRevE.69.066138", Year = 2004, Authors = "Alexander Kraskov, Harald Stoegbauer, Peter Grassberger")]
public class KraskovMIAlgorithm<T> : InfoTheoreticBase<T>
{
    private const double CoincidenceTolerance = 1e-15;

    /// <summary>
    /// Maximum number of entries in one block of the pairwise distance matrix (rows per block times n),
    /// which bounds the working memory of the neighbour search.
    /// </summary>
    private const int MaxDistanceBlockElements = 1 << 21;

    /// <inheritdoc/>
    public override string Name => "KraskovMI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    private readonly double _threshold;

    public KraskovMIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyInfoOptions(options);
        _threshold = options?.EdgeThreshold ?? 0.1;
        if (double.IsNaN(_threshold) || double.IsInfinity(_threshold) || _threshold < 0)
            throw new ArgumentException("EdgeThreshold must be non-negative and finite.");
        if (KNeighbors < 1)
            throw new ArgumentException($"KNeighbors must be at least 1, got {KNeighbors}.");
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int k = Math.Min(KNeighbors, n - 1);

        if (d < 2)
            throw new ArgumentException($"KraskovMI requires at least 2 variables, got {d}.");
        if (k < 1)
            throw new ArgumentException($"KraskovMI requires at least 2 samples (got {n}). Configured KNeighbors={KNeighbors} was clamped to {k}.");
        if (n < k + 2)
            throw new ArgumentException($"KraskovMI requires at least {k + 2} samples for k={k}, got {n}.");

        var weights = new Dictionary<int, double>();

        // Compute KSG mutual information for each pair
        // KSG is an undirected MI estimator — emit undirected adjacency (both directions)
        // and leave orientation to a dedicated algorithm (e.g., PC, FCI)
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                double mi = ComputeKSGMutualInformation(data, i, j, n, k);

                if (mi > _threshold)
                {
                    // Undirected: MI is symmetric, place in both directions
                    weights[i * d + j] = mi;
                    weights[j * d + i] = mi;
                }
            }
        }

        return CausalTensorHelpers.FromStatistics<T>(d, d, weights);
    }

    private double ComputeKSGMutualInformation(Matrix<T> data, int col1, int col2, int n, int k)
    {
        // Column vectors as rows [1 x n], compared against blocks of points [len x 1]
        var x = Engine.GetColumn(data, col1).AsTensor();
        var y = Engine.GetColumn(data, col2).AsTensor();
        var xRow = x.Reshape(new[] { 1, n });
        var yRow = y.Reshape(new[] { 1, n });
        var xColumn = x.Reshape(new[] { n, 1 });
        var yColumn = y.Reshape(new[] { n, 1 });
        T farAway = NumOps.MaxValue;
        T tolerance = NumOps.FromDouble(CoincidenceTolerance);
        T one = NumOps.One;

        // For each point, find k-th nearest neighbor distance in joint space (Chebyshev/max norm)
        // Then count marginal neighbors within that distance. Points are processed in row blocks so
        // the [len x n] distance matrices stay bounded.
        int block = Math.Max(1, Math.Min(n, MaxDistanceBlockElements / Math.Max(1, n)));
        var psiX = new List<Tensor<T>>();
        var psiY = new List<Tensor<T>>();
        for (int start = 0; start < n; start += block)
        {
            int len = Math.Min(block, n - start);
            var shape = new[] { len, n };

            // self[r, start + r] = 1: a point is never its own neighbour
            var selfParts = new List<Tensor<T>>();
            if (start > 0) selfParts.Add(new Tensor<T>(new[] { len, start }));
            selfParts.Add(Engine.TensorEye<T>(len));
            if (start + len < n) selfParts.Add(new Tensor<T>(new[] { len, n - start - len }));
            var self = Engine.TensorConcatenate(selfParts.ToArray(), 1);
            var far = Engine.TensorFullLike(self, farAway);

            var dx = Engine.TensorWhere(self, far, Engine.TensorAbs(Engine.TensorSubtract(
                Engine.TensorBroadcastTo(xRow, shape),
                Engine.TensorBroadcastTo(Engine.TensorSlice(xColumn, new[] { start, 0 }, new[] { len, 1 }), shape))));
            var dy = Engine.TensorWhere(self, far, Engine.TensorAbs(Engine.TensorSubtract(
                Engine.TensorBroadcastTo(yRow, shape),
                Engine.TensorBroadcastTo(Engine.TensorSlice(yColumn, new[] { start, 0 }, new[] { len, 1 }), shape))));

            // Find k-th nearest neighbor distance
            var (sortedDistances, _) = Engine.TensorSort(Engine.TensorMax(dx, dy), 1, false);
            var epsilon = Engine.TensorSlice(sortedDistances, new[] { 0, k - 1 }, new[] { len, 1 });

            // Handle zero-distance ties: when duplicate samples collapse epsilon to 0,
            // strict < epsilon would leave nx/ny at 0, producing spurious MI.
            // Degenerate case (k-th neighbor at distance ~0): count exact duplicates in each
            // marginal and use at least one neighbor.
            var degenerate = Engine.TensorLessThan(epsilon, tolerance);
            var radius = Engine.TensorBroadcastTo(
                Engine.TensorWhere(degenerate, Engine.TensorFullLike(epsilon, tolerance), epsilon), shape);

            // Count marginal neighbors within the radius (strict inequality per KSG)
            var nx = Engine.ReduceSum(Engine.TensorLessThan(dx, radius), new[] { 1 }, true);
            var ny = Engine.ReduceSum(Engine.TensorLessThan(dy, radius), new[] { 1 }, true);
            nx = Engine.TensorWhere(degenerate, Engine.TensorMax(nx, one), nx);
            ny = Engine.TensorWhere(degenerate, Engine.TensorMax(ny, one), ny);

            psiX.Add(Engine.TensorDigamma(Engine.TensorAddScalar(nx, one)));
            psiY.Add(Engine.TensorDigamma(Engine.TensorAddScalar(ny, one)));
        }

        T sumPsiNx = Engine.TensorSum(Engine.TensorConcatenate(psiX.ToArray(), 0));
        T sumPsiNy = Engine.TensorSum(Engine.TensorConcatenate(psiY.ToArray(), 0));

        // MI = psi(k) - <psi(nx+1) + psi(ny+1)> / N + psi(N)
        T mi = NumOps.Add(NumOps.Subtract(NumOps.FromDouble(Digamma(k)),
            NumOps.Divide(NumOps.Add(sumPsiNx, sumPsiNy), NumOps.FromDouble(n))), NumOps.FromDouble(Digamma(n)));
        return Math.Max(NumOps.ToDouble(mi), 0); // MI is non-negative
    }

    private static double Digamma(int x)
    {
        // Digamma function approximation for positive integers
        // psi(x) = -gamma + sum_{k=1}^{x-1} 1/k for integer x
        if (x <= 0) return -0.5772156649; // Euler-Mascheroni for x=0

        double result = -0.5772156649015329; // Euler-Mascheroni constant
        for (int k = 1; k < x; k++)
            result += 1.0 / k;

        return result;
    }

    private static double Digamma(double x)
    {
        // Asymptotic series for non-integer x
        if (x <= 0) return -0.5772156649;

        double result = 0;
        while (x < 6)
        {
            result -= 1.0 / x;
            x += 1;
        }

        result += Math.Log(x) - 1.0 / (2.0 * x);
        double x2 = 1.0 / (x * x);
        result -= x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 / 252.0));
        return result;
    }
}
