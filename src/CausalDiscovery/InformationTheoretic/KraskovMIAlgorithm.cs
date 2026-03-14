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
[ModelPaper("Estimating Mutual Information", "https://doi.org/10.1103/PhysRevE.69.066138", Year = 2004, Authors = "Alexander Kraskov, Harald Stoegbauer, Peter Grassberger")]
public class KraskovMIAlgorithm<T> : InfoTheoreticBase<T>
{
    private const double CoincidenceTolerance = 1e-15;

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
            throw new ArgumentException($"KraskovMI requires at least {KNeighbors + 1} samples for k={KNeighbors} (k was clamped to {k}), got {n}.");
        if (n < k + 2)
            throw new ArgumentException($"KraskovMI requires at least {k + 2} samples for k={k}, got {n}.");

        var result = new Matrix<T>(d, d);

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
                    T miVal = NumOps.FromDouble(mi);
                    // Undirected: MI is symmetric, place in both directions
                    result[i, j] = miVal;
                    result[j, i] = miVal;
                }
            }
        }

        return result;
    }

    private double ComputeKSGMutualInformation(Matrix<T> data, int col1, int col2, int n, int k)
    {
        // Extract column vectors
        var x = new double[n];
        var y = new double[n];
        for (int t = 0; t < n; t++)
        {
            x[t] = NumOps.ToDouble(data[t, col1]);
            y[t] = NumOps.ToDouble(data[t, col2]);
        }

        // For each point, find k-th nearest neighbor distance in joint space (Chebyshev/max norm)
        // Then count marginal neighbors within that distance
        double sumPsiNx = 0, sumPsiNy = 0;

        for (int i = 0; i < n; i++)
        {
            // Compute Chebyshev distances to all other points in joint space
            var jointDists = new double[n];
            for (int j = 0; j < n; j++)
            {
                if (j == i) { jointDists[j] = double.MaxValue; continue; }
                jointDists[j] = Math.Max(Math.Abs(x[j] - x[i]), Math.Abs(y[j] - y[i]));
            }

            // Find k-th nearest neighbor distance
            double epsilon = FindKthSmallest(jointDists, n, k);

            // Handle zero-distance ties: when duplicate samples collapse epsilon to 0,
            // strict < epsilon would leave nx/ny at 0, producing spurious MI.
            // Use <= for ties and ensure at least k neighbors are counted.
            if (epsilon < CoincidenceTolerance)
            {
                // Degenerate case: k-th neighbor is at distance ~0
                // Count exact duplicates in each marginal
                int nx0 = 0, ny0 = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    if (Math.Abs(x[j] - x[i]) < CoincidenceTolerance) nx0++;
                    if (Math.Abs(y[j] - y[i]) < CoincidenceTolerance) ny0++;
                }
                sumPsiNx += Digamma(Math.Max(nx0, 1) + 1);
                sumPsiNy += Digamma(Math.Max(ny0, 1) + 1);
                continue;
            }

            // Count marginal neighbors within epsilon (strict inequality per KSG)
            int nx = 0, ny = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                if (Math.Abs(x[j] - x[i]) < epsilon) nx++;
                if (Math.Abs(y[j] - y[i]) < epsilon) ny++;
            }

            sumPsiNx += Digamma(nx + 1);
            sumPsiNy += Digamma(ny + 1);
        }

        // MI = psi(k) - <psi(nx+1) + psi(ny+1)> / N + psi(N)
        double mi = Digamma(k) - (sumPsiNx + sumPsiNy) / n + Digamma(n);
        return Math.Max(mi, 0); // MI is non-negative
    }

    private static double FindKthSmallest(double[] arr, int n, int k)
    {
        // Simple partial sort to find k-th smallest (excluding MaxValue sentinels)
        var sorted = new double[k];
        for (int i = 0; i < k; i++) sorted[i] = double.MaxValue;

        for (int i = 0; i < n; i++)
        {
            if (arr[i] >= sorted[k - 1]) continue;
            sorted[k - 1] = arr[i];

            // Bubble down to maintain sorted order
            for (int j = k - 1; j > 0 && sorted[j] < sorted[j - 1]; j--)
                (sorted[j], sorted[j - 1]) = (sorted[j - 1], sorted[j]);
        }

        return sorted[k - 1];
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
