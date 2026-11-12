using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// IGCI (Information-Geometric Causal Inference) — bivariate causal discovery via entropy.
/// </summary>
/// <remarks>
/// <para>
/// IGCI determines causal direction between pairs of variables by exploiting the
/// information-geometric principle: if X causes Y, the marginal distribution of X
/// and the conditional P(Y|X) are independent, which means the slope of Y=f(X)
/// tends to correlate negatively with the density of X at that point.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each pair (X, Y), uniformize both variables to [0,1] via rank transform</item>
/// <item>Sort by X values and compute the average log-slope: sum(log|dy/dx|)/n</item>
/// <item>The causal direction score is: score = mean(log|f'(x)|)</item>
/// <item>If score &lt; 0, X→Y is preferred; if score &gt; 0, Y→X is preferred</item>
/// <item>Build adjacency matrix from pairwise decisions</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> IGCI works by checking whether the relationship between two
/// variables is "smoother" in one direction than the other. If X causes Y, the mapping
/// from X to Y tends to have slopes that anti-correlate with the density of X — a
/// natural geometric property of cause-effect relationships. This method is very fast
/// (pairwise comparisons only) but only works for bivariate, deterministic-ish relationships.
/// </para>
/// <para>
/// Reference: Janzing et al. (2012), "Information-Geometric Approach to Inferring
/// Causal Directions", Artificial Intelligence.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Information-Geometric Approach to Inferring Causal Directions", "https://doi.org/10.1016/j.artint.2012.01.002", Year = 2012, Authors = "Dominik Janzing, Joris Mooij, Kun Zhang, Jan Lemeire, Jakob Zscheischler, Povilas Daniusis, Bernhard Steudel, Bernhard Scholkopf")]
public class IGCIAlgorithm<T> : FunctionalBase<T>
{
    private readonly double _threshold;

    /// <inheritdoc/>
    public override string Name => "IGCI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes IGCI with optional configuration.
    /// </summary>
    public IGCIAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _threshold = options?.EdgeThreshold ?? 0.0;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var W = new Matrix<T>(d, d);

        // Pairwise IGCI score computation
        for (int i = 0; i < d; i++)
        {
            for (int j = i + 1; j < d; j++)
            {
                double score = ComputeIGCIScore(data, n, i, j);

                // score < 0 implies i → j; score > 0 implies j → i
                if (Math.Abs(score) > _threshold)
                {
                    if (score < 0)
                    {
                        W[i, j] = NumOps.FromDouble(-score);
                    }
                    else
                    {
                        W[j, i] = NumOps.FromDouble(score);
                    }
                }
                else
                {
                    // IGCI score ≈ 0 means the relationship is linear or indeterminate.
                    // Per the IGCI paper (Daniusis et al., 2012), the method has no power
                    // to orient linear edges. Leave W[i,j] and W[j,i] as zero (undirected)
                    // rather than using a scale-dependent fallback that can reverse edges
                    // based on variable units instead of causal signal.
                }
            }
        }

        return W;
    }

    /// <summary>
    /// Computes the IGCI score for a pair of variables.
    /// Negative score means col1→col2, positive means col2→col1.
    /// Uses the entropy-based estimator: score = mean(log|f'|) where f maps uniformized X to uniformized Y.
    /// </summary>
    private double ComputeIGCIScore(Matrix<T> data, int n, int col1, int col2)
    {
        // Extract and uniformize both columns via rank transform
        var x = new double[n];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = NumOps.ToDouble(data[i, col1]);
            y[i] = NumOps.ToDouble(data[i, col2]);
        }

        // Affine min-max rescale to [0, 1] — the "uniform reference measure" of
        // the IGCI slope estimator (Daniusis et al. 2010 / Janzing et al. 2012,
        // estimator Ĉ_X→Y rescales both variables' RANGE to [0,1] and averages
        // log-slopes between x-sorted neighbors). The previous code applied a
        // RANK transform to both sides, which collapses every deterministic
        // monotone relationship to the identity map (the ranks of x and f(x)
        // coincide): all slopes become 1, every log-slope is 0, and the score
        // was exactly 0 for ANY monotone data — IGCI could never orient an edge.
        var ux = MinMaxRescale(x);
        var uy = MinMaxRescale(y);

        // Sort by uniformized X to compute slopes dy/dx
        var indices = Enumerable.Range(0, n).ToArray();
        Array.Sort(indices, (a, b) => ux[a].CompareTo(ux[b]));

        // Compute mean log|slope| for X→Y direction
        double sumLogSlope = 0;
        int validCount = 0;

        for (int k = 0; k < n - 1; k++)
        {
            int i0 = indices[k];
            int i1 = indices[k + 1];
            double dx = ux[i1] - ux[i0];
            double dy = uy[i1] - uy[i0];

            if (Math.Abs(dx) > 1e-15)
            {
                double slope = Math.Abs(dy / dx);
                if (slope > 1e-15)
                {
                    sumLogSlope += Math.Log(slope);
                    validCount++;
                }
            }
        }

        if (validCount == 0) return 0;

        // score = mean(log|f'(x)|)
        // Negative means X→Y, positive means Y→X
        return sumLogSlope / validCount;
    }

    /// <summary>
    /// Affinely rescales a vector to [0,1] (the IGCI uniform reference measure:
    /// the variable's RANGE is normalized, preserving the shape of the function
    /// between the variables, unlike a rank transform which destroys it).
    /// </summary>
    private static double[] MinMaxRescale(double[] values)
    {
        int n = values.Length;
        var result = new double[n];

        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            if (values[i] < min) min = values[i];
            if (values[i] > max) max = values[i];
        }

        double range = max - min;
        if (range < 1e-15)
        {
            // Constant variable: map everything to 0 (no slope information).
            return result;
        }

        for (int i = 0; i < n; i++)
            result[i] = (values[i] - min) / range;

        return result;
    }
}
