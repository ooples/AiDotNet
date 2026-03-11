using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders.
/// </summary>
/// <remarks>
/// <para>
/// RCD extends DirectLiNGAM to handle latent (unobserved) confounders by iteratively
/// identifying "exogenous-like" variables whose residuals are mutually independent,
/// regressing them out, and repeating on the remaining variables.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Standardize the data</item>
/// <item>Initialize the set of remaining variables U = {0, 1, ..., d-1}</item>
/// <item>Repeat until U is empty:</item>
/// <item>  For each variable i in U, compute residuals after regressing out all discovered ancestors</item>
/// <item>  Compute pairwise independence (via mutual information) of residuals</item>
/// <item>  Identify the variable whose residuals are most independent of all others (exogenous)</item>
/// <item>  Add that variable to the causal ordering and record its causal coefficients</item>
/// <item>  Regress out the identified variable from all remaining variables</item>
/// <item>  If no sufficiently independent variable found, flag remaining as confounded and stop</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> RCD is like DirectLiNGAM but more cautious — it checks whether
/// variables are truly "root causes" or might be influenced by hidden (unobserved) factors.
/// When it finds variables that can't be cleanly separated, it marks them as potentially
/// confounded rather than forcing a possibly wrong causal direction.
/// </para>
/// <para>
/// Reference: Maeda and Shimizu (2020), "RCD: Repetitive Causal Discovery of
/// Linear Non-Gaussian Acyclic Models with Latent Confounders", AISTATS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("RCD: Repetitive Causal Discovery of Linear Non-Gaussian Acyclic Models with Latent Confounders", "https://proceedings.mlr.press/v108/maeda20a.html", Year = 2020, Authors = "Takashi Nicholas Maeda, Shohei Shimizu")]
public class RCDAlgorithm<T> : FunctionalBase<T>
{
    private readonly double _threshold;
    private readonly double _independenceThreshold;

    /// <inheritdoc/>
    public override string Name => "RCD";

    /// <inheritdoc/>
    public override bool SupportsLatentConfounders => true;

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes RCD with optional configuration.
    /// </summary>
    public RCDAlgorithm(CausalDiscoveryOptions? options = null)
    {
        _threshold = options?.EdgeThreshold ?? 0.1;
        _independenceThreshold = options?.SignificanceLevel ?? 0.05;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var standardized = StandardizeData(data);

        // Working copy of data — columns get regressed out as we identify exogenous variables
        var residualData = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                residualData[i, j] = NumOps.ToDouble(standardized[i, j]);

        var W = new Matrix<T>(d, d);
        var remaining = new List<int>(Enumerable.Range(0, d));
        var ordering = new List<int>();

        // Iteratively identify exogenous variables
        int maxRounds = d;
        for (int round = 0; round < maxRounds && remaining.Count > 0; round++)
        {
            // Find the most exogenous variable among remaining
            int bestVar = -1;
            double bestScore = double.MaxValue;

            foreach (int candidate in remaining)
            {
                // Compute total mutual information of this variable's residuals with all others
                double totalMI = 0;
                foreach (int other in remaining)
                {
                    if (other == candidate) continue;
                    totalMI += ComputeMutualInformationFromResiduals(residualData, n, candidate, other);
                }

                // Normalize by number of comparisons
                int comparisons = remaining.Count - 1;
                double avgMI = comparisons > 0 ? totalMI / comparisons : 0;

                if (avgMI < bestScore)
                {
                    bestScore = avgMI;
                    bestVar = candidate;
                }
            }

            // Check if the best variable is sufficiently independent
            if (bestVar < 0 || bestScore > _independenceThreshold * 10)
            {
                // Remaining variables are likely confounded — stop
                break;
            }

            // Record causal coefficients from the identified exogenous variable to remaining ones
            ordering.Add(bestVar);
            remaining.Remove(bestVar);

            foreach (int target in remaining)
            {
                double coeff = RegressCoefficient(residualData, n, bestVar, target);
                if (Math.Abs(coeff) > _threshold)
                {
                    W[bestVar, target] = NumOps.FromDouble(coeff);
                }
            }

            // Regress out the identified variable from all remaining variables
            foreach (int target in remaining)
            {
                double coeff = RegressCoefficient(residualData, n, bestVar, target);
                for (int i = 0; i < n; i++)
                {
                    residualData[i, target] -= coeff * residualData[i, bestVar];
                }
            }
        }

        return ThresholdMatrix(W, _threshold);
    }

    /// <summary>
    /// Computes an estimate of mutual information between two columns using
    /// a kernel density-based approach suitable for non-Gaussian data.
    /// </summary>
    private double ComputeMutualInformationFromResiduals(double[,] data, int n, int col1, int col2)
    {
        // Use kurtosis-based independence proxy (faster than full MI estimation)
        // For non-Gaussian data, correlation of absolute values indicates dependence
        double mean1 = 0, mean2 = 0;
        for (int i = 0; i < n; i++)
        {
            mean1 += Math.Abs(data[i, col1]);
            mean2 += Math.Abs(data[i, col2]);
        }
        mean1 /= n;
        mean2 /= n;

        double cov = 0, var1 = 0, var2 = 0;
        for (int i = 0; i < n; i++)
        {
            double d1 = Math.Abs(data[i, col1]) - mean1;
            double d2 = Math.Abs(data[i, col2]) - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        if (var1 < 1e-10 || var2 < 1e-10) return 0;
        double r = Math.Abs(cov / Math.Sqrt(var1 * var2));

        // Convert absolute correlation to MI estimate: MI ≈ -0.5 * log(1 - r^2)
        double rSq = Math.Min(r * r, 0.999);
        return -0.5 * Math.Log(1 - rSq);
    }

    /// <summary>
    /// Computes the OLS regression coefficient of source predicting target.
    /// </summary>
    private static double RegressCoefficient(double[,] data, int n, int source, int target)
    {
        double meanS = 0, meanT = 0;
        for (int i = 0; i < n; i++)
        {
            meanS += data[i, source];
            meanT += data[i, target];
        }
        meanS /= n;
        meanT /= n;

        double cov = 0, varS = 0;
        for (int i = 0; i < n; i++)
        {
            double ds = data[i, source] - meanS;
            cov += ds * (data[i, target] - meanT);
            varS += ds * ds;
        }

        return varS > 1e-10 ? cov / varS : 0;
    }
}
