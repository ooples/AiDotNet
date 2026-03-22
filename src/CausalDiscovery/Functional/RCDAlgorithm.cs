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
    private readonly double _independenceCutoff;

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
        // Dedicated independence cutoff for MI-based test (decoupled from EdgeThreshold)
        _independenceCutoff = options?.SignificanceLevel ?? 0.05;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (d < 2)
            throw new ArgumentException($"RCD requires at least 2 variables, got {d}.");
        if (n < d + 3)
            throw new ArgumentException($"RCD requires at least {d + 3} samples for {d} variables, got {n}.");

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

            // Cache pairwise MI scores within each round to avoid recomputation
            var miCache = new Dictionary<(int, int), double>();
            foreach (int candidate in remaining)
            {
                // Compute total mutual information of this variable's residuals with all others
                double totalMI = 0;
                foreach (int other in remaining)
                {
                    if (other == candidate) continue;
                    var key = (Math.Min(candidate, other), Math.Max(candidate, other));
                    if (!miCache.TryGetValue(key, out double mi))
                    {
                        mi = ComputeEntropyBasedMI(residualData, n, candidate, other);
                        miCache[key] = mi;
                    }
                    totalMI += mi;
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

            // Check if the best variable is sufficiently independent using MI cutoff directly
            if (bestVar < 0)
            {
                break;
            }

            if (bestScore > _independenceCutoff && ordering.Count > 0)
            {
                // Remaining variables are likely confounded — mark as confounded and stop.
                // Per RCD: confounded variables cannot be cleanly ordered, so we do NOT
                // assign edges among them. The zero entries in W for these variables
                // indicate "unidentified due to latent confounding", which is the correct
                // RCD behavior. Callers can check: if W[i,j]==0 AND W[j,i]==0 for
                // remaining variables, those relationships are confounded.
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
    /// Computes mutual information between two columns using entropy-based estimation.
    /// Uses differential entropy via histogram binning: MI = H(X) + H(Y) - H(X,Y).
    /// </summary>
    private static double ComputeEntropyBasedMI(double[,] data, int n, int col1, int col2)
    {
        if (n < 4) return 0;

        // Number of bins via Sturges' rule
        int numBins = Math.Max(3, (int)Math.Ceiling(Math.Log(n, 2) + 1));

        // Find ranges for each variable
        double min1 = double.MaxValue, max1 = double.MinValue;
        double min2 = double.MaxValue, max2 = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double v1 = data[i, col1], v2 = data[i, col2];
            if (v1 < min1) min1 = v1; if (v1 > max1) max1 = v1;
            if (v2 < min2) min2 = v2; if (v2 > max2) max2 = v2;
        }

        double range1 = max1 - min1, range2 = max2 - min2;
        if (range1 < 1e-15 || range2 < 1e-15) return 0;

        // Build joint and marginal histograms
        var joint = new int[numBins, numBins];
        var marginal1 = new int[numBins];
        var marginal2 = new int[numBins];

        for (int i = 0; i < n; i++)
        {
            int b1 = Math.Min((int)((data[i, col1] - min1) / range1 * numBins), numBins - 1);
            int b2 = Math.Min((int)((data[i, col2] - min2) / range2 * numBins), numBins - 1);
            joint[b1, b2]++;
            marginal1[b1]++;
            marginal2[b2]++;
        }

        // MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        double mi = 0;
        double logN = Math.Log(n);
        for (int b1 = 0; b1 < numBins; b1++)
        {
            if (marginal1[b1] == 0) continue;
            for (int b2 = 0; b2 < numBins; b2++)
            {
                if (joint[b1, b2] == 0 || marginal2[b2] == 0) continue;
                mi += (double)joint[b1, b2] / n *
                      (Math.Log(joint[b1, b2]) - Math.Log(marginal1[b1]) - Math.Log(marginal2[b2]) + logN);
            }
        }

        return Math.Max(mi, 0);
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
