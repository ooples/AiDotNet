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
[ResearchPaper("RCD: Repetitive Causal Discovery of Linear Non-Gaussian Acyclic Models with Latent Confounders", "https://proceedings.mlr.press/v108/maeda20a.html", Year = 2020, Authors = "Takashi Nicholas Maeda, Shohei Shimizu")]
public class RCDAlgorithm<T> : FunctionalBase<T>
{
    private readonly double _threshold;

    /// <summary>
    /// Cutoff on the confounding-evidence score used at line ~149 to trigger the
    /// "remaining variables are confounded" stop. The score is
    /// <c>Σᵢⱼ min(0, DiffMutualInfo(residualᵢ, residualⱼ))²</c> (sum of squared
    /// negative diffs) — a NON-NEGATIVE quantity whose scale is bounded by
    /// MI² per pair × candidate count, NOT a statistical p-value.
    /// <para>
    /// We source the cutoff from <see cref="CausalDiscoveryOptions.SignificanceLevel"/>
    /// because that option is the shared "how strict" knob across algorithms and
    /// no dedicated RCD-scale option exists in the current API. The default of
    /// <c>0.05</c> is a rough scale match for the DiffMI residual scale — NOT a
    /// significance level in the α-of-a-hypothesis-test sense. Callers who need
    /// tighter calibration can override via <c>SignificanceLevel</c>; a
    /// dedicated <c>ConfoundingEvidenceCutoff</c> option is future work
    /// (would need calibration tests to justify).
    /// </para>
    /// </summary>
    private readonly double _confoundingScoreCutoff;

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
        // See _confoundingScoreCutoff doc: sourced from SignificanceLevel as a
        // scale proxy, NOT a statistical α. Default 0.05 is calibrated for the
        // sum-of-squared-negative-DiffMI scale, not a p-value threshold.
        _confoundingScoreCutoff = options?.SignificanceLevel ?? 0.05;
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

            // Find the MOST EXOGENOUS variable among remaining using the DirectLiNGAM / RCD
            // criterion (Shimizu et al. 2011 DirectLiNGAM; Maeda & Shimizu 2020 RCD): the
            // exogenous variable is the one whose regression RESIDUALS of the other variables are
            // most INDEPENDENT of it. For each candidate, regress every other remaining variable
            // on the candidate and measure the mutual information between the candidate and each
            // residual r_j = x_j - reg(x_j ~ x_candidate). A true root leaves those residuals
            // independent of it (score ~0); a descendant does not.
            //
            // NOTE: scoring RAW pairwise MI between the candidate and the other variables (the
            // previous implementation) is wrong — a root cause has HIGH raw MI with its
            // descendants because it drives them, so raw-MI minimisation systematically selects a
            // LEAF and inverts the causal order, leaving the true directed edges undetected.
            // Score each candidate by the DirectLiNGAM entropy criterion: for every other
            // variable j, DiffMutualInfo(candidate, j) > 0 iff candidate → j (candidate is the
            // cause). Accumulate min(0, DiffMI)² — the squared EVIDENCE that the candidate is an
            // EFFECT of some j. The exogenous variable has no such evidence (score ≈ 0), so we
            // pick the argmin. This uses Hyvärinen's max-entropy differential-entropy
            // approximation, which detects the non-Gaussian dependence that identifies causal
            // direction even for near-collinear variables (a histogram MI cannot).
            foreach (int candidate in remaining)
            {
                double effectEvidence = 0;
                foreach (int other in remaining)
                {
                    if (other == candidate) continue;
                    double diffMI = DiffMutualInfo(residualData, n, candidate, other);
                    double neg = Math.Min(0.0, diffMI);
                    effectEvidence += neg * neg;
                }

                if (effectEvidence < bestScore)
                {
                    bestScore = effectEvidence;
                    bestVar = candidate;
                }
            }

            // Check if the best variable is sufficiently independent using MI cutoff directly
            if (bestVar < 0)
            {
                break;
            }

            // bestScore = sum of squared negative DiffMI across all (candidate, other)
            // residual pairs — a non-negative confounding-evidence measure. When it
            // exceeds the scale-matched cutoff, remaining variables can't be cleanly
            // ordered under RCD and are treated as confounded. See _confoundingScoreCutoff
            // doc for the scale rationale (NOT a statistical significance level).
            if (bestScore > _confoundingScoreCutoff && ordering.Count > 0)
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
    /// Hyvärinen's (1998) maximum-entropy approximation of the differential entropy of a
    /// STANDARDIZED (zero-mean, unit-variance) variable. Lower entropy ⇒ more non-Gaussian.
    /// This is the entropy term DirectLiNGAM (Shimizu et al. 2011) — which RCD (Maeda &amp;
    /// Shimizu 2020) builds on — uses to score causal direction, because the non-Gaussianity of
    /// a residual is what identifies direction; a histogram MI cannot resolve this on
    /// near-collinear data.
    /// </summary>
    private static double DifferentialEntropy(double[] u, int n)
    {
        // Constants from Hyvärinen (1998) "New approximations of differential entropy…":
        // the two non-quadratic contrast functions G1(u)=log cosh(u), G2(u)=u·exp(-u²/2),
        // with gamma = E[G1(standard normal)].
        const double k1 = 79.047;
        const double k2 = 7.4129;
        const double gamma = 0.37457;
        double gaussEntropy = (1.0 + Math.Log(2.0 * Math.PI)) / 2.0;

        double m1 = 0, m2 = 0;
        for (int i = 0; i < n; i++)
        {
            double x = u[i];
            m1 += Math.Log(Math.Cosh(x));
            m2 += x * Math.Exp(-0.5 * x * x);
        }
        m1 /= n;
        m2 /= n;

        return gaussEntropy - k1 * (m1 - gamma) * (m1 - gamma) - k2 * m2 * m2;
    }

    /// <summary>
    /// DirectLiNGAM's entropy-based mutual-information difference for the ordered pair (i, j):
    /// a positive value means <c>i → j</c> (variable i is the cause), a negative value means
    /// <c>j → i</c>. Both variables are standardized; the regression residuals in each direction
    /// are r_{i|j} = x_i − ρ·x_j and r_{j|i} = x_j − ρ·x_i (ρ = correlation), and the measure is
    /// [H(x_j) + H(r_{i|j})] − [H(x_i) + H(r_{j|i})] using
    /// <see cref="DifferentialEntropy(double[], int)"/>.
    /// </summary>
    private static double DiffMutualInfo(double[,] data, int n, int i, int j)
    {
        double meanI = 0, meanJ = 0;
        for (int k = 0; k < n; k++) { meanI += data[k, i]; meanJ += data[k, j]; }
        meanI /= n; meanJ /= n;

        double varI = 0, varJ = 0, cov = 0;
        for (int k = 0; k < n; k++)
        {
            double di = data[k, i] - meanI, dj = data[k, j] - meanJ;
            varI += di * di; varJ += dj * dj; cov += di * dj;
        }
        double stdI = Math.Sqrt(varI / n), stdJ = Math.Sqrt(varJ / n);
        if (stdI < 1e-12 || stdJ < 1e-12) return 0;
        double rho = (cov / n) / (stdI * stdJ);
        if (rho > 1) rho = 1; else if (rho < -1) rho = -1;
        double residStd = Math.Sqrt(Math.Max(1e-12, 1.0 - rho * rho));

        var xi = new double[n];
        var xj = new double[n];
        var rIgivenJ = new double[n];
        var rJgivenI = new double[n];
        for (int k = 0; k < n; k++)
        {
            double xis = (data[k, i] - meanI) / stdI;
            double xjs = (data[k, j] - meanJ) / stdJ;
            xi[k] = xis;
            xj[k] = xjs;
            rIgivenJ[k] = (xis - rho * xjs) / residStd;
            rJgivenI[k] = (xjs - rho * xis) / residStd;
        }

        return (DifferentialEntropy(xj, n) + DifferentialEntropy(rIgivenJ, n))
             - (DifferentialEntropy(xi, n) + DifferentialEntropy(rJgivenI, n));
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
