using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability;

/// <summary>
/// The faithfulness of a feature-attribution explanation to a model: do the features it highlights
/// actually drive the model's output? Reported as deletion/insertion AUC and ERASER-style
/// comprehensiveness/sufficiency, plus a single normalized score.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class FaithfulnessReport<T>
{
    /// <summary>
    /// Deletion AUC (Most-Relevant-First): mean model output as top-attributed features are progressively
    /// removed. <b>Lower is better</b> — a faithful explanation makes the output collapse quickly.
    /// </summary>
    public double DeletionAuc { get; init; }

    /// <summary>
    /// Insertion AUC: mean model output as top-attributed features are progressively inserted into a
    /// baseline. <b>Higher is better</b> — a faithful explanation restores the output quickly.
    /// </summary>
    public double InsertionAuc { get; init; }

    /// <summary>
    /// Comprehensiveness (ERASER): output drop when the top-attributed features are removed.
    /// <b>Higher is better</b> — the highlighted features are truly important.
    /// </summary>
    public double Comprehensiveness { get; init; }

    /// <summary>
    /// Sufficiency (ERASER): output drop when ONLY the top-attributed features are kept.
    /// <b>Lower (nearer zero) is better</b> — the highlighted features alone nearly reproduce the output.
    /// </summary>
    public double Sufficiency { get; init; }

    /// <summary>
    /// A single normalized faithfulness score in roughly [0, 1] combining the above (insertion minus
    /// deletion, plus comprehensiveness minus sufficiency). Higher means the attributions are more
    /// faithful to the model.
    /// </summary>
    public double FaithfulnessScore { get; init; }

    /// <summary>Number of rows the audit was averaged over.</summary>
    public int SampleCount { get; init; }
}

/// <summary>
/// Audits how faithful a per-feature attribution vector is to a model, using perturbation curves. This is
/// the check mainstream explainability libraries omit: they produce attributions but do not measure
/// whether removing the "important" features actually changes the model's output.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class FaithfulnessAuditor<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _maxSteps;
    private readonly double _topFraction;

    /// <summary>Creates a faithfulness auditor.</summary>
    /// <param name="maxSteps">Perturbation steps along each curve (features are bucketed into this many). Defaults to 20.</param>
    /// <param name="topFraction">Fraction of features treated as "top" for comprehensiveness/sufficiency. Defaults to 0.2.</param>
    public FaithfulnessAuditor(int maxSteps = 20, double topFraction = 0.2)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _maxSteps = Math.Max(2, maxSteps);
        _topFraction = Math.Max(0.01, Math.Min(1.0, topFraction));
    }

    /// <summary>
    /// Audits <paramref name="attributions"/> against <paramref name="score"/> (a scalar model output for
    /// an input) over the rows of <paramref name="data"/>.
    /// </summary>
    public FaithfulnessReport<T> Audit(Func<Vector<T>, double> score, Matrix<T> data, Vector<T> attributions)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n == 0 || d == 0) return new FaithfulnessReport<T> { SampleCount = 0 };

        // Baseline = column means (the value a "removed" feature reverts to).
        var baseline = new double[d];
        for (int j = 0; j < d; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += _numOps.ToDouble(data[i, j]);
            baseline[j] = sum / n;
        }

        // Feature order by descending absolute attribution.
        var order = Enumerable.Range(0, d)
            .OrderByDescending(j => Math.Abs(_numOps.ToDouble(attributions[j])))
            .ToArray();

        int topK = Math.Max(1, (int)Math.Round(_topFraction * d));
        int stepSize = Math.Max(1, d / _maxSteps);

        double delSum = 0, insSum = 0, compSum = 0, suffSum = 0;
        for (int i = 0; i < n; i++)
        {
            var x = new double[d];
            for (int j = 0; j < d; j++) x[j] = _numOps.ToDouble(data[i, j]);
            double full = score(ToVector(x));

            // Curves track the ABSOLUTE deviation of the output from the full-input output as features are
            // perturbed — magnitudes, so a symmetric feature's large effect does not average to zero.
            delSum += CurveAuc(score, x, baseline, order, stepSize, full, deletion: true);
            insSum += CurveAuc(score, x, baseline, order, stepSize, full, deletion: false);

            // Comprehensiveness: removing the top-k should move the output a lot (magnitude).
            var removedTop = (double[])x.Clone();
            for (int k = 0; k < topK; k++) removedTop[order[k]] = baseline[order[k]];
            compSum += Math.Abs(full - score(ToVector(removedTop)));

            // Sufficiency: keeping ONLY the top-k should nearly reproduce the output (small residual).
            var keptTop = (double[])baseline.Clone();
            for (int k = 0; k < topK; k++) keptTop[order[k]] = x[order[k]];
            suffSum += Math.Abs(full - score(ToVector(keptTop)));
        }

        // Deletion AUC = mean deviation as top features are REMOVED (higher = faithful: output collapses
        // quickly). Insertion AUC = mean deviation as top features are INSERTED (lower = faithful: output
        // is restored quickly). Comprehensiveness higher / Sufficiency lower = faithful.
        double deletion = delSum / n, insertion = insSum / n, comp = compSum / n, suff = suffSum / n;
        double raw = (deletion - insertion) + (comp - suff);
        double squashed = 1.0 / (1.0 + Math.Exp(-raw));

        return new FaithfulnessReport<T>
        {
            DeletionAuc = deletion,
            InsertionAuc = insertion,
            Comprehensiveness = comp,
            Sufficiency = suff,
            FaithfulnessScore = squashed,
            SampleCount = n,
        };
    }

    private double CurveAuc(Func<Vector<T>, double> score, double[] x, double[] baseline, int[] order, int stepSize, double full, bool deletion)
    {
        int d = x.Length;
        // Deletion starts from the full input; insertion starts from the baseline.
        var work = deletion ? (double[])x.Clone() : (double[])baseline.Clone();
        var trajectory = new List<double> { Math.Abs(score(ToVector(work)) - full) };

        for (int pos = 0; pos < d; pos += stepSize)
        {
            int end = Math.Min(d, pos + stepSize);
            for (int p = pos; p < end; p++)
            {
                int feature = order[p];
                work[feature] = deletion ? baseline[feature] : x[feature];
            }

            trajectory.Add(Math.Abs(score(ToVector(work)) - full));
        }

        // Trapezoidal mean of the deviation trajectory.
        double area = 0;
        for (int k = 1; k < trajectory.Count; k++) area += 0.5 * (trajectory[k] + trajectory[k - 1]);
        return area / Math.Max(1, trajectory.Count - 1);
    }

    private Vector<T> ToVector(double[] x)
    {
        var v = new Vector<T>(x.Length);
        for (int i = 0; i < x.Length; i++) v[i] = _numOps.FromDouble(x[i]);
        return v;
    }
}
