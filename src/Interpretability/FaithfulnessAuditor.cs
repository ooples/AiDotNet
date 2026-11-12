using AiDotNet.Helpers;
using AiDotNet.Interpretability.Helpers;
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
    public T DeletionAuc { get; init; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Insertion AUC: mean model output as top-attributed features are progressively inserted into a
    /// baseline. <b>Higher is better</b> — a faithful explanation restores the output quickly.
    /// </summary>
    public T InsertionAuc { get; init; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Comprehensiveness (ERASER): output drop when the top-attributed features are removed.
    /// <b>Higher is better</b> — the highlighted features are truly important.
    /// </summary>
    public T Comprehensiveness { get; init; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Sufficiency (ERASER): output drop when ONLY the top-attributed features are kept.
    /// <b>Lower (nearer zero) is better</b> — the highlighted features alone nearly reproduce the output.
    /// </summary>
    public T Sufficiency { get; init; } = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// A single normalized faithfulness score in roughly [0, 1] combining the above (insertion minus
    /// deletion, plus comprehensiveness minus sufficiency). Higher means the attributions are more
    /// faithful to the model.
    /// </summary>
    public T FaithfulnessScore { get; init; } = MathHelper.GetNumericOperations<T>().Zero;

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
    public FaithfulnessReport<T> Audit(Func<Vector<T>, T> score, Matrix<T> data, Vector<T> attributions)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n == 0 || d == 0) return new FaithfulnessReport<T> { SampleCount = 0 };

        var engine = AiDotNetEngine.Current;

        // Baseline = column means (the value a "removed" feature reverts to).
        var baseline = engine.ReduceMean(Tensor<T>.FromMatrix(data), new[] { 0 }, false);

        // Feature order by descending absolute attribution; rank[f] = position of feature f in that order.
        var order = DescendingAbsoluteOrder(attributions, d);
        var rank = engine.TensorScatter(new Tensor<T>(new[] { d }), new Tensor<int>(order, new[] { d }),
            engine.TensorArange<T>(0, d, 1), 0);

        int topK = Math.Max(1, (int)Math.Round(_topFraction * d));
        int stepSize = Math.Max(1, d / _maxSteps);

        // Row s of the step mask marks the features already perturbed after curve step s: the first
        // min(d, (s + 1) * stepSize) features in attribution order.
        var stepEnds = engine.TensorMin(engine.TensorArange<T>(stepSize, d + stepSize, stepSize), _numOps.FromDouble(d));
        int steps = stepEnds.Length;
        var stepShape = new[] { steps, d };
        var stepMask = engine.TensorLessThan(
            engine.TensorBroadcastTo(rank.Reshape(new[] { 1, d }), stepShape),
            engine.TensorBroadcastTo(stepEnds.Reshape(new[] { steps, 1 }), stepShape));
        var topMask = engine.TensorLessThan(rank, _numOps.FromDouble(topK));

        var fullScores = new List<T>(n);
        var deletionAucs = new List<T>(n);
        var insertionAucs = new List<T>(n);
        var removedTopScores = new List<T>(n);
        var keptTopScores = new List<T>(n);
        for (int i = 0; i < n; i++)
        {
            var x = data.GetRow(i).AsTensor();
            T full = score(x.ToVector());
            fullScores.Add(full);

            // Curves track the ABSOLUTE deviation of the output from the full-input output as features are
            // perturbed — magnitudes, so a symmetric feature's large effect does not average to zero.
            deletionAucs.Add(CurveAuc(score, x, baseline, stepMask, stepShape, full, deletion: true));
            insertionAucs.Add(CurveAuc(score, x, baseline, stepMask, stepShape, full, deletion: false));

            // Comprehensiveness: removing the top-k should move the output a lot (magnitude).
            removedTopScores.Add(score(engine.TensorWhere(topMask, baseline, x).ToVector()));

            // Sufficiency: keeping ONLY the top-k should nearly reproduce the output (small residual).
            keptTopScores.Add(score(engine.TensorWhere(topMask, x, baseline).ToVector()));
        }

        // Deletion AUC = mean deviation as top features are REMOVED (higher = faithful: output collapses
        // quickly). Insertion AUC = mean deviation as top features are INSERTED (lower = faithful: output
        // is restored quickly). Comprehensiveness higher / Sufficiency lower = faithful.
        var fullVector = new Vector<T>(fullScores);
        T deletion = engine.Mean(new Vector<T>(deletionAucs));
        T insertion = engine.Mean(new Vector<T>(insertionAucs));
        T comp = engine.Mean(engine.Abs(engine.Subtract(fullVector, new Vector<T>(removedTopScores))));
        T suff = engine.Mean(engine.Abs(engine.Subtract(fullVector, new Vector<T>(keptTopScores))));
        T raw = _numOps.Add(_numOps.Subtract(deletion, insertion), _numOps.Subtract(comp, suff));
        T squashed = _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, _numOps.Exp(_numOps.Negate(raw))));

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

    /// <summary>
    /// Feature indices ordered by descending absolute attribution, ties kept in index order.
    /// </summary>
    private int[] DescendingAbsoluteOrder(Vector<T> attributions, int d)
    {
        var engine = AiDotNetEngine.Current;
        var magnitudes = engine.TensorAbs(attributions.GetSubVector(0, d).AsTensor());
        return ExplainerTensorOps<T>.StableDescendingOrder(magnitudes);
    }

    private T CurveAuc(Func<Vector<T>, T> score, Tensor<T> x, Tensor<T> baseline, Tensor<T> stepMask, int[] stepShape,
        T full, bool deletion)
    {
        var engine = AiDotNetEngine.Current;

        // Deletion starts from the full input and replaces features with the baseline; insertion starts
        // from the baseline and restores features from the input.
        var start = deletion ? x : baseline;
        var target = deletion ? baseline : x;
        var steps = engine.TensorWhere(stepMask, engine.TensorBroadcastTo(target.Reshape(new[] { 1, x.Length }), stepShape),
            engine.TensorBroadcastTo(start.Reshape(new[] { 1, x.Length }), stepShape));

        var trajectory = new List<T>(stepShape[0] + 1) { score(start.ToVector()) };
        for (int s = 0; s < stepShape[0]; s++)
        {
            trajectory.Add(score(steps.GetSlice(s).ToVector()));
        }

        // Trapezoidal mean of the deviation trajectory.
        var deviation = engine.TensorAbs(engine.TensorSubtractScalar(new Vector<T>(trajectory).AsTensor(), full));
        T area = engine.TensorSum(engine.TensorTrapezoid(deviation));
        return _numOps.Divide(area, _numOps.FromDouble(Math.Max(1, trajectory.Count - 1)));
    }
}
