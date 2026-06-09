using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the pairwise RankNet learning-to-rank loss with an optional tail-weighting knob.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This loss treats every prediction vector and target vector as a single <i>ranking group</i>
/// (for example, all stocks in one asset class on one date). It does not learn to match each
/// target value pointwise; instead it learns the correct <b>relative order</b> of the items.
/// For every ordered pair (i, j) where the true score of i is greater than the true score of j,
/// the RankNet loss penalizes the model when it predicts s_i &#8804; s_j:
/// </para>
/// <para>
/// loss(i, j) = log(1 + exp(-(s_i - s_j)))
/// </para>
/// <para>
/// where s_i and s_j are the predicted scores. Summed over all such pairs and averaged by the
/// number of pairs, this yields a smooth, convex, gradient-friendly surrogate for "fraction of
/// pairs ordered incorrectly". With the default tail weight (1.0) it is the standard RankNet
/// loss of Burges et al. (2005).
/// </para>
/// <para>
/// <b>Tail weighting.</b> In cross-sectional trading only the extremes are actionable: you go
/// long the top names and short the bottom names, and the middle of the ranking is never traded.
/// The optional <c>tailWeightPower</c> knob makes each pair contribute in proportion to how
/// extreme its two items are in the target distribution. Each item is assigned an extremity in
/// [0, 1] measuring its distance from the median target (0 at the median, 1 at the most extreme
/// top or bottom name). A pair's weight is
/// </para>
/// <para>
/// w(i, j) = (1 + max(extremity_i, extremity_j))^tailWeightPower
/// </para>
/// <para>
/// With <c>tailWeightPower = 0</c> every weight is 1 and you recover plain RankNet (backward
/// compatible). With a positive power the biggest movers dominate the loss, so the model spends
/// its capacity getting the tradeable tails right.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most loss functions ask "how close is each predicted number to the
/// true number?". A ranking loss asks a different, often more useful, question: "did you put the
/// items in the right <i>order</i>?". If you only care about buying the best things and selling
/// the worst things, the exact predicted values do not matter &#8212; only their order does.
/// This loss looks at every pair of items, checks whether the one with the higher true value also
/// got the higher predicted score, and nudges the model when it got a pair backwards.
/// The tail-weighting option lets you tell the model "I care much more about getting the
/// top and bottom right than the middle", which is exactly what a long/short trader wants.
/// </para>
/// <para>
/// <b>How it plugs in.</b> This is a standard <see cref="ILossFunction{T}"/>, so any
/// gradient-trained model in AiDotNet can use it. For a neural-network cross-sectional ranker:
/// <code>
/// var ranker = new NeuralNetwork&lt;double&gt;(architecture, optimizer,
///     lossFunction: new PairwiseRankingLoss&lt;double&gt;(tailWeightPower: 1.0));
/// var model = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(ranker)
///     .BuildAsync(features, forwardReturns);
/// </code>
/// Each training example's feature matrix is one cross-section (one date / one segment) and the
/// target vector is the signed forward returns; the network outputs one score per name and the
/// loss ranks them.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Ranking)]
[LossTask(LossTask.Ranking)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, ExpectedOutput = OutputType.Continuous)]
public class PairwiseRankingLoss<T> : LossFunctionBase<T>
{
    private readonly double _tailWeightPower;

    /// <summary>
    /// Creates a new pairwise RankNet ranking loss.
    /// </summary>
    /// <param name="tailWeightPower">
    /// Controls how strongly pairs involving extreme (top/bottom) target values are emphasized.
    /// <c>0</c> (the default) gives every pair equal weight and reproduces the standard RankNet
    /// loss. Larger positive values increasingly concentrate the loss on the most extreme movers.
    /// Must be non-negative.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="tailWeightPower"/> is negative.</exception>
    public PairwiseRankingLoss(double tailWeightPower = 0.0)
    {
        if (tailWeightPower < 0.0 || double.IsNaN(tailWeightPower) || double.IsInfinity(tailWeightPower))
        {
            throw new ArgumentOutOfRangeException(nameof(tailWeightPower),
                "Tail weight power must be a finite, non-negative value.");
        }

        _tailWeightPower = tailWeightPower;
    }

    /// <summary>
    /// Gets the tail-weighting power this loss was configured with. 0 means standard RankNet.
    /// </summary>
    public double TailWeightPower => _tailWeightPower;

    /// <summary>
    /// Computes a per-item extremity weight in [0, 1] for tail weighting: 0 at the median target,
    /// 1 at the most extreme top/bottom target. Returns all-ones when tail weighting is disabled
    /// or the spread is degenerate, so the loss reduces exactly to standard RankNet.
    /// </summary>
    private double[] ComputeExtremities(Vector<T> actual)
    {
        int n = actual.Length;
        var extremity = new double[n];

        if (_tailWeightPower == 0.0)
        {
            for (int i = 0; i < n; i++) extremity[i] = 0.0;
            return extremity;
        }

        // Median of the target values (robust center for "how extreme" each name is).
        var sorted = new double[n];
        for (int i = 0; i < n; i++) sorted[i] = NumOps.ToDouble(actual[i]);
        Array.Sort(sorted);
        double median = (n % 2 == 1)
            ? sorted[n / 2]
            : 0.5 * (sorted[(n / 2) - 1] + sorted[n / 2]);

        // Largest absolute deviation from the median normalizes extremity to [0, 1].
        double maxDev = 0.0;
        for (int i = 0; i < n; i++)
        {
            double dev = Math.Abs(NumOps.ToDouble(actual[i]) - median);
            if (dev > maxDev) maxDev = dev;
        }

        if (maxDev <= 0.0)
        {
            // All targets identical: no tails to emphasize.
            for (int i = 0; i < n; i++) extremity[i] = 0.0;
            return extremity;
        }

        for (int i = 0; i < n; i++)
        {
            extremity[i] = Math.Abs(NumOps.ToDouble(actual[i]) - median) / maxDev;
        }

        return extremity;
    }

    /// <summary>
    /// Weight of the pair (i, j) given the precomputed per-item extremities.
    /// </summary>
    private double PairWeight(double[] extremity, int i, int j)
    {
        if (_tailWeightPower == 0.0) return 1.0;
        double e = Math.Max(extremity[i], extremity[j]);
        return Math.Pow(1.0 + e, _tailWeightPower);
    }

    /// <summary>
    /// Numerically stable softplus: log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|)).
    /// </summary>
    private static double Softplus(double x)
    {
        double ax = Math.Abs(x);
        return Math.Max(x, 0.0) + Math.Log(1.0 + Math.Exp(-ax));
    }

    /// <summary>
    /// Logistic sigmoid 1 / (1 + exp(-x)), computed in a numerically stable way.
    /// </summary>
    private static double Sigmoid(double x)
    {
        if (x >= 0.0)
        {
            double z = Math.Exp(-x);
            return 1.0 / (1.0 + z);
        }
        else
        {
            double z = Math.Exp(x);
            return z / (1.0 + z);
        }
    }

    /// <summary>
    /// Calculates the (tail-weighted) pairwise RankNet loss over all valid pairs in the group.
    /// </summary>
    /// <param name="predicted">The predicted scores, one per item in the ranking group.</param>
    /// <param name="actual">The true relevance/return values, one per item.</param>
    /// <returns>The weighted-average pairwise loss; 0 when there are no orderable pairs.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        int n = predicted.Length;
        var extremity = ComputeExtremities(actual);

        double lossSum = 0.0;
        double weightSum = 0.0;

        for (int i = 0; i < n; i++)
        {
            double ai = NumOps.ToDouble(actual[i]);
            double si = NumOps.ToDouble(predicted[i]);
            for (int j = 0; j < n; j++)
            {
                double aj = NumOps.ToDouble(actual[j]);
                // Only count ordered pairs where i is the true winner. Ties contribute nothing.
                if (ai <= aj) continue;

                double sj = NumOps.ToDouble(predicted[j]);
                double w = PairWeight(extremity, i, j);

                // log(1 + exp(-(s_i - s_j)))
                lossSum += w * Softplus(-(si - sj));
                weightSum += w;
            }
        }

        if (weightSum <= 0.0) return NumOps.Zero;
        return NumOps.FromDouble(lossSum / weightSum);
    }

    /// <summary>
    /// Calculates the gradient of the (tail-weighted) pairwise RankNet loss with respect to each
    /// predicted score.
    /// </summary>
    /// <param name="predicted">The predicted scores, one per item in the ranking group.</param>
    /// <param name="actual">The true relevance/return values, one per item.</param>
    /// <returns>A vector of partial derivatives dL/ds_k, one per item.</returns>
    /// <remarks>
    /// For a single pair (i, j) with i the true winner, dloss/ds_i = -sigma(-(s_i - s_j)) and
    /// dloss/ds_j = +sigma(-(s_i - s_j)). These contributions accumulate over every pair an item
    /// participates in, each scaled by the pair weight and divided by the total weight so the
    /// gradient matches the averaged loss above.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        int n = predicted.Length;
        var extremity = ComputeExtremities(actual);
        var grad = new double[n];

        double weightSum = 0.0;

        for (int i = 0; i < n; i++)
        {
            double ai = NumOps.ToDouble(actual[i]);
            double si = NumOps.ToDouble(predicted[i]);
            for (int j = 0; j < n; j++)
            {
                double aj = NumOps.ToDouble(actual[j]);
                if (ai <= aj) continue;

                double sj = NumOps.ToDouble(predicted[j]);
                double w = PairWeight(extremity, i, j);

                // d/ds_i log(1+exp(-(s_i - s_j))) = -sigmoid(-(s_i - s_j))
                double g = Sigmoid(-(si - sj));
                grad[i] += w * (-g);
                grad[j] += w * (g);
                weightSum += w;
            }
        }

        var result = new T[n];
        if (weightSum <= 0.0)
        {
            for (int k = 0; k < n; k++) result[k] = NumOps.Zero;
            return new Vector<T>(result);
        }

        for (int k = 0; k < n; k++)
        {
            result[k] = NumOps.FromDouble(grad[k] / weightSum);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the loss as a tape-differentiable scalar tensor for automatic backpropagation.
    /// </summary>
    /// <param name="predicted">The predicted scores tensor from the forward pass.</param>
    /// <param name="target">The true relevance/return tensor.</param>
    /// <returns>A scalar tensor whose gradient w.r.t. <paramref name="predicted"/> equals the analytic RankNet gradient.</returns>
    /// <remarks>
    /// <para>
    /// The pairwise RankNet objective is not a pointwise function of the predictions, so it cannot
    /// be expressed directly with the broadcasting tensor primitives. Instead this builds a
    /// first-order surrogate scalar
    /// </para>
    /// <para>
    /// L_tape = &#931;_k predicted_k * stopgrad(g_k)
    /// </para>
    /// <para>
    /// where g_k is the exact analytic gradient from <see cref="CalculateDerivative"/> treated as a
    /// constant (no gradient flows through the target side). Because dL_tape/dpredicted_k = g_k,
    /// the gradient that flows back through the tape is exactly the RankNet gradient, which is all
    /// the optimizer needs. The reported scalar value is the true RankNet loss for monitoring.
    /// </para>
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var predVec = predicted.ToVector();
        var targetVec = target.ToVector();

        // Constant analytic gradient (the target side carries no gradient).
        var g = CalculateDerivative(predVec, targetVec);
        var gradConst = new Tensor<T>(predicted._shape, g);

        // L_tape = sum(predicted * g)  =>  dL_tape/dpredicted = g  (g is a leaf constant).
        var weighted = Engine.TensorMultiply(predicted, gradConst);
        var allAxes = Enumerable.Range(0, weighted.Shape.Length).ToArray();
        var surrogate = Engine.ReduceSum(weighted, allAxes, keepDims: false);

        // Add the true loss as a detached constant so GetLastLoss reports a meaningful value
        // without changing the gradient (constant => zero gradient contribution).
        double trueLoss = NumOps.ToDouble(CalculateLoss(predVec, targetVec));
        double surrogateValue = 0.0;
        for (int k = 0; k < predVec.Length; k++)
        {
            surrogateValue += NumOps.ToDouble(predVec[k]) * NumOps.ToDouble(g[k]);
        }

        // surrogate currently equals surrogateValue; shift it to report trueLoss while keeping
        // the same gradient (adding a scalar constant does not change the derivative).
        var shift = NumOps.FromDouble(trueLoss - surrogateValue);
        return Engine.TensorAddScalar(surrogate, shift);
    }
}
