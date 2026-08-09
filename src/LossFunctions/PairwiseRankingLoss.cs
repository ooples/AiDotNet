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
        // Normalize by the ITEM count n, not the pair count (~n^2). The original RankNet (Burges et al.
        // 2005) accumulates per-pair gradients (sum); dividing by the pair count shrinks the per-item
        // gradient to O(1/n), too weak for the optimizer to learn on larger groups (the constant-output
        // collapse). Per-item normalization keeps the gradient O(1) at any group size while staying
        // consistent with CalculateDerivative below.
        return NumOps.FromDouble(lossSum / n);
    }

    /// <summary>
    /// Computes the loss as a tape-differentiable scalar tensor for automatic backpropagation.
    /// </summary>
    /// <param name="predicted">The predicted scores tensor from the forward pass.</param>
    /// <param name="target">The true relevance/return tensor.</param>
    /// <returns>A rank-0 scalar tensor holding the weighted RankNet loss.</returns>
    /// <remarks>
    /// <para>
    /// RankNet is a genuinely pairwise objective, so the forward is built over the full n x n grid of
    /// score differences rather than element-wise. The grid is formed with two rank-1 matrix products
    /// against constant one-vectors -- <c>A[i,j] = s_i</c> and <c>B[i,j] = s_j</c> -- because the
    /// element-wise engine operations require matching shapes and do not broadcast.
    /// </para>
    /// <para>
    /// L = (1/n) * sum_ij W[i,j] * softplus(-(s_i - s_j))
    /// </para>
    /// <para>
    /// <c>W</c> depends only on the targets -- it is the tail weight where <c>a_i &gt; a_j</c> and zero
    /// elsewhere, so ties and reversed pairs drop out -- which makes it a constant with respect to the
    /// predictions and therefore safe to materialize outside the tape.
    /// </para>
    /// <para>
    /// Normalization is per ITEM (divide by n), not per pair. Dividing by the pair count (~n^2) shrinks
    /// the per-item gradient to O(1/n), which is too weak for the optimizer to learn from on larger
    /// groups and collapses the model to a constant output.
    /// </para>
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var targetVec = target.ToVector();
        int n = targetVec.Length;

        if (predicted.Length != n)
        {
            throw new ArgumentException(
                $"Predicted length ({predicted.Length}) must match target length ({n}).",
                nameof(predicted));
        }

        // Pair weights are a function of the targets alone, so they are a tape constant.
        var extremity = ComputeExtremities(targetVec);
        var weights = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++)
        {
            double ai = NumOps.ToDouble(targetVec[i]);
            for (int j = 0; j < n; j++)
            {
                // Only ordered pairs where i is the true winner contribute; ties contribute nothing.
                if (ai <= NumOps.ToDouble(targetVec[j])) continue;
                weights[(i * n) + j] = NumOps.FromDouble(PairWeight(extremity, i, j));
            }
        }

        var onesRow = new Tensor<T>(new[] { 1, n });
        var onesCol = new Tensor<T>(new[] { n, 1 });
        for (int k = 0; k < n; k++)
        {
            onesRow[k] = NumOps.One;
            onesCol[k] = NumOps.One;
        }

        var scoreCol = Engine.Reshape(predicted, new[] { n, 1 });
        var scoreRow = Engine.Reshape(predicted, new[] { 1, n });

        var sI = Engine.TensorMatMul(scoreCol, onesRow);   // [n,n], sI[i,j] = s_i
        var sJ = Engine.TensorMatMul(onesCol, scoreRow);   // [n,n], sJ[i,j] = s_j

        var negDiff = Engine.TensorMultiplyScalar(
            Engine.TensorSubtract(sI, sJ), NumOps.FromDouble(-1.0));

        var perPair = Engine.TensorMultiply(Engine.Softplus(negDiff), weights);
        var summed = Engine.ReduceSum(perPair, new[] { 0, 1 }, keepDims: false);

        return Engine.TensorDivideScalar(summed, NumOps.FromDouble(n));
    }
}
