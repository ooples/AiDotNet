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
    private Tensor<T> ComputeExtremities(Tensor<T> actual)
    {
        int n = actual.Length;

        if (_tailWeightPower == 0.0 || n == 0)
        {
            return new Tensor<T>(new[] { n });
        }

        // Median of the target values (robust center for "how extreme" each name is).
        var sorted = Engine.TensorSort(actual, 0, descending: false).Values;
        T median = (n % 2 == 1)
            ? sorted.GetFlat(n / 2)
            : NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(sorted.GetFlat((n / 2) - 1), sorted.GetFlat(n / 2)));

        // Largest absolute deviation from the median normalizes extremity to [0, 1].
        var deviation = Engine.TensorAbs(Engine.TensorSubtractScalar(actual, median));
        T maxDev = Engine.TensorMaxValue(deviation);

        if (!NumOps.GreaterThan(maxDev, NumOps.Zero))
        {
            // All targets identical: no tails to emphasize.
            return new Tensor<T>(new[] { n });
        }

        return Engine.TensorDivideScalar(deviation, maxDev);
    }

    /// <summary>
    /// Pair weights W[i, j]: the tail weight (1 + max(e_i, e_j))^power where a_i &gt; a_j, and 0
    /// elsewhere, so ties and reversed pairs contribute nothing. With tail weighting disabled the
    /// weight of every ordered pair is exactly 1 (standard RankNet).
    /// </summary>
    private Tensor<T> ComputePairWeights(Tensor<T> actual)
    {
        return ComputePairWeights(actual, ComputeExtremities(actual), 0, actual.Length);
    }

    /// <summary>
    /// Rows <paramref name="start"/> .. <paramref name="start"/> + <paramref name="rows"/> - 1 of the pair
    /// weights W, as a [rows, n] block, for the precomputed per-item <paramref name="extremity"/>.
    /// </summary>
    private Tensor<T> ComputePairWeights(Tensor<T> actual, Tensor<T> extremity, int start, int rows)
    {
        int n = actual.Length;
        var pairShape = new[] { rows, n };
        var columnShape = new[] { rows, 1 };
        var rowShape = new[] { 1, n };
        var blockActual = Engine.TensorNarrow(actual, 0, start, rows).Contiguous();

        // Only ordered pairs where i is the true winner contribute; ties contribute nothing.
        var winner = Engine.TensorGreaterThan(
            Engine.TensorBroadcastTo(Engine.Reshape(blockActual, columnShape), pairShape),
            Engine.TensorBroadcastTo(Engine.Reshape(actual, rowShape), pairShape));

        if (_tailWeightPower == 0.0)
        {
            return winner;
        }

        var pairExtremity = Engine.TensorMax(
            Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorNarrow(extremity, 0, start, rows).Contiguous(), columnShape), pairShape),
            Engine.TensorBroadcastTo(Engine.Reshape(extremity, rowShape), pairShape));
        var tailWeight = Engine.TensorPower(
            Engine.TensorAddScalar(pairExtremity, NumOps.One), NumOps.FromDouble(_tailWeightPower));
        return Engine.TensorMultiply(winner, tailWeight);
    }

    /// <summary>
    /// Upper bound on the elements of one [rows, n] pair block in <see cref="CalculateLoss"/>, so the
    /// non-tape loss holds O(n) rather than several dense [n, n] tensors.
    /// </summary>
    private const int PairBlockElements = 1 << 18;

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
        if (n == 0) return NumOps.Zero;

        // L = (1/n) * sum_ij W[i,j] * softplus(-(s_i - s_j)), accumulated over bounded blocks of rows i
        var targets = actual.AsTensor();
        var extremity = ComputeExtremities(targets);
        var scores = predicted.AsTensor();
        int block = Math.Max(1, Math.Min(n, PairBlockElements / n));
        T weightSum = NumOps.Zero;
        T lossSum = NumOps.Zero;
        for (int start = 0; start < n; start += block)
        {
            int rows = Math.Min(block, n - start);
            var pairShape = new[] { rows, n };
            var weights = ComputePairWeights(targets, extremity, start, rows);
            var sI = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorNarrow(scores, 0, start, rows).Contiguous(), new[] { rows, 1 }), pairShape);
            var sJ = Engine.TensorBroadcastTo(Engine.Reshape(scores, new[] { 1, n }), pairShape);
            var negDiff = Engine.TensorNegate(Engine.TensorSubtract(sI, sJ));
            weightSum = NumOps.Add(weightSum, Engine.TensorSum(weights));
            lossSum = NumOps.Add(lossSum, Engine.TensorSum(Engine.TensorMultiply(Engine.Softplus(negDiff), weights)));
        }

        if (!NumOps.GreaterThan(weightSum, NumOps.Zero)) return NumOps.Zero;

        // Normalize by the ITEM count n, not the pair count (~n^2). The original RankNet (Burges et al.
        // 2005) accumulates per-pair gradients (sum); dividing by the pair count shrinks the per-item
        // gradient to O(1/n), too weak for the optimizer to learn on larger groups (the constant-output
        // collapse). Per-item normalization keeps the gradient O(1) at any group size while staying
        // consistent with CalculateDerivative below.
        return NumOps.Divide(lossSum, NumOps.FromDouble(n));
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
        var weights = ComputePairWeights(targetVec.AsTensor());

        var onesRow = Tensor<T>.CreateDefault(new[] { 1, n }, NumOps.One);
        var onesCol = Tensor<T>.CreateDefault(new[] { n, 1 }, NumOps.One);

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
