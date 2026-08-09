using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Paraformer's training objective: <c>gamma * L_CE + L_MAE</c> (Gao et al., arXiv 2206.08317, Eq 6).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Eq 6 states <c>L_total = gamma*L_CE + L_MAE + L_Nwerr</c>. This composes the first two terms; MWER
/// is a sampling-based sequence-level term requiring N-best decoding, so it is not folded in here.
/// </para>
/// <para>
/// The MAE term is what distinguishes this from plain cross-entropy: it supervises the CIF predictor's
/// predicted TOKEN COUNT against the target length. Paraformer §2.2/2.4 describe it as guiding "the
/// predictor to convergence", and without it that head trains unsupervised.
/// </para>
/// <para>
/// The predicted count is supplied by a callback rather than derived from
/// <c>(predicted, target)</c>, because it is produced inside the CIF layer (as
/// <c>sum_t alpha_t</c>) and is not recoverable from the logits. The callback returns a TAPE-TRACKED
/// tensor, so gradient from this term flows back into the alpha predictor's weights — which only became
/// possible once the CIF forward was expressed in differentiable Engine ops.
/// </para>
/// </remarks>
public sealed class ParaformerObjective<T> : LossFunctionBase<T>
{
    private readonly LossFunctionBase<T> _crossEntropy;
    private readonly Func<Tensor<T>?> _predictedTokenCount;

    /// <summary>
    /// Returns the per-instance count of REAL target tokens, or null to fall back to the target's
    /// sequence-axis length.
    /// </summary>
    /// <remarks>
    /// Sits next to <see cref="_predictedTokenCount"/> deliberately: the MAE term compares the two,
    /// and a length inferred from a shape cannot exclude padding. Without this the term supervises
    /// the alpha head toward the padded sequence length rather than the utterance length.
    /// </remarks>
    private readonly Func<Tensor<T>?>? _targetTokenCount;

    private readonly double _ceWeight;
    private readonly double _maeWeight;

    /// <summary>
    /// Creates the objective.
    /// </summary>
    /// <param name="crossEntropy">The CE term, weighted by <paramref name="ceWeight"/>.</param>
    /// <param name="predictedTokenCount">Returns the CIF predictor's per-instance token count
    /// (<c>sum_t alpha_t</c>) from the most recent forward, or null when unavailable — in which case the
    /// MAE term is skipped rather than guessed at.</param>
    /// <param name="targetTokenCount">
    /// Returns the per-instance number of REAL target tokens, excluding padding. Optional; when it
    /// is null the count falls back to the target's sequence-axis length, which counts padded
    /// positions as tokens. Supply it whenever targets are padded, because a shape alone cannot
    /// distinguish a pad from a token and the MAE term then trains the alpha head toward the padded
    /// length.
    /// </param>
    /// <param name="ceWeight">gamma in Eq 6.</param>
    /// <param name="maeWeight">Weight on the MAE term; Eq 6 states it unweighted, so 1.0 is faithful.</param>
    /// <exception cref="ArgumentNullException">If a required argument is null.</exception>
    public ParaformerObjective(
        LossFunctionBase<T> crossEntropy,
        Func<Tensor<T>?> predictedTokenCount,
        Func<Tensor<T>?>? targetTokenCount = null,
        double ceWeight = 1.0,
        double maeWeight = 1.0)
    {
        _crossEntropy = crossEntropy ?? throw new ArgumentNullException(nameof(crossEntropy));
        _predictedTokenCount = predictedTokenCount ?? throw new ArgumentNullException(nameof(predictedTokenCount));
        _targetTokenCount = targetTokenCount;
        _ceWeight = ceWeight;
        _maeWeight = maeWeight;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Scalar path: the CE term only. The MAE term needs the CIF predictor's tensor output, which the
    /// vector API does not carry; returning CE alone here keeps this path consistent with what it can
    /// actually see rather than silently reporting a different number from the tape path.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        => NumOps.Multiply(NumOps.FromDouble(_ceWeight), _crossEntropy.CalculateLoss(predicted, actual));

    /// <inheritdoc/>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var d = _crossEntropy.CalculateDerivative(predicted, actual);
        var scale = NumOps.FromDouble(_ceWeight);
        var scaled = new Vector<T>(d.Length);
        for (int i = 0; i < d.Length; i++)
        {
            scaled[i] = NumOps.Multiply(scale, d[i]);
        }

        return scaled;
    }

    /// <inheritdoc/>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var ce = _crossEntropy.ComputeTapeLoss(predicted, target);
        var total = _ceWeight == 1.0
            ? ce
            : Engine.TensorMultiplyScalar(ce, NumOps.FromDouble(_ceWeight));

        var predictedCount = _predictedTokenCount();
        if (predictedCount is null || _maeWeight == 0.0)
        {
            return total;
        }
        // Target length: the number of supervised TOKENS. Supervision, not a differentiable
        // quantity, so it is built as a constant.
        //
        // A caller-supplied count wins, because only the caller can exclude padding. Falling back to
        // a shape reads the SEQUENCE axis -- Shape[1] for [batch, sequence] and
        // [batch, sequence, vocab] alike. It previously read Shape[0], which for every batched target
        // is the BATCH SIZE. The CIF predictor was therefore trained to emit the batch size, so
        // L_MAE drove the alpha head to a constant unrelated to utterance length: the term the paper
        // adds to supervise an otherwise unsupervised head was mis-supervising it instead. Silently,
        // because a plausible loss number came back either way.
        var suppliedTargetCount = _targetTokenCount?.Invoke();

        // Shape from the PREDICTED count via the public accessor. Reading Tensor<T>._shape reached
        // into its private backing field from loss code.
        var countShape = (int[])predictedCount.Shape.ToArray().Clone();
        var targetLength = suppliedTargetCount ?? new Tensor<T>(countShape);
        if (suppliedTargetCount is not null)
        {
            // A SUPPLIED COUNT IS NOT CHECKED BY ANYTHING ELSE. It goes straight into TensorSubtract
            // below, where a [1] against a predicted [8], or an [8, 1] against an [8], BROADCASTS
            // rather than fails -- and the MAE term then supervises the alpha head against the wrong
            // per-instance counts while returning a finite, plausible loss. The fallback branch cannot
            // reach this because it builds its tensor from countShape.
            if (suppliedTargetCount.Shape.Length != predictedCount.Shape.Length)
            {
                throw new ArgumentException(
                    $"Paraformer's supplied target token count has rank {suppliedTargetCount.Shape.Length}, "
                    + $"but the predicted count has rank {predictedCount.Shape.Length}. They must match "
                    + "exactly; a broadcastable shape would silently score against the wrong counts.");
            }

            for (int d = 0; d < countShape.Length; d++)
            {
                if (suppliedTargetCount.Shape[d] == countShape[d]) continue;

                throw new ArgumentException(
                    $"Paraformer's supplied target token count differs from the predicted count at "
                    + $"dimension {d} ({suppliedTargetCount.Shape[d]} vs {countShape[d]}).");
            }
        }
        else
        {
            int positions = target.Rank > 1 ? target.Shape[1] : target.Length;
            var lengthValue = NumOps.FromDouble(positions);
            for (int i = 0; i < targetLength.Length; i++)
            {
                targetLength.Data.Span[i] = lengthValue;
            }
        }

        // L_MAE = mean |sum_t alpha_t - targetLength|, on the tape so it reaches the alpha predictor.
        var deviation = Engine.TensorAbs(Engine.TensorSubtract(predictedCount, targetLength));
        var axes = new int[deviation.Shape.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            axes[i] = i;
        }

        var mae = Engine.ReduceMean(deviation, axes, keepDims: false);
        var maeTerm = _maeWeight == 1.0
            ? mae
            : Engine.TensorMultiplyScalar(mae, NumOps.FromDouble(_maeWeight));

        // Both terms are already rank-0: each ends in a full ReduceMean with keepDims: false. The
        // AsScalarRank helper that used to wrap them reshaped to rank-1 [1], which is the opposite
        // of the documented ComputeTapeLoss contract -- a [1] root leaves the tape with no scalar to
        // seed the backward from. Adding two rank-0 tensors needs no normalization at all.
        return Engine.TensorAdd(total, maeTerm);
    }
}
