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
    private readonly double _ceWeight;
    private readonly double _maeWeight;

    /// <summary>
    /// Creates the objective.
    /// </summary>
    /// <param name="crossEntropy">The CE term, weighted by <paramref name="ceWeight"/>.</param>
    /// <param name="predictedTokenCount">Returns the CIF predictor's per-instance token count
    /// (<c>sum_t alpha_t</c>) from the most recent forward, or null when unavailable — in which case the
    /// MAE term is skipped rather than guessed at.</param>
    /// <param name="ceWeight">gamma in Eq 6.</param>
    /// <param name="maeWeight">Weight on the MAE term; Eq 6 states it unweighted, so 1.0 is faithful.</param>
    /// <exception cref="ArgumentNullException">If a required argument is null.</exception>
    public ParaformerObjective(
        LossFunctionBase<T> crossEntropy,
        Func<Tensor<T>?> predictedTokenCount,
        double ceWeight = 1.0,
        double maeWeight = 1.0)
    {
        _crossEntropy = crossEntropy ?? throw new ArgumentNullException(nameof(crossEntropy));
        _predictedTokenCount = predictedTokenCount ?? throw new ArgumentNullException(nameof(predictedTokenCount));
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

        // Target length: the number of supervised positions, taken from the target's leading axis. This
        // is supervision, not a differentiable quantity, so it is built as a constant.
        int positions = target.Rank > 1 ? target.Shape[0] : target.Length;
        var targetLength = new Tensor<T>((int[])predictedCount._shape.Clone());
        var lengthValue = NumOps.FromDouble(positions);
        for (int i = 0; i < targetLength.Length; i++)
        {
            targetLength.Data.Span[i] = lengthValue;
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
