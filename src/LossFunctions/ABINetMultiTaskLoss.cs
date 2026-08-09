using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// ABINet's multi-task objective: the weighted sum of the vision, language and fusion losses.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ABINet (Fang et al., CVPR 2021, arXiv:2103.06495) supervises all three of its branches at
/// once. Paper Eq. 5:
/// </para>
/// <code>
/// L = lambda_v * L_v + lambda_l * L_l + L_f
/// </code>
/// <para>
/// where L_v is the vision model's character loss, L_l the language model's, and L_f the fused
/// output's. Each term is the same underlying character loss (cross-entropy with logits by
/// default), just applied to a different branch's prediction.
/// </para>
/// <para>
/// This matters for more than accuracy. The paper's AUTONOMOUS principle blocks gradient flow
/// from the language model back into the vision model, and that is only sound because L_v
/// supervises the vision model directly. With a single fused loss the barrier leaves the vision
/// encoder with no gradient source at all.
/// </para>
/// <para><b>For Beginners:</b> ABINet has three parts that each make their own guess at the
/// text: one that looks at the image, one that reasons about language, and one that combines
/// them. Rather than grading only the combined guess, this grades all three and adds the scores
/// up, so every part gets told how it did.</para>
/// </remarks>
public sealed class ABINetMultiTaskLoss<T> : LossFunctionBase<T>
{
    private readonly LossFunctionBase<T> _characterLoss;
    private readonly double _visionLossWeight;
    private readonly double _languageLossWeight;

    /// <summary>
    /// Creates the multi-task objective.
    /// </summary>
    /// <param name="characterLoss">
    /// The per-branch character loss. ABINet uses cross-entropy over the charset.
    /// </param>
    /// <param name="visionLossWeight">lambda_v in paper Eq. 5.</param>
    /// <param name="languageLossWeight">lambda_l in paper Eq. 5.</param>
    public ABINetMultiTaskLoss(
        LossFunctionBase<T> characterLoss,
        double visionLossWeight,
        double languageLossWeight)
    {
        _characterLoss = characterLoss ?? throw new ArgumentNullException(nameof(characterLoss));
        _visionLossWeight = visionLossWeight;
        _languageLossWeight = languageLossWeight;
    }

    /// <summary>Gets the per-branch character loss this objective sums over.</summary>
    public LossFunctionBase<T> CharacterLoss => _characterLoss;

    /// <inheritdoc/>
    /// <remarks>
    /// <paramref name="predicted"/> is the concatenation of the vision, language and fusion
    /// predictions along axis 0, and <paramref name="target"/> is the character target repeated
    /// three times to match. ABINet's training forward and <c>Train</c> produce that pairing.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        int predBlock = SplitLength(predicted, nameof(predicted));
        int targetBlock = SplitLength(target, nameof(target));

        var vision = Engine.TensorNarrow(predicted, dim: 0, start: 0, length: predBlock);
        var language = Engine.TensorNarrow(predicted, dim: 0, start: predBlock, length: predBlock);
        var fusion = Engine.TensorNarrow(predicted, dim: 0, start: 2 * predBlock, length: predBlock);

        var visionTarget = Engine.TensorNarrow(target, dim: 0, start: 0, length: targetBlock);
        var languageTarget = Engine.TensorNarrow(target, dim: 0, start: targetBlock, length: targetBlock);
        var fusionTarget = Engine.TensorNarrow(target, dim: 0, start: 2 * targetBlock, length: targetBlock);

        var lossV = _characterLoss.ComputeTapeLoss(vision, visionTarget);
        var lossL = _characterLoss.ComputeTapeLoss(language, languageTarget);
        var lossF = _characterLoss.ComputeTapeLoss(fusion, fusionTarget);

        return Engine.TensorAdd(
            Engine.TensorAdd(Scale(lossV, _visionLossWeight), Scale(lossL, _languageLossWeight)),
            lossF);
    }

    /// <summary>Multiplies a scalar loss tensor by a weight, staying on the tape.</summary>
    private Tensor<T> Scale(Tensor<T> loss, double weight)
    {
        if (weight == 1.0) return loss;

        var scalar = new Tensor<T>(loss.Shape.ToArray());
        var value = NumOps.FromDouble(weight);
        for (int i = 0; i < scalar.Length; i++) scalar[i] = value;
        return Engine.TensorMultiply(loss, scalar);
    }

    private static int SplitLength(Tensor<T> tensor, string name)
    {
        if (tensor.Rank < 1)
            throw new ArgumentException(
                $"ABINet's multi-task loss needs a tensor carrying all three branches, got rank {tensor.Rank}.",
                name);

        int rows = tensor.Shape[0];
        if (rows % 3 != 0)
            throw new ArgumentException(
                $"ABINet's multi-task loss expects {name} to stack the vision, language and fusion blocks along axis 0, so its leading dimension must be divisible by 3; got {rows}.",
                name);

        return rows / 3;
    }

    /// <inheritdoc/>
    /// <summary>
    /// The multi-task objective on the flat-vector surface:
    /// <c>lambda_v*L_v + lambda_l*L_l + L_f</c>.
    /// </summary>
    /// <remarks>
    /// THE SPLIT IS AVAILABLE HERE, contrary to what this remark used to claim. ComputeTapeLoss
    /// splits along axis 0, and axis-0 blocks are contiguous in the row-major buffer, so thirds of
    /// the flattened vector ARE those same three blocks. Forwarding the whole concatenated vector to
    /// the character loss instead computed a different objective from the tape path -- and every
    /// caller on this surface (optimizer evaluation, fitness scoring, reported training loss) got
    /// that different number with no error and no warning.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        var (visionP, languageP, fusionP) = SplitThirds(predicted, nameof(predicted));
        var (visionA, languageA, fusionA) = SplitThirds(actual, nameof(actual));

        T lossV = _characterLoss.CalculateLoss(visionP, visionA);
        T lossL = _characterLoss.CalculateLoss(languageP, languageA);
        T lossF = _characterLoss.CalculateLoss(fusionP, fusionA);

        return NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_visionLossWeight), lossV),
                NumOps.Multiply(NumOps.FromDouble(_languageLossWeight), lossL)),
            lossF);
    }


    /// <inheritdoc/>
    /// <remarks>
    /// Reuses <see cref="ComputeTapeLoss"/>'s split directly: this surface receives tensors, so the
    /// same axis-0 narrowing applies and there is no reason for it to compute a different objective
    /// from the tape path.
    /// </remarks>
    public override (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        int predBlock = SplitLength(predicted, nameof(predicted));
        int targetBlock = SplitLength(actual, nameof(actual));

        var (lossV, gradV) = _characterLoss.CalculateLossAndGradientGpu(
            Engine.TensorNarrow(predicted, dim: 0, start: 0, length: predBlock),
            Engine.TensorNarrow(actual, dim: 0, start: 0, length: targetBlock));
        var (lossL, gradL) = _characterLoss.CalculateLossAndGradientGpu(
            Engine.TensorNarrow(predicted, dim: 0, start: predBlock, length: predBlock),
            Engine.TensorNarrow(actual, dim: 0, start: targetBlock, length: targetBlock));
        var (lossF, gradF) = _characterLoss.CalculateLossAndGradientGpu(
            Engine.TensorNarrow(predicted, dim: 0, start: 2 * predBlock, length: predBlock),
            Engine.TensorNarrow(actual, dim: 0, start: 2 * targetBlock, length: targetBlock));

        T loss = NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_visionLossWeight), lossV),
                NumOps.Multiply(NumOps.FromDouble(_languageLossWeight), lossL)),
            lossF);

        var gradient = Engine.Concat(
            new[]
            {
                Engine.TensorMultiplyScalar(gradV, NumOps.FromDouble(_visionLossWeight)),
                Engine.TensorMultiplyScalar(gradL, NumOps.FromDouble(_languageLossWeight)),
                gradF,
            },
            0);

        return (loss, gradient);
    }

    /// <summary>
    /// Splits a flat vector into its vision, language and fusion thirds.
    /// </summary>
    /// <remarks>
    /// Axis-0 blocks are contiguous in the row-major buffer, so the tape path's narrowing and this
    /// slicing select the same elements. That equivalence is the whole reason this surface can
    /// compute the real objective.
    /// </remarks>
    private static (Vector<T> Vision, Vector<T> Language, Vector<T> Fusion) SplitThirds(
        Vector<T> value, string parameterName)
    {
        if (value.Length % 3 != 0)
        {
            throw new ArgumentException(
                $"ABINet concatenates three equal branches, so the length must be a multiple of 3; "
                + $"got {value.Length}.", parameterName);
        }

        int block = value.Length / 3;
        var vision = new Vector<T>(block);
        var language = new Vector<T>(block);
        var fusion = new Vector<T>(block);
        for (int i = 0; i < block; i++)
        {
            vision[i] = value[i];
            language[i] = value[block + i];
            fusion[i] = value[(2 * block) + i];
        }

        return (vision, language, fusion);
    }
}
