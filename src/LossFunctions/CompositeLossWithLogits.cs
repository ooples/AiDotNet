using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Applies a sigmoid to raw logits and then evaluates a weighted composite probability loss.
/// </summary>
/// <remarks>
/// This is the logits-domain counterpart to <see cref="CompositeLoss{T}"/>. Keeping the activation
/// at the loss boundary prevents models from applying sigmoid in their output head (which would
/// hide logits from numerically stable objectives) and prevents probability-domain terms such as
/// focal and Dice loss from receiving unbounded values. The sigmoid is part of the autodiff graph,
/// so gradients propagate through the transform on every engine backend.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LossCategory(LossCategory.Segmentation)]
[LossTask(LossTask.SemanticSegmentation)]
[LossTask(LossTask.InstanceSegmentation)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, HandlesImbalancedData = true, RequiresProbabilityInputs = false, TestInputFormat = LossTestInputFormat.RawLogits, ExpectedOutput = OutputType.Logits)]
public sealed class CompositeLossWithLogits<T> : LossFunctionBase<T>
{
    private readonly CompositeLoss<T> _probabilityLoss;

    /// <summary>
    /// Creates a logits-aware composite from probability-domain loss terms and their absolute
    /// coefficients. An empty term list selects <see cref="CompositeLoss{T}"/>'s standard 20:1
    /// focal-plus-Dice segmentation objective.
    /// </summary>
    public CompositeLossWithLogits(params (LossFunctionBase<T> Loss, double Weight)[] terms)
    {
        _probabilityLoss = new CompositeLoss<T>(terms);
    }

    /// <summary>Gets the number of terms in the wrapped composite objective.</summary>
    public int TermCount => _probabilityLoss.TermCount;

    /// <inheritdoc/>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        var probabilities = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            double logit = NumOps.ToDouble(predicted[i]);
            double probability;
            if (logit >= 0.0)
            {
                double expNeg = Math.Exp(-logit);
                probability = 1.0 / (1.0 + expNeg);
            }
            else
            {
                double expPos = Math.Exp(logit);
                probability = expPos / (1.0 + expPos);
            }

            probabilities[i] = NumOps.FromDouble(probability);
        }

        return _probabilityLoss.CalculateLoss(probabilities, actual);
    }

    /// <inheritdoc/>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
        => _probabilityLoss.ComputeTapeLoss(Engine.TensorSigmoid(predicted), target);
}
