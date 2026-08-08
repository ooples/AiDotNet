using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// A weighted sum of several loss functions: <c>L = sum_i w_i * L_i</c>.
/// </summary>
/// <remarks>
/// <para>
/// Many published training objectives are explicitly a linear combination of simpler terms, and
/// implementing only one of those terms is a silent departure from the paper. Examples in this
/// library:
/// </para>
/// <list type="bullet">
/// <item><description>SAM (Kirillov et al. 2023, §3) supervises masks with focal + dice loss in a
/// <b>20:1</b> ratio.</description></item>
/// <item><description>Paraformer / SeACo-Paraformer (arXiv 2206.08317 / 2308.03266) combine
/// cross-entropy with CTC and the predictor's MAE.</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> a loss function scores how wrong a prediction is. Some models are trained
/// against several such scores at once, each contributing a different amount — for instance "mostly
/// care about focal loss, and a little about dice loss". This class expresses exactly that: give it
/// the individual losses and how much each one counts.
/// </para>
/// <para>
/// Weights are used as given and are NOT normalised, because published ratios (such as SAM's 20:1)
/// are stated as absolute coefficients. Normalising would silently rescale the objective and change
/// the effective learning rate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LossCategory(LossCategory.Segmentation)]
[LossTask(LossTask.SemanticSegmentation)]
[LossTask(LossTask.InstanceSegmentation)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, HandlesImbalancedData = true, RequiresProbabilityInputs = true, TestInputFormat = LossTestInputFormat.SegmentationMask, ExpectedOutput = OutputType.Probabilities)]
public class CompositeLoss<T> : LossFunctionBase<T>
{
    private readonly LossFunctionBase<T>[] _losses;
    private readonly T[] _weights;

    /// <summary>
    /// Creates a composite loss from (loss, weight) pairs.
    /// </summary>
    /// <param name="terms">The loss terms and their absolute coefficients. Terms are <see cref="LossFunctionBase{T}"/> rather than <see cref="ILossFunction{T}"/> because the composite must forward <c>ComputeTapeLoss</c>, which the interface does not declare.</param>
    /// <exception cref="ArgumentNullException">If <paramref name="terms"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// If no terms are supplied, or any individual loss is null.
    /// </exception>
    public CompositeLoss(params (LossFunctionBase<T> Loss, double Weight)[] terms)
    {
        if (terms is null || terms.Length == 0)
        {
            // Default to SAM's published mask objective (Kirillov et al. 2023, §3): focal + dice in a
            // 20:1 ratio, with focal at the paper's gamma=2 / alpha=0.25. This keeps a parameterless
            // construction meaningful and consistent with the Segmentation category declared above,
            // rather than throwing for a caller that just wants the standard segmentation composite.
            terms = new (LossFunctionBase<T> Loss, double Weight)[]
            {
                (new FocalLoss<T>(gamma: 2.0, alpha: 0.25), 20.0),
                (new DiceLoss<T>(), 1.0),
            };
        }

        _losses = new LossFunctionBase<T>[terms.Length];
        _weights = new T[terms.Length];
        for (int i = 0; i < terms.Length; i++)
        {
            if (terms[i].Loss is null)
                throw new ArgumentException($"Loss term {i} is null.", nameof(terms));
            _losses[i] = terms[i].Loss;
            _weights[i] = NumOps.FromDouble(terms[i].Weight);
        }
    }

    /// <summary>Gets the number of terms in this composite objective.</summary>
    public int TermCount => _losses.Length;

    /// <inheritdoc/>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        T total = NumOps.Zero;
        for (int i = 0; i < _losses.Length; i++)
            total = NumOps.Add(total, NumOps.Multiply(_weights[i], _losses[i].CalculateLoss(predicted, actual)));
        return total;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The derivative of a weighted sum is the weighted sum of the derivatives, so each term's
    /// gradient is scaled by its own coefficient and accumulated.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T>? accumulated = null;
        for (int i = 0; i < _losses.Length; i++)
        {
            var term = _losses[i].CalculateDerivative(predicted, actual);
            if (accumulated is null)
            {
                accumulated = new Vector<T>(term.Length);
                for (int j = 0; j < term.Length; j++)
                    accumulated[j] = NumOps.Multiply(_weights[i], term[j]);
                continue;
            }

            if (term.Length != accumulated.Length)
                throw new InvalidOperationException(
                    $"Composite loss term {i} produced a gradient of length {term.Length}, " +
                    $"but term 0 produced {accumulated.Length}. All terms must score the same prediction.");

            for (int j = 0; j < term.Length; j++)
                accumulated[j] = NumOps.Add(accumulated[j], NumOps.Multiply(_weights[i], term[j]));
        }

        return accumulated!;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Combining through <c>Engine</c> keeps every term on the autodiff tape, so the composite
    /// objective is differentiated as one graph rather than losing the terms' gradients.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        Tensor<T>? accumulated = null;
        for (int i = 0; i < _losses.Length; i++)
        {
            var term = _losses[i].ComputeTapeLoss(predicted, target);

            // NO SHAPE NORMALIZATION HERE ANY MORE. This used to reshape every term to rank-1 [1]
            // because implementations disagreed -- the focal([1]) + dice([]) combination on SAM2
            // threw "Tensor shapes must match. Got [1] and []", aborting the step and surfacing as
            // the misleading "no parameters changed after training". ComputeTapeLoss now documents
            // rank-0 [] as its contract and every implementation honours it, so absorbing a wrong
            // shape here would hide a real contract violation instead of reporting it. The
            // sameShape check below is the loud failure that replaces it.
            var scaled = Engine.TensorMultiplyScalar(term, _weights[i]);

            if (accumulated is null)
            {
                accumulated = scaled;
                continue;
            }

            bool sameShape = accumulated.Shape.Length == scaled.Shape.Length;
            for (int d = 0; sameShape && d < scaled.Shape.Length; d++)
            {
                sameShape = accumulated.Shape[d] == scaled.Shape[d];
            }

            if (!sameShape)
            {
                throw new InvalidOperationException(
                    $"Composite loss term {i} produced shape [{string.Join(",", scaled.Shape)}] but the " +
                    $"running total has shape [{string.Join(",", accumulated.Shape)}]. All terms must " +
                    "reduce the same prediction to a comparable scalar.");
            }

            accumulated = Engine.TensorAdd(accumulated, scaled);
        }

        return accumulated!;
    }
}
