using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Losses;

/// <summary>
/// Binary Cross-Entropy loss for mask prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BCE loss is the standard loss for binary segmentation masks.
/// It computes the cross-entropy between predicted probabilities and binary ground truth.</para>
/// </remarks>
public class MaskBCELoss<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _eps;

    /// <summary>
    /// Creates a new mask BCE loss.
    /// </summary>
    /// <param name="eps">Small value for numerical stability.</param>
    public MaskBCELoss(double eps = 1e-7)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _eps = eps;
    }

    /// <summary>
    /// Computes BCE loss between predicted and target masks.
    /// </summary>
    /// <param name="predicted">Predicted mask probabilities [batch, height, width] or [batch, num_masks, height, width].</param>
    /// <param name="target">Target binary masks.</param>
    /// <returns>Scalar loss value.</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted.Length != target.Length)
            throw new ArgumentException("Predicted and target must have same shape");

        double loss = 0;

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = MathHelper.Clamp(_numOps.ToDouble(predicted[i]), _eps, 1 - _eps);
            double t = _numOps.ToDouble(target[i]);

            loss -= t * Math.Log(p) + (1 - t) * Math.Log(1 - p);
        }

        return _numOps.FromDouble(loss / predicted.Length);
    }

    /// <summary>
    /// Computes gradient of BCE loss.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> predicted, Tensor<T> target)
    {
        var gradient = new Tensor<T>(predicted.Shape);

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = MathHelper.Clamp(_numOps.ToDouble(predicted[i]), _eps, 1 - _eps);
            double t = _numOps.ToDouble(target[i]);

            // d/dp [-(t*log(p) + (1-t)*log(1-p))] = -t/p + (1-t)/(1-p)
            double grad = -t / p + (1 - t) / (1 - p);
            gradient[i] = _numOps.FromDouble(grad / predicted.Length);
        }

        return gradient;
    }
}

/// <summary>
/// Dice loss for mask prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Dice loss directly optimizes the Dice coefficient (F1 score),
/// making it better for imbalanced masks where foreground is much smaller than background.</para>
/// </remarks>
public class MaskDiceLoss<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _smooth;

    /// <summary>
    /// Creates a new mask Dice loss.
    /// </summary>
    /// <param name="smooth">Smoothing factor to avoid division by zero.</param>
    public MaskDiceLoss(double smooth = 1.0)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _smooth = smooth;
    }

    /// <summary>
    /// Computes Dice loss between predicted and target masks.
    /// </summary>
    /// <param name="predicted">Predicted mask probabilities.</param>
    /// <param name="target">Target binary masks.</param>
    /// <returns>Scalar loss value (1 - Dice coefficient).</returns>
    public T Compute(Tensor<T> predicted, Tensor<T> target)
    {
        double intersection = 0;
        double predSum = 0;
        double targetSum = 0;

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = _numOps.ToDouble(predicted[i]);
            double t = _numOps.ToDouble(target[i]);

            intersection += p * t;
            predSum += p * p;
            targetSum += t * t;
        }

        double dice = (2 * intersection + _smooth) / (predSum + targetSum + _smooth);

        return _numOps.FromDouble(1 - dice);
    }

    /// <summary>
    /// Computes gradient of Dice loss.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> predicted, Tensor<T> target)
    {
        double intersection = 0;
        double predSum = 0;
        double targetSum = 0;

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = _numOps.ToDouble(predicted[i]);
            double t = _numOps.ToDouble(target[i]);

            intersection += p * t;
            predSum += p * p;
            targetSum += t * t;
        }

        double numerator = 2 * intersection + _smooth;
        double denominator = predSum + targetSum + _smooth;

        var gradient = new Tensor<T>(predicted.Shape);

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = _numOps.ToDouble(predicted[i]);
            double t = _numOps.ToDouble(target[i]);

            // Quotient rule: d(Dice)/dp = (N' * D - N * D') / D^2
            // where N' = 2*t, D' = 2*p
            double grad = (2 * t * denominator - numerator * 2 * p) / (denominator * denominator);
            gradient[i] = _numOps.FromDouble(-grad); // Negative because loss = 1 - Dice
        }

        return gradient;
    }
}

/// <summary>
/// Focal loss for mask prediction (addresses class imbalance).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Focal loss down-weights easy examples (confident predictions)
/// and focuses training on hard examples. This helps with the severe class imbalance
/// in instance segmentation where background dominates.</para>
/// </remarks>
public class MaskFocalLoss<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _alpha;
    private readonly double _gamma;
    private readonly double _eps;

    /// <summary>
    /// Creates a new mask focal loss.
    /// </summary>
    /// <param name="alpha">Weighting factor for positive class (default 0.25).</param>
    /// <param name="gamma">Focusing parameter (default 2.0).</param>
    public MaskFocalLoss(double alpha = 0.25, double gamma = 2.0, double eps = 1e-7)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _alpha = alpha;
        _gamma = gamma;
        _eps = eps;
    }

    /// <summary>
    /// Computes focal loss between predicted and target masks.
    /// </summary>
    public T Compute(Tensor<T> predicted, Tensor<T> target)
    {
        double loss = 0;

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = MathHelper.Clamp(_numOps.ToDouble(predicted[i]), _eps, 1 - _eps);
            double t = _numOps.ToDouble(target[i]);

            // Focal loss: -alpha * (1-pt)^gamma * log(pt)
            double pt = t * p + (1 - t) * (1 - p);
            double alphaT = t * _alpha + (1 - t) * (1 - _alpha);

            loss -= alphaT * Math.Pow(1 - pt, _gamma) * Math.Log(pt);
        }

        return _numOps.FromDouble(loss / predicted.Length);
    }

    /// <summary>
    /// Computes gradient of focal loss.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> predicted, Tensor<T> target)
    {
        var gradient = new Tensor<T>(predicted.Shape);

        for (int i = 0; i < predicted.Length; i++)
        {
            double p = MathHelper.Clamp(_numOps.ToDouble(predicted[i]), _eps, 1 - _eps);
            double t = _numOps.ToDouble(target[i]);

            double pt = t * p + (1 - t) * (1 - p);
            double alphaT = t * _alpha + (1 - t) * (1 - _alpha);

            // Product rule: d/dp [(1-pt)^g * log(pt)]
            // = d[(1-pt)^g]/dp * log(pt) + (1-pt)^g * d[log(pt)]/dp
            // d[(1-pt)^g]/dp = g*(1-pt)^(g-1)*(-dpt/dp)  ← note negative sign
            // d[log(pt)]/dp = (1/pt)*dpt/dp
            double dpt_dp = 2 * t - 1;
            double term1 = -_gamma * Math.Pow(1 - pt, _gamma - 1) * Math.Log(pt) * dpt_dp;
            double term2 = Math.Pow(1 - pt, _gamma) * (1 / pt) * dpt_dp;

            double grad = -alphaT * (term1 + term2);
            gradient[i] = _numOps.FromDouble(grad / predicted.Length);
        }

        return gradient;
    }
}

/// <summary>
/// Combined mask loss using BCE + Dice.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Combining BCE and Dice loss often gives better results
/// than using either alone. BCE provides pixel-level supervision while Dice optimizes
/// the overall mask quality.</para>
/// </remarks>
public class CombinedMaskLoss<T>
{
    private readonly MaskBCELoss<T> _bceLoss;
    private readonly MaskDiceLoss<T> _diceLoss;
    private readonly double _bceWeight;
    private readonly double _diceWeight;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a combined BCE + Dice loss.
    /// </summary>
    /// <param name="bceWeight">Weight for BCE component.</param>
    /// <param name="diceWeight">Weight for Dice component.</param>
    public CombinedMaskLoss(double bceWeight = 1.0, double diceWeight = 1.0)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _bceLoss = new MaskBCELoss<T>();
        _diceLoss = new MaskDiceLoss<T>();
        _bceWeight = bceWeight;
        _diceWeight = diceWeight;
    }

    /// <summary>
    /// Computes combined loss.
    /// </summary>
    public T Compute(Tensor<T> predicted, Tensor<T> target)
    {
        double bce = _numOps.ToDouble(_bceLoss.Compute(predicted, target));
        double dice = _numOps.ToDouble(_diceLoss.Compute(predicted, target));

        return _numOps.FromDouble(_bceWeight * bce + _diceWeight * dice);
    }

    /// <summary>
    /// Computes gradient of combined loss.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> predicted, Tensor<T> target)
    {
        var bceGrad = _bceLoss.Backward(predicted, target);
        var diceGrad = _diceLoss.Backward(predicted, target);

        var gradient = new Tensor<T>(predicted.Shape);

        for (int i = 0; i < predicted.Length; i++)
        {
            double g = _bceWeight * _numOps.ToDouble(bceGrad[i]) +
                       _diceWeight * _numOps.ToDouble(diceGrad[i]);
            gradient[i] = _numOps.FromDouble(g);
        }

        return gradient;
    }
}
