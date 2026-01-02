namespace AiDotNet.LossFunctions;

/// <summary>
/// Scale-invariant depth loss function for depth estimation training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This loss function is specifically designed for training depth estimation models.
/// It handles the inherent scale ambiguity in monocular depth estimation by focusing on the
/// relative depth relationships between pixels rather than absolute depth values.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// The loss is computed as: (1/n) * Σ(d²) - (λ/n²) * (Σd)²
/// where d = log(pred) - log(actual), and λ controls the scale-invariance penalty.
/// </para>
/// <para>
/// <b>Reference:</b> Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
/// </para>
/// </remarks>
public class ScaleInvariantDepthLoss<T> : LossFunctionBase<T>
{
    private readonly double _lambda;

    /// <summary>
    /// Initializes a new instance of the ScaleInvariantDepthLoss class.
    /// </summary>
    /// <param name="lambda">The scale-invariance weight (default: 0.5). Higher values increase scale invariance.</param>
    public ScaleInvariantDepthLoss(double lambda = 0.5)
    {
        _lambda = lambda;
    }

    /// <inheritdoc/>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        int n = predicted.Length;
        T sum = NumOps.Zero;
        T sumDiff = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            // Log difference
            double predLog = Math.Log(Convert.ToDouble(predicted[i]) + 1e-8);
            double actLog = Math.Log(Convert.ToDouble(actual[i]) + 1e-8);
            double diff = predLog - actLog;

            sum = NumOps.Add(sum, NumOps.FromDouble(diff * diff));
            sumDiff = NumOps.Add(sumDiff, NumOps.FromDouble(diff));
        }

        // Scale-invariant loss: (1/n) * sum(d^2) - (lambda/n^2) * sum(d)^2
        double nDouble = n;
        double sumDouble = Convert.ToDouble(sum);
        double sumDiffDouble = Convert.ToDouble(sumDiff);

        double loss = (1.0 / nDouble) * sumDouble - (_lambda / (nDouble * nDouble)) * (sumDiffDouble * sumDiffDouble);

        return NumOps.FromDouble(loss);
    }

    /// <inheritdoc/>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        int n = predicted.Length;
        var derivative = new Vector<T>(n);

        double sumDiff = 0;
        for (int i = 0; i < n; i++)
        {
            double predLog = Math.Log(Convert.ToDouble(predicted[i]) + 1e-8);
            double actLog = Math.Log(Convert.ToDouble(actual[i]) + 1e-8);
            sumDiff += predLog - actLog;
        }

        for (int i = 0; i < n; i++)
        {
            double pred = Convert.ToDouble(predicted[i]);
            double predLog = Math.Log(pred + 1e-8);
            double actLog = Math.Log(Convert.ToDouble(actual[i]) + 1e-8);
            double diff = predLog - actLog;

            // Derivative: (2/n) * d / pred - (2*lambda/n^2) * sum(d) / pred
            double grad = (2.0 / n) * diff / (pred + 1e-8) - (2.0 * _lambda / (n * n)) * sumDiff / (pred + 1e-8);
            derivative[i] = NumOps.FromDouble(grad);
        }

        return derivative;
    }
}
