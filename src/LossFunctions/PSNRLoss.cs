using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// The negative-PSNR objective BasicSR trains restoration networks with: the weighted mean over samples of
/// <c>10/ln 10 · log(MSE + 1e-8)</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// BasicSR's <c>PSNRLoss</c> (basicsr/models/losses/losses.py) takes the MSE of each sample over its remaining axes,
/// adds 1e-8, and averages <c>loss_weight · 10/ln 10 · log(·)</c> over the batch. Minimizing it maximizes each
/// sample's PSNR up to a constant, and because each sample's gradient is divided by its own MSE, samples that are
/// already clean are not drowned out by noisy ones. A tensor of rank 1 is one sample.
/// </para>
/// <para>
/// <b>For Beginners:</b> PSNR (peak signal-to-noise ratio) is the standard quality score for restored images, in
/// decibels. Training directly on (minus) PSNR instead of on the plain squared error makes every image's
/// improvement count in decibels, so small errors on easy images still matter.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Regression)]
[LossCategory(LossCategory.Reconstruction)]
[LossTask(LossTask.Denoising)]
[LossTask(LossTask.SuperResolution)]
[LossProperty(IsNonNegative = false, ZeroForIdentical = false, ExpectedOutput = OutputType.Continuous)]
public class PSNRLoss<T> : LossFunctionBase<T>
{
    private const double Epsilon = 1e-8;
    private static readonly double Scale = 10.0 / Math.Log(10.0);
    private readonly double _weight;

    /// <summary>
    /// Initializes a new instance of the PSNRLoss class.
    /// </summary>
    /// <param name="lossWeight">BasicSR's <c>loss_weight</c>; Shift-Net and NAFNet train with 0.5. Default: 1.</param>
    public PSNRLoss(double lossWeight = 1.0)
    {
        if (lossWeight <= 0 || double.IsNaN(lossWeight) || double.IsInfinity(lossWeight))
            throw new ArgumentOutOfRangeException(nameof(lossWeight), "The loss weight must be positive and finite.");
        _weight = lossWeight;
    }

    /// <inheritdoc />
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        double sum = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = NumOps.ToDouble(predicted[i]) - NumOps.ToDouble(actual[i]);
            sum += diff * diff;
        }
        double mse = predicted.Length == 0 ? 0 : sum / predicted.Length;
        return NumOps.FromDouble(_weight * Scale * Math.Log(mse + Epsilon));
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(diff, diff);
        // One sample per leading index; the MSE runs over the remaining axes.
        var sampleAxes = squared.Rank >= 2
            ? Enumerable.Range(1, squared.Rank - 1).ToArray()
            : Enumerable.Range(0, squared.Rank).ToArray();
        var mse = Engine.ReduceMean(squared, sampleAxes, keepDims: false);
        var logs = Engine.TensorLog(Engine.TensorAddScalar(mse, NumOps.FromDouble(Epsilon)));
        var mean = logs.Rank == 0 ? logs : Engine.ReduceMean(logs, Enumerable.Range(0, logs.Rank).ToArray(), keepDims: false);
        return Engine.TensorMultiplyScalar(mean, NumOps.FromDouble(_weight * Scale));
    }
}
