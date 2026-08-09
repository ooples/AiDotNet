using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Root Mean Squared Error (RMSE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type (float or double).</typeparam>
/// <remarks>
/// RMSE measures the square root of the average squared differences between predicted and actual values.
/// It is particularly useful for regression problems and gives more weight to larger errors.
///
/// Formula: RMSE = sqrt(mean((predicted - actual)^2))
///
/// The derivative with respect to predicted values is:
/// d(RMSE)/d(predicted) = (predicted - actual) / (n * RMSE)
/// where n is the number of samples and RMSE is the loss value.
///
/// This implementation leverages the existing StatisticsHelper.CalculateRootMeanSquaredError() method
/// for efficient and consistent calculation across the library.
/// </remarks>
[LossCategory(LossCategory.Regression)]
[LossTask(LossTask.Regression)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = true, ExpectedOutput = OutputType.Continuous)]
public class RootMeanSquaredErrorLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Root Mean Squared Error loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="actual">The actual (ground truth) values.</param>
    /// <returns>The RMSE loss value.</returns>
    /// <exception cref="ArgumentException">Thrown when predicted and actual vectors have different lengths.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        return StatisticsHelper<T>.CalculateRootMeanSquaredError(predicted, actual);
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
        var mse = Engine.ReduceMean(squared, allAxes, keepDims: false);

        // Offset before the square root. d(sqrt(x))/dx = 1/(2*sqrt(x)) is unbounded as x -> 0, so
        // a PERFECT prediction -- mse exactly 0 -- produced 0 * infinity = NaN and poisoned every
        // downstream parameter gradient. The offset is far below any meaningful loss, and it makes
        // the gradient at a perfect fit 0, which is the mathematically right answer there.
        return Engine.TensorSqrt(Engine.TensorAddScalar(mse, NumOps.FromDouble(1e-12)));
    }
}
