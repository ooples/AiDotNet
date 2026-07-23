using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Mean-squared error measured in <i>probability</i> space for models whose final
/// layer emits quantum <i>amplitudes</i>: <c>loss = mean((predicted² − target)²)</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// A quantum neural network's layer chain produces amplitudes; the observable
/// (Born's rule) is the squared magnitude, <c>p = amplitude²</c>. Training such a
/// model on probability targets with a plain MSE on the amplitudes optimises the
/// wrong objective — minimising <c>‖amplitude − √target‖²</c> does NOT monotonically
/// minimise the measured <c>‖amplitude² − target‖²</c> because the square is
/// non-linear. This loss folds the Born-rule square into the objective so the tape
/// trains exactly the measured quantity the model reports at inference, keeping the
/// training loss and the measured (probability-space) error in lock-step.
/// </para>
/// <para>
/// Gradient w.r.t. the amplitude prediction <c>a</c>:
/// <c>d/da mean((a² − t)²) = (4/n) · a · (a² − t)</c>.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Regression)]
[LossTask(LossTask.Regression)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, IsSymmetric = false, HasStandardGradientSign = false, ExpectedOutput = OutputType.Continuous)]
public class BornRuleMseLoss<T> : LossFunctionBase<T>
{
    /// <inheritdoc />
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            // measured probability = amplitude²
            T p = NumOps.Multiply(predicted[i], predicted[i]);
            T diff = NumOps.Subtract(p, actual[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <inheritdoc />
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var derivative = new Vector<T>(predicted.Length);
        T fourOverN = NumOps.FromDouble(4.0 / predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T a = predicted[i];
            T p = NumOps.Multiply(a, a);
            T diff = NumOps.Subtract(p, actual[i]);
            // d/da mean((a² − t)²) = (4/n) · a · (a² − t)
            derivative[i] = NumOps.Multiply(fourOverN, NumOps.Multiply(a, diff));
        }

        return derivative;
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // measured = predicted²  (Born's rule), then MSE(measured, target).
        var measured = Engine.TensorMultiply(predicted, predicted);
        var diff = Engine.TensorSubtract(measured, target);
        var squared = Engine.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
        return Engine.ReduceMean(squared, allAxes, keepDims: false);
    }
}
