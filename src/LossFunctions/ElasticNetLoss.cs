using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Elastic Net Loss function, which combines Mean Squared Error with L1 and L2 regularization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Elastic Net Loss combines the Mean Squared Error (which measures prediction accuracy) 
/// with two types of regularization (which prevent overfitting):
/// 
/// - L1 regularization (also called Lasso): Helps select only the most important features by pushing some weights to zero
/// - L2 regularization (also called Ridge): Prevents any single weight from becoming too large
/// 
/// The formula is: MSE + a * [l1Ratio * |weights|_1 + (1-l1Ratio) * 0.5 * |weights|_2²]
/// Where:
/// - MSE is the Mean Squared Error
/// - |weights|_1 is the L1 norm (sum of absolute values)
/// - |weights|_2² is the squared L2 norm (sum of squared values)
/// - a is the regularization strength
/// - l1Ratio controls the mix between L1 and L2 regularization
/// 
/// The l1Ratio parameter (between 0 and 1) controls the balance:
/// - When l1Ratio = 1: Only L1 regularization is used (Lasso)
/// - When l1Ratio = 0: Only L2 regularization is used (Ridge)
/// - Values in between: A mix of both (Elastic Net)
/// 
/// This loss function is particularly useful when:
/// - You have many correlated features
/// - You want to perform feature selection (L1 component)
/// - You still want the stability of L2 regularization
/// - You want to balance between model simplicity and prediction accuracy
/// </para>
/// </remarks>
public class ElasticNetLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// The mixing parameter between L1 and L2 regularization.
    /// </summary>
    private readonly T _l1Ratio;

    /// <summary>
    /// The regularization strength parameter.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the ElasticNetLoss class.
    /// </summary>
    /// <param name="l1Ratio">The mixing parameter between L1 and L2 regularization (0 to 1). Default is 0.5.</param>
    /// <param name="alpha">The regularization strength parameter. Default is 0.01.</param>
    public ElasticNetLoss(double l1Ratio = 0.5, double alpha = 0.01)
    {
        if (l1Ratio < 0 || l1Ratio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(l1Ratio), "L1 ratio must be between 0 and 1.");
        }

        _l1Ratio = NumOps.FromDouble(l1Ratio);
        _alpha = NumOps.FromDouble(alpha);
    }

    /// <summary>
    /// Calculates the Elastic Net Loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>The elastic net loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Calculate the Mean Squared Error component
        T mseLoss = StatisticsHelper<T>.CalculateMeanSquaredError(predicted, actual);

        // Calculate L1 and L2 regularization terms
        T l1Regularization = NumOps.Zero;
        T l2Regularization = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            l1Regularization = NumOps.Add(l1Regularization, NumOps.Abs(predicted[i]));
            l2Regularization = NumOps.Add(l2Regularization, NumOps.Power(predicted[i], NumOps.FromDouble(2)));
        }

        // Scale the regularization terms
        T l1Term = NumOps.Multiply(
            NumOps.Multiply(_alpha, _l1Ratio),
            l1Regularization
        );

        T l2Term = NumOps.Multiply(
            NumOps.Multiply(
                NumOps.Multiply(_alpha, NumOps.Subtract(NumOps.One, _l1Ratio)),
                NumOps.FromDouble(0.5)
            ),
            l2Regularization
        );

        // Combine all terms
        return NumOps.Add(NumOps.Add(mseLoss, l1Term), l2Term);
    }

    /// <summary>
    /// Calculates the derivative of the Elastic Net Loss function.
    /// </summary>
    /// <param name="predicted">The predicted values vector.</param>
    /// <param name="actual">The actual (ground truth) values vector.</param>
    /// <returns>A vector containing the derivatives of the elastic net loss with respect to each predicted value.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            // MSE gradient component: 2*(predicted - actual)/n
            T mseGradient = NumOps.Multiply(
                NumOps.FromDouble(2),
                NumOps.Divide(
                    NumOps.Subtract(predicted[i], actual[i]),
                    NumOps.FromDouble(predicted.Length)
                )
            );

            // L1 gradient component: a * l1Ratio * sign(predicted)
            T l1Gradient = NumOps.Multiply(
                NumOps.Multiply(_alpha, _l1Ratio),
                SignOf(predicted[i])
            );

            // L2 gradient component: a * (1-l1Ratio) * predicted
            T l2Gradient = NumOps.Multiply(
                NumOps.Multiply(_alpha, NumOps.Subtract(NumOps.One, _l1Ratio)),
                predicted[i]
            );

            // Combine all gradient components
            derivative[i] = NumOps.Add(NumOps.Add(mseGradient, l1Gradient), l2Gradient);
        }

        return derivative;
    }

    /// <summary>
    /// Returns the sign of a value: -1 for negative, 1 for positive, 0 for zero.
    /// </summary>
    /// <param name="value">The value to determine the sign of.</param>
    /// <returns>The sign of the value.</returns>
    private T SignOf(T value)
    {
        if (NumOps.GreaterThan(value, NumOps.Zero))
        {
            return NumOps.One;
        }
        else if (NumOps.LessThan(value, NumOps.Zero))
        {
            return NumOps.Negate(NumOps.One);
        }
        else
        {
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Calculates both Elastic Net loss and gradient on GPU in a single efficient pass.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    public override (T Loss, IGpuTensor<T> Gradient) CalculateLossAndGradientGpu(IGpuTensor<T> predicted, IGpuTensor<T> actual)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        var backend = engine?.GetBackend();

        if (backend == null)
        {
            return base.CalculateLossAndGradientGpu(predicted, actual);
        }

        int size = predicted.ElementCount;
        float l1Ratio = Convert.ToSingle(NumOps.ToDouble(_l1Ratio));
        float alpha = Convert.ToSingle(NumOps.ToDouble(_alpha));
        // Calculate l1Weight and l2Weight from l1Ratio and alpha
        float l1Weight = alpha * l1Ratio;
        float l2Weight = alpha * (1 - l1Ratio);

        // Compute loss on GPU
        float lossValue = backend.ElasticNetLoss(predicted.Buffer, actual.Buffer, size, l1Weight, l2Weight);

        // Allocate gradient buffer and compute gradient on GPU
        var gradientBuffer = backend.AllocateBuffer(size);
        backend.ElasticNetBackward(predicted.Buffer, actual.Buffer, gradientBuffer, size, l1Weight, l2Weight);

        // Create gradient tensor
        var gradientTensor = new GpuTensor<T>(backend, gradientBuffer, predicted.Shape, GpuTensorRole.Gradient);

        return (NumOps.FromDouble(lossValue), gradientTensor);
    }
}
