namespace AiDotNet.Regularization;

/// <summary>
/// Implements Elastic Net regularization, a hybrid approach that combines L1 (Lasso) and L2 (Ridge) regularization techniques.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// Elastic Net regularization provides the benefits of both L1 and L2 regularization methods. It helps prevent overfitting
/// by penalizing large coefficient values, while also encouraging sparsity (more coefficients set to zero) when appropriate.
/// The L1Ratio parameter controls the balance between L1 and L2 regularization.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is like having two different tools to prevent your model from becoming too complex.
/// 
/// Think of it like this:
/// - L1 (Lasso) regularization tends to completely eliminate less important features (setting them to zero)
/// - L2 (Ridge) regularization keeps all features but makes them smaller overall
/// - Elastic Net lets you blend these two approaches for the best of both worlds
/// </para>
/// </remarks>
public class ElasticNetRegularization<T, TInput, TOutput> : RegularizationBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the ElasticNetRegularization class with optional custom options.
    /// </summary>
    /// <param name="options">
    /// Configuration options for regularization including strength and L1 ratio. If not provided,
    /// default values will be used (strength = 0.1, L1 ratio = 0.5).
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an Elastic Net regularization instance with the specified options or default values.
    /// The L1 ratio determines the balance between L1 and L2 regularization, with values ranging from 0 (pure L2) to 1 (pure L1).
    /// </para>
    /// </remarks>
    public ElasticNetRegularization(RegularizationOptions? options = null) : base(options ?? new RegularizationOptions
    {
        Type = RegularizationType.ElasticNet,
        Strength = 0.1, // Default Elastic Net regularization strength
        L1Ratio = 0.5  // Default balance between L1 and L2
    })
    {
    }

    /// <summary>
    /// Adjusts the gradient vector to account for Elastic Net regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method modifies the gradient vector during optimization to account for the Elastic Net regularization
    /// penalty. It adds the derivative of the Elastic Net penalty term with respect to the coefficients to the
    /// original gradient, steering the optimization toward solutions with appropriate regularization.
    /// </para>
    /// </remarks>
    public override TOutput Regularize(TOutput gradient, TOutput coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);

        if (gradient is Vector<T> gradientVector && coefficients is Vector<T> coefficientVector)
        {
            var result = gradientVector.Add(coefficientVector.Transform(c =>
            {
                var l1Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l1Ratio, NumOps.SignOrZero(c)));
                var l2Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l2Ratio, c));
                return NumOps.Add(l1Part, l2Part);
            }));

            return (TOutput)(object)result;
        }
        else if (gradient is Tensor<T> gradientTensor && coefficients is Tensor<T> coefficientTensor)
        {
            // Convert tensors to vectors for calculation
            var gradientFlattenedVector = gradientTensor.ToVector();
            var coefficientFlattenedVector = coefficientTensor.ToVector();

            // Apply regularization to gradient
            var result = gradientFlattenedVector.Add(coefficientFlattenedVector.Transform(c =>
            {
                var l1Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l1Ratio, NumOps.SignOrZero(c)));
                var l2Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l2Ratio, c));
                return NumOps.Add(l1Part, l2Part);
            }));

            // Convert back to tensor with the same shape as the gradient tensor
            var resultTensor = Tensor<T>.FromVector(result);
            if (gradientTensor.Shape.Length > 1)
            {
                resultTensor = resultTensor.Reshape(gradientTensor.Shape);
            }

            return (TOutput)(object)resultTensor;
        }

        throw new InvalidOperationException(
            $"Unsupported output types {typeof(TOutput).Name} for Elastic Net regularization gradient. " +
            $"Supported types are Vector<{typeof(T).Name}> and Tensor<{typeof(T).Name}>.");
    }

    /// <summary>
    /// Applies Elastic Net regularization to a matrix.
    /// </summary>
    /// <param name="data">The matrix data to regularize.</param>
    /// <returns>The regularized matrix data.</returns>
    /// <remarks>
    /// <para>
    /// This method implements Elastic Net regularization for matrix data by applying a combination of L1 and L2 penalties.
    /// The L1 component promotes sparsity (more zeros), while the L2 component shrinks all values.
    /// The L1Ratio parameter controls the balance between these two effects.
    /// </para>
    /// </remarks>
    public override Matrix<T> Regularize(Matrix<T> data)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);
        var result = new Matrix<T>(data.Rows, data.Columns);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                var value = data[i, j];
                var subPart = NumOps.Subtract(NumOps.Abs(value), NumOps.Multiply(regularizationStrength, l1Ratio));

                var l1Part = NumOps.Multiply(
                    NumOps.SignOrZero(value),
                    NumOps.GreaterThan(subPart, NumOps.Zero) ? subPart : NumOps.Zero
                );

                var l2Part = NumOps.Multiply(
                    value,
                    NumOps.Subtract(NumOps.One, NumOps.Multiply(regularizationStrength, l2Ratio))
                );

                result[i, j] = NumOps.Add(l1Part, l2Part);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies Elastic Net regularization to a vector.
    /// </summary>
    /// <param name="data">The vector data to regularize.</param>
    /// <returns>The regularized vector data.</returns>
    /// <remarks>
    /// <para>
    /// This method implements Elastic Net regularization for vector data by applying a combination of L1 and L2 penalties.
    /// The L1 component promotes sparsity by potentially setting some elements to zero, while the L2 component 
    /// shrinks all values proportionally. The L1Ratio parameter (between 0 and 1) controls the balance between these effects.
    /// </para>
    /// </remarks>
    public override Vector<T> Regularize(Vector<T> data)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);
        var result = new Vector<T>(data.Length);

        for (int i = 0; i < data.Length; i++)
        {
            var value = data[i];
            var subPart = NumOps.Subtract(NumOps.Abs(value), NumOps.Multiply(regularizationStrength, l1Ratio));

            var l1Part = NumOps.Multiply(
                NumOps.SignOrZero(value),
                NumOps.GreaterThan(subPart, NumOps.Zero) ? subPart : NumOps.Zero
            );

            var l2Part = NumOps.Multiply(
                value,
                NumOps.Subtract(NumOps.One, NumOps.Multiply(regularizationStrength, l2Ratio))
            );

            result[i] = NumOps.Add(l1Part, l2Part);
        }

        return result;
    }
}
