
namespace AiDotNet.Regularization;

/// <summary>
/// Implements a no-op regularization class that applies no regularization penalty to the model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// NoRegularization serves as a baseline implementation that passes through all values unchanged.
/// This class is useful when a regularization object is required by the architecture but you
/// don't want to apply any actual regularization to your model.
/// </para>
/// <para><b>For Beginners:</b> This class essentially turns off regularization.
/// 
/// Think of regularization as optional guardrails that keep your model from becoming too complex:
/// - L1 and L2 regularization add these guardrails
/// - NoRegularization removes all guardrails
/// - Your model is free to find the exact fit to your training data
/// 
/// For example:
/// - When using NoRegularization, your model can assign very large coefficients to features
/// - Nothing prevents the model from perfectly memorizing your training data
/// - This can be good for very simple problems or when you have a lot of data relative to features
/// 
/// You might use NoRegularization when:
/// - You're confident your model won't overfit
/// - You want to see how your model performs without regularization
/// - You're using other techniques to prevent overfitting (like early stopping)
/// </para>
/// </remarks>
public class NoRegularization<T, TInput, TOutput> : RegularizationBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the NoRegularization class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a NoRegularization instance, which will not apply any regularization
    /// to the model during training or prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a "do nothing" regularization component.
    /// 
    /// When you initialize NoRegularization:
    /// - You're essentially telling the system "don't change anything"
    /// - No modification will be made to your input data or coefficients
    /// - Your model will train to fit the training data as closely as possible
    /// 
    /// It's like choosing not to use any filters on a photograph - you get
    /// the raw, unmodified version.
    /// </para>
    /// </remarks>
    public NoRegularization()
    {
    }

    /// <summary>
    /// Returns the input data unchanged, applying no regularization.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <returns>The same input data, unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the input matrix unchanged, representing no regularization penalty.
    /// This is consistent with the Vector version which also returns the input unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns your data exactly as you gave it.
    ///
    /// When called:
    /// - Your input matrix goes in
    /// - The exact same matrix comes out
    /// - No modification, shrinkage, or penalty is applied
    ///
    /// Think of it like a pass-through - nothing changes.
    /// </para>
    /// </remarks>
    public override Matrix<T> Regularize(Matrix<T> data)
    {
        // Return the original matrix unchanged - no regularization applied
        return data;
    }

    /// <summary>
    /// Returns the output data or coefficients unchanged, applying no regularization.
    /// </summary>
    /// <param name="data">The output data or coefficient vector.</param>
    /// <returns>The same output data or coefficient vector, unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method simply returns the output data or coefficient vector without any modification, 
    /// effectively applying no regularization to the model coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This method keeps your model's coefficients exactly as they are.
    /// 
    /// When called:
    /// - The coefficients from your model go in
    /// - The exact same coefficients come out
    /// - No shrinking, elimination, or adjustment of any kind
    /// 
    /// Unlike L1 or L2 regularization, which would reduce coefficient values,
    /// this method preserves your model's coefficients exactly as they were learned
    /// from the training data.
    /// </para>
    /// </remarks>
    public override Vector<T> Regularize(Vector<T> data)
    {
        return data;
    }

    /// <summary>
    /// Returns the gradient vector unchanged, applying no regularization.
    /// </summary>
    /// <param name="gradient">The gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The same gradient vector, unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method simply returns the gradient vector without any modification, effectively applying
    /// no regularization adjustment to the gradient during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets your model learn without any regularization influence.
    /// 
    /// During model training:
    /// - The gradient tells the model which direction to adjust coefficients to improve
    /// - Other regularization methods would modify this gradient to encourage simpler models
    /// - This method leaves the gradient exactly as is
    /// 
    /// Think of it like letting your model follow its natural learning path without any
    /// external guidance pushing it toward simplicity. It will focus entirely on fitting
    /// the training data as closely as possible.
    /// </para>
    /// </remarks>
    public override TOutput Regularize(TOutput gradient, TOutput coefficients)
    {
        return gradient;
    }
}
