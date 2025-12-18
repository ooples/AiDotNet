namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for applying regularization techniques to machine learning models.
/// </summary>
/// <remarks>
/// Regularization helps prevent overfitting by adding constraints to the model,
/// typically by penalizing large coefficient values.
/// 
/// <b>For Beginners:</b> Regularization is like adding training wheels to your AI model.
/// 
/// Imagine you're teaching a child to recognize dogs. If you only show them pictures of
/// German Shepherds, they might think only German Shepherds are dogs (this is "overfitting").
/// Regularization helps your model focus on the most important patterns and ignore "noise"
/// in the training data, making it better at generalizing to new examples.
/// 
/// Common regularization methods include:
/// - L1 (Lasso): Can zero out less important features completely
/// - L2 (Ridge): Keeps all features but makes their impact smaller
/// - Elastic Net: A combination of L1 and L2
/// 
/// Without regularization, models often perform well on training data but poorly on new data.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
public interface IRegularization<T, TInput, TOutput>
{
    /// <summary>
    /// Applies regularization to a matrix of input features.
    /// </summary>
    /// <remarks>
    /// This method applies regularization to a matrix of input features.
    /// 
    /// <b>For Beginners:</b> This method helps prevent overfitting by modifying
    /// your input data according to the regularization strategy. It can reduce
    /// the influence of extreme values or unnecessary features.
    /// 
    /// This is typically used when preparing data for model training.
    /// </remarks>
    /// <param name="data">The input matrix to regularize.</param>
    /// <returns>The regularized matrix.</returns>
    Matrix<T> Regularize(Matrix<T> data);

    /// <summary>
    /// Applies regularization to output data or model coefficients.
    /// </summary>
    /// <remarks>
    /// This method applies regularization to output data or model coefficients.
    /// 
    /// <b>For Beginners:</b> Model coefficients are the "importance weights" your model assigns to different features.
    /// This method helps keep these weights from getting too large, which can cause overfitting.
    /// 
    /// For example:
    /// - With L1 regularization, some coefficients might be reduced to exactly zero
    /// - With L2 regularization, all coefficients are typically reduced but remain non-zero
    /// 
    /// Think of it as encouraging your model to use simpler explanations by penalizing complex solutions.
    /// </remarks>
    /// <param name="data">The output data or coefficients to regularize.</param>
    /// <returns>The regularized data or coefficients.</returns>
    Vector<T> Regularize(Vector<T> data);

    /// <summary>
    /// Applies regularization to the gradient during optimization.
    /// </summary>
    /// <remarks>
    /// This method modifies the gradient used in optimization algorithms to account for regularization.
    /// 
    /// <b>For Beginners:</b> When a model is learning (training), it follows a "gradient" that tells it
    /// which direction to adjust its parameters to improve. This method modifies that gradient
    /// to include the regularization constraints.
    /// 
    /// It's like adding a gentle force that pulls the model toward simpler solutions during training.
    /// This helps prevent the model from becoming too complex and overfitting the training data.
    /// </remarks>
    /// <param name="gradient">The original gradient vector from the optimization algorithm.</param>
    /// <param name="coefficients">The current coefficient values.</param>
    /// <returns>The regularized gradient vector.</returns>
    TOutput Regularize(TOutput gradient, TOutput coefficients);

    /// <summary>
    /// Retrieves the current regularization options.
    /// </summary>
    /// <remarks>
    /// This method returns the configuration settings for the regularization technique.
    /// 
    /// <b>For Beginners:</b> This method lets you check what regularization settings are being used.
    /// The options typically include:
    /// 
    /// - The type of regularization (L1, L2, Elastic Net, etc.)
    /// - The strength of regularization (how much to penalize complex models)
    /// - Other specific settings for the chosen regularization method
    /// </remarks>
    /// <returns>The regularization options currently in use.</returns>
    RegularizationOptions GetOptions();
}
