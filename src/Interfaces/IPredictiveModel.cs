namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality of a trained predictive model that can make predictions on new data.
/// </summary>
/// <remarks>
/// This interface represents a machine learning model that has been trained and is ready to use.
/// 
/// <b>For Beginners:</b> Think of a predictive model like a calculator that has been specially programmed
/// to solve a specific type of problem. After you've "trained" it with examples (like showing it
/// houses and their prices), it can make educated guesses about new examples (predicting prices
/// for houses it hasn't seen before).
/// 
/// This interface provides the methods you need to:
/// - Make predictions with your trained model
/// - Get information about how the model was created and how well it performs
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("PredictiveModel")]
public interface IPredictiveModel<T, TInput, TOutput> : IModelSerializer
{
    /// <summary>
    /// Makes predictions using the trained model on new input data.
    /// </summary>
    /// <remarks>
    /// This is the primary function of a predictive model - taking new data and generating predictions.
    /// 
    /// <b>For Beginners:</b> This is where the magic happens! After your model has learned patterns from
    /// training data, this method lets you feed it new information and get predictions back.
    /// 
    /// For example:
    /// - If you trained a model to predict house prices, you would provide details about new houses
    ///   and get back predicted prices.
    /// - If you trained a model to identify spam emails, you would provide new emails and get back
    ///   predictions about whether they're spam or not.
    /// 
    /// The input is organized as a matrix where:
    /// - Each row represents one example (one house, one email, etc.)
    /// - Each column represents one feature or characteristic (square footage, number of bedrooms,
    ///   or for emails: number of exclamation marks, certain keywords, etc.)
    /// </remarks>
    /// <param name="input">A matrix containing the new data points to make predictions for.
    /// Each row is a separate data point, and each column is a feature.</param>
    /// <returns>A vector containing the predicted values, one for each input row.</returns>
    TOutput Predict(TInput input);

    /// <summary>
    /// Retrieves metadata and performance information about the trained model.
    /// </summary>
    /// <remarks>
    /// This method provides access to information about how the model was created and how well it performs.
    /// 
    /// <b>For Beginners:</b> This method gives you a "report card" for your model. It tells you:
    /// - How accurate the model is
    /// - What settings were used to create it
    /// - When it was trained
    /// - What features (input variables) it uses
    /// - Other technical details that help you understand its strengths and limitations
    /// 
    /// This information is useful for:
    /// - Comparing different models to choose the best one
    /// - Documenting how your model was created
    /// - Understanding where your model might make mistakes
    /// - Deciding if your model needs to be improved or retrained
    /// </remarks>
    /// <returns>A metadata object containing information about the model's performance and configuration.</returns>
    ModelMetadata<T> GetModelMetadata();
}
