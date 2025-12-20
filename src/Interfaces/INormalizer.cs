namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for normalizing and denormalizing data for machine learning models.
/// </summary>
/// <remarks>
/// This interface provides functionality to transform data into a standardized range
/// (normalization) and to reverse this transformation (denormalization).
/// 
/// <b>For Beginners:</b> Normalization is like converting different measurements to the same scale.
/// 
/// What is normalization?
/// - Normalization transforms your data values to a standard range (usually between 0 and 1 or -1 and 1)
/// - It's like converting different units of measurement to a common scale
/// 
/// Why normalize data?
/// - Machine learning algorithms work better with normalized data
/// - It prevents features with large values from dominating those with smaller values
/// - It helps algorithms converge faster during training
/// 
/// Real-world example:
/// Imagine you're analyzing house prices based on:
/// - Size (1,000-5,000 square feet)
/// - Number of bedrooms (1-6)
/// - Age (0-100 years)
/// 
/// Without normalization, the size feature (with values in thousands) would have much more
/// influence than the bedroom count (with values of just 1-6). Normalization puts all these
/// features on the same scale so they can be compared fairly.
/// 
/// After using your model, you'll need to "denormalize" the results to convert them back
/// to their original scale (e.g., from a normalized value back to actual dollars).
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
public interface INormalizer<T, TInput, TOutput>
{
    /// <summary>
    /// Normalizes output data to a standard range.
    /// </summary>
    /// <remarks>
    /// This method transforms the input data to a normalized scale and returns both
    /// the normalized data and the parameters used for normalization.
    /// 
    /// <b>For Beginners:</b> This converts a list of numbers to a standard scale.
    /// 
    /// For example:
    /// - If your data contains ages (5, 18, 35, 62, 80), normalization might convert these to
    ///   values between 0 and 1: (0.0, 0.17, 0.4, 0.76, 1.0)
    /// - The normalization parameters (like minimum and maximum values) are also returned
    ///   so you can reverse this process later
    /// 
    /// This is typically used to prepare a single feature or target variable for machine learning.
    /// </remarks>
    /// <param name="data">The data to normalize.</param>
    /// <returns>
    /// A tuple containing:
    /// - The normalized data
    /// - The normalization parameters that were used (needed for later denormalization)
    /// </returns>
    (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data);

    /// <summary>
    /// Normalizes input data to a standard range.
    /// </summary>
    /// <remarks>
    /// This method transforms each feature of the input data to a normalized scale and returns
    /// both the normalized data and the parameters used for normalization of each feature.
    /// 
    /// <b>For Beginners:</b> This converts a table of numbers to a standard scale.
    /// 
    /// For example:
    /// - If your data contains information about houses (columns for size, bedrooms, age, price),
    ///   each feature is normalized separately
    /// - Size values might range from 1,000-5,000 sq ft, but are converted to 0-1
    /// - Bedroom values might range from 1-6, but are also converted to 0-1
    /// - The normalization parameters for each feature are saved separately
    /// 
    /// This is typically used to prepare multiple features (input variables) for machine learning.
    /// </remarks>
    /// <param name="data">The data to normalize.</param>
    /// <returns>
    /// A tuple containing:
    /// - The normalized data
    /// - A list of normalization parameters for each feature (needed for later denormalization)
    /// </returns>
    (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data);

    /// <summary>
    /// Reverses the normalization of data using the original normalization parameters.
    /// </summary>
    /// <remarks>
    /// This method transforms normalized data back to its original scale using the
    /// normalization parameters that were generated during the normalization process.
    /// 
    /// <b>For Beginners:</b> This converts normalized values back to their original scale.
    /// 
    /// For example:
    /// - If normalization converted ages (5, 18, 35, 62, 80) to (0.0, 0.17, 0.4, 0.76, 1.0)
    /// - Denormalization would convert (0.0, 0.17, 0.4, 0.76, 1.0) back to (5, 18, 35, 62, 80)
    /// 
    /// This is typically used:
    /// - After making predictions with a model (to convert predictions back to meaningful units)
    /// - When you need to interpret normalized data in its original context
    /// </remarks>
    /// <param name="data">The normalized data to denormalize.</param>
    /// <param name="parameters">The normalization parameters that were used during normalization.</param>
    /// <returns>The denormalized data in its original scale.</returns>
    TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters);

    /// <summary>
    /// Denormalizes model coefficients to make them applicable to non-normalized input data.
    /// </summary>
    /// <remarks>
    /// This method adjusts the coefficients of a model trained on normalized data so they can
    /// be used directly with non-normalized input data.
    /// 
    /// <b>For Beginners:</b> This converts the model's internal numbers to work with your original data.
    /// 
    /// When you train a model on normalized data:
    /// - The model learns coefficients (weights) that work with normalized values
    /// - To use the model with original, non-normalized data, these coefficients need adjustment
    /// - This method performs that adjustment
    /// 
    /// For example:
    /// - You trained a house price model on normalized data (size, bedrooms, age)
    /// - The model learned coefficients like [0.7, 0.2, -0.1]
    /// - This method converts these to coefficients that work with the original units
    ///   (e.g., dollars per square foot, dollars per bedroom, etc.)
    /// 
    /// This is typically used when:
    /// - You want to interpret what the model learned (e.g., "each additional bedroom adds $20,000")
    /// - You want to use the model with non-normalized input data
    /// </remarks>
    /// <param name="coefficients">The model coefficients from a model trained on normalized data.</param>
    /// <param name="xParams">The normalization parameters used for the input features (X).</param>
    /// <param name="yParams">The normalization parameters used for the target variable (Y).</param>
    /// <returns>Denormalized coefficients that can be used with original, non-normalized data.</returns>
    TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams);

    /// <summary>
    /// Calculates the denormalized Y-intercept (constant term) for a linear model.
    /// </summary>
    /// <remarks>
    /// This method computes the Y-intercept for a model trained on normalized data so it can
    /// be used with non-normalized input data.
    /// 
    /// <b>For Beginners:</b> This calculates the starting point (base value) for your model's predictions.
    /// 
    /// In a linear model like house price prediction:
    /// - Coefficients tell you how much each feature contributes (e.g., $100 per square foot)
    /// - The Y-intercept is the base value (e.g., $50,000 base price before adjustments)
    /// 
    /// When you train on normalized data:
    /// - The Y-intercept learned by the model works for normalized values
    /// - This method calculates the correct Y-intercept for original, non-normalized data
    /// 
    /// For example:
    /// - Your house price model might have a normalized Y-intercept of 0.5
    /// - This method converts it to a meaningful value like $150,000
    /// 
    /// This is typically used together with denormalized coefficients to create a complete
    /// model that works with original, non-normalized data.
    /// </remarks>
    /// <param name="xMatrix">The original input feature matrix (before normalization).</param>
    /// <param name="y">The original target vector (before normalization).</param>
    /// <param name="coefficients">The model coefficients (can be either normalized or denormalized).</param>
    /// <param name="xParams">The normalization parameters used for the input features (X).</param>
    /// <param name="yParams">The normalization parameters used for the target variable (Y).</param>
    /// <returns>The denormalized Y-intercept (constant term) for use with non-normalized data.</returns>
    T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams);
}
