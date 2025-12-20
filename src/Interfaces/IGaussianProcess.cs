namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for Gaussian Process regression, a powerful probabilistic machine learning technique.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A Gaussian Process is a flexible machine learning approach that not only makes predictions 
/// but also tells you how confident it is about each prediction.
/// 
/// Imagine you're trying to predict house prices in different neighborhoods:
/// - Traditional models might just say "this house costs $300,000"
/// - A Gaussian Process would say "this house costs about $300,000, and I'm very confident 
///   the price is between $290,000 and $310,000"
/// 
/// This is especially useful when:
/// - You have limited data
/// - You need to know how certain the model is about its predictions
/// - You want to make decisions that account for uncertainty
/// 
/// Unlike many other machine learning methods, Gaussian Processes:
/// - Don't assume a specific form for the relationship between inputs and outputs
/// - Automatically adapt to the complexity of your data
/// - Provide uncertainty estimates for each prediction
/// 
/// The "Gaussian" part refers to the normal distribution (bell curve) used to represent uncertainty.
/// The "Process" part means it works with functions rather than simple values.
/// </remarks>
public interface IGaussianProcess<T>
{
    /// <summary>
    /// Trains the Gaussian Process model on the provided data.
    /// </summary>
    /// <param name="X">The input features matrix, where each row represents an observation and each column represents a feature.</param>
    /// <param name="y">The target values vector corresponding to each observation in X.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method teaches the model using your training data.
    /// 
    /// The parameters:
    /// - X: Your input data organized as a matrix (a table of numbers)
    ///   - Each row represents one example (like one house)
    ///   - Each column represents one feature (like size, location, age of the house)
    /// - y: Your output data as a vector (a list of numbers)
    ///   - Each value is what you want to predict (like the price of each house)
    /// 
    /// During fitting, the Gaussian Process:
    /// 1. Analyzes patterns in your data
    /// 2. Learns how different features relate to the target values
    /// 3. Builds an internal model of the relationships
    /// 4. Prepares to make predictions with uncertainty estimates
    /// 
    /// Unlike many other models, Gaussian Processes don't just find coefficients or weights.
    /// Instead, they remember the training data and use it directly when making predictions.
    /// </remarks>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts the mean value and variance (uncertainty) for a new input point.
    /// </summary>
    /// <param name="x">The input feature vector for which to make a prediction.</param>
    /// <returns>A tuple containing the predicted mean value and the variance (uncertainty) of the prediction.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method makes a prediction and tells you how confident it is about that prediction.
    /// 
    /// The parameter:
    /// - x: The features of the new example you want to predict (like the size, location, and age of a house)
    /// 
    /// The method returns two values:
    /// 1. mean: The predicted value (like the estimated house price)
    /// 2. variance: How uncertain the model is about this prediction
    ///    - A small variance means the model is confident (narrow range of likely values)
    ///    - A large variance means the model is uncertain (wide range of possible values)
    /// 
    /// This uncertainty information is what makes Gaussian Processes special. It helps you:
    /// - Know when to trust or question the model's predictions
    /// - Identify areas where you need more training data
    /// - Make better decisions when the prediction is uncertain
    /// 
    /// For example, if predicting house prices:
    /// - "This house costs $300,000 ± $5,000" (low variance, high confidence)
    /// - "This house costs $300,000 ± $50,000" (high variance, low confidence)
    /// </remarks>
    (T mean, T variance) Predict(Vector<T> x);

    /// <summary>
    /// Updates the kernel function used by the Gaussian Process.
    /// </summary>
    /// <param name="kernel">The new kernel function to use.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method changes how the model measures similarity between data points.
    /// 
    /// The parameter:
    /// - kernel: A function that determines how similar two data points are
    /// 
    /// The kernel function is at the heart of a Gaussian Process. It defines:
    /// - How influence spreads between data points
    /// - What patterns the model can recognize (linear, periodic, etc.)
    /// - How smooth the resulting predictions will be
    /// 
    /// Think of the kernel as defining the "personality" of your model:
    /// - Some kernels make the model see smooth, gradual changes
    /// - Others allow the model to capture more abrupt changes
    /// - Some can detect periodic patterns (like seasonal effects)
    /// 
    /// Changing the kernel can dramatically change how your model behaves, even with the same training data.
    /// This method allows you to update the kernel without retraining the entire model from scratch.
    /// </remarks>
    void UpdateKernel(IKernelFunction<T> kernel);
}
