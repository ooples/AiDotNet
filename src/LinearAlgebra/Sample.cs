namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a single data sample consisting of features and a target value for machine learning algorithms.
/// </summary>
/// <typeparam name="T">The data type of the target value and feature vector elements (e.g., float, double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A Sample is like a single example that we use to train or test a machine learning model.
/// 
/// Think of it this way: if you were teaching someone to identify fruits, each Sample would be one fruit.
/// The "Features" would be the characteristics you observe (color, size, shape, texture),
/// and the "Target" would be the correct answer (apple, banana, orange).
/// 
/// For instance, in a house price prediction model:
/// - Features might include: number of bedrooms, square footage, neighborhood rating, etc.
/// - Target would be the actual price of the house
/// 
/// The machine learning algorithm learns from many of these samples to make predictions on new data.
/// </para>
/// </remarks>
public class Sample<T>
{
    /// <summary>
    /// Gets or sets the feature vector containing the input values for this sample.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features are the measurable properties or characteristics of what we're studying.
    /// They are the inputs that our machine learning model uses to make predictions.
    /// 
    /// A Vector is simply a collection of numbers arranged in a specific order. In this case, each number
    /// in the vector represents a different feature or characteristic of our sample.
    /// 
    /// For example, if we're predicting house prices:
    /// - Features[0] might be the number of bedrooms
    /// - Features[1] might be the square footage
    /// - Features[2] might be the age of the house
    /// - And so on...
    /// 
    /// The model learns how these features relate to the target value during training.
    /// </para>
    /// </remarks>
    public Vector<T> Features { get; set; }

    /// <summary>
    /// Gets or sets the target value (or label) for this sample.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Target is the correct answer or outcome that we're trying to predict.
    /// 
    /// In supervised learning, we provide both the features and the target to the algorithm during training.
    /// The algorithm learns to predict the target based on the features.
    /// 
    /// For example:
    /// - In a house price prediction model, the target would be the actual price of the house.
    /// - In an email spam filter, the target would be whether the email is "spam" or "not spam".
    /// - In a medical diagnosis system, the target might be whether a patient has a certain condition.
    /// 
    /// After training, when we only have features (inputs) but don't know the target,
    /// the model can make predictions for us.
    /// </para>
    /// </remarks>
    public T Target { get; set; }

    /// <summary>
    /// Initializes a new instance of the Sample class with the specified features and target.
    /// </summary>
    /// <param name="features">The feature vector containing input values.</param>
    /// <param name="target">The target value or label for this sample.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the constructor method that creates a new Sample object.
    /// When you want to create a new sample in your code, you would use this method by providing:
    /// 
    /// 1. A vector of features (the input characteristics)
    /// 2. A target value (the correct answer)
    /// 
    /// For example:
    /// <code>
    /// // Create a feature vector for a house with 3 bedrooms, 1500 sq ft, and 10 years old
    /// var features = new Vector&lt;double&gt;([3, 1500, 10]);
    /// 
    /// // The house price is $250,000
    /// double price = 250000;
    /// 
    /// // Create a sample with these features and target
    /// var houseSample = new Sample&lt;double&gt;(features, price);
    /// </code>
    /// </para>
    /// </remarks>
    public Sample(Vector<T> features, T target)
    {
        Features = features;
        Target = target;
    }
}
