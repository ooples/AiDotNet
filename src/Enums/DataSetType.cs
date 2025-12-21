namespace AiDotNet.Enums;

/// <summary>
/// Represents the different types of datasets used in machine learning workflows.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In machine learning, we typically split our data into different sets for different purposes.
/// 
/// Think of it like learning to cook:
/// - Training data is like practicing recipes while learning (the model learns from this data)
/// - Validation data is like having someone taste your food and give feedback while you're still learning (helps tune your model)
/// - Testing data is like serving to customers who've never had your food before (final evaluation of your model)
/// 
/// This separation helps ensure that your model can generalize well to new, unseen data rather than just memorizing 
/// the examples it was trained on.
/// </para>
/// </remarks>
public enum DataSetType
{
    /// <summary>
    /// The dataset used to train the model by adjusting its parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Training dataset is the largest portion of your data (typically 60-80%) and is used to teach the model.
    /// 
    /// During training:
    /// - The model sees both the input features and the correct answers (labels)
    /// - It adjusts its internal parameters to minimize errors
    /// - It learns patterns and relationships in the data
    /// 
    /// Think of this as the practice data that the model uses to learn how to make predictions.
    /// Just like a student studying examples to learn concepts, the model studies this data to learn patterns.
    /// 
    /// The model directly learns from and fits to this data, which is why we need separate validation and testing
    /// sets to ensure the model hasn't just memorized the training examples.
    /// </para>
    /// </remarks>
    Training,

    /// <summary>
    /// The dataset used to evaluate the final model performance on unseen data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Testing dataset is kept completely separate from training and validation (typically 10-20% of your data).
    /// It's only used once the model is fully trained and tuned.
    /// 
    /// Key characteristics:
    /// - The model never sees this data during training or tuning
    /// - It provides an unbiased evaluation of the final model
    /// - It simulates how the model will perform on new, real-world data
    /// 
    /// Think of this as the final exam that truly tests what the model has learned.
    /// Just like a final exam tests a student's knowledge on new problems they haven't seen before,
    /// the test set evaluates how well the model can generalize to new data.
    /// 
    /// The performance on the test set is what you report as the expected real-world performance of your model.
    /// </para>
    /// </remarks>
    Testing,

    /// <summary>
    /// The dataset used during model development to tune hyperparameters and prevent overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Validation dataset is separate from training data (typically 10-20% of your data) and helps you make
    /// decisions about model design and hyperparameters.
    /// 
    /// Key uses:
    /// - Tuning hyperparameters (like learning rate or model complexity)
    /// - Selecting the best model architecture
    /// - Determining when to stop training to prevent overfitting
    /// - Comparing different modeling approaches
    /// 
    /// Think of this as practice tests while studying.
    /// Just like practice tests help a student adjust their study strategy before the final exam,
    /// the validation set helps you adjust your model before final testing.
    /// 
    /// Unlike the test set, the validation set can influence model design decisions, but the model
    /// doesn't directly learn from this data (it doesn't update its parameters based on validation data).
    /// </para>
    /// </remarks>
    Validation
}
