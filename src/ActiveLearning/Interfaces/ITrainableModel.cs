namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for models that support training on datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A trainable model is one that can learn from data.
/// The training process adjusts the model's internal parameters to improve its
/// predictions based on the provided examples.</para>
///
/// <para><b>Common Uses:</b></para>
/// <list type="bullet">
/// <item><description>Active learning iteratively trains models on newly labeled samples</description></item>
/// <item><description>Query by Committee trains multiple models on bootstrap samples</description></item>
/// <item><description>Continual learning trains models on sequential tasks</description></item>
/// </list>
/// </remarks>
public interface ITrainableModel<T, TInput, TOutput>
{
    /// <summary>
    /// Trains the model on the provided dataset.
    /// </summary>
    /// <param name="dataset">The dataset containing input-output pairs for training.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method updates the model's internal parameters
    /// by learning patterns from the training data. After training, the model should
    /// produce better predictions for similar inputs.</para>
    /// </remarks>
    void Train(IDataset<T, TInput, TOutput> dataset);

    /// <summary>
    /// Trains the model for a specified number of epochs.
    /// </summary>
    /// <param name="dataset">The dataset containing input-output pairs for training.</param>
    /// <param name="epochs">The number of training epochs (passes through the data).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> An epoch is one complete pass through all training data.
    /// More epochs generally lead to better learning, but too many can cause overfitting
    /// (memorizing the training data rather than learning general patterns).</para>
    /// </remarks>
    void Train(IDataset<T, TInput, TOutput> dataset, int epochs);

    /// <summary>
    /// Gets whether the model has been trained.
    /// </summary>
    bool IsTrained { get; }

    /// <summary>
    /// Resets the model to its initial untrained state.
    /// </summary>
    /// <remarks>
    /// <para>Clears all learned parameters, effectively creating a fresh model.
    /// Useful when retraining from scratch or for cross-validation.</para>
    /// </remarks>
    void Reset();
}
