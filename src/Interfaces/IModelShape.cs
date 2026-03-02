namespace AiDotNet.Interfaces;

/// <summary>
/// Provides shape metadata for a machine learning model, describing its expected input and output dimensions.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Every machine learning model expects data in a specific shape (size and dimensions).
/// For example, a model trained on images might expect input of shape [3, 224, 224] (3 color channels, 224x224 pixels),
/// and output a shape of [1000] (1000 class probabilities).
///
/// This interface allows models to self-describe their shapes, which is useful for:
/// - Validating input data before prediction
/// - Auto-configuring serving infrastructure
/// - Displaying model information without loading full weights
/// - Building model pipelines where output of one model feeds into another
///
/// This is an optional interface — not all models need to implement it. Base classes like
/// NeuralNetworkBase and ClusteringBase implement it automatically.
/// </remarks>
public interface IModelShape
{
    /// <summary>
    /// Gets the expected input shape of the model.
    /// </summary>
    /// <returns>
    /// An array of integers representing the input dimensions.
    /// For example, [784] for a flat input of 784 features, or [3, 224, 224] for a 3-channel image.
    /// </returns>
    int[] GetInputShape();

    /// <summary>
    /// Gets the output shape of the model.
    /// </summary>
    /// <returns>
    /// An array of integers representing the output dimensions.
    /// For example, [10] for 10-class classification, or [1] for single-value regression.
    /// </returns>
    int[] GetOutputShape();
}
