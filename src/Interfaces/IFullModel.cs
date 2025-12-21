using AiDotNet.LossFunctions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a complete machine learning model that combines prediction capabilities with serialization and checkpointing support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface combines all the important capabilities that a complete AI model needs.
///
/// Think of IFullModel as a "complete package" for a machine learning model. It combines:
///
/// 1. The ability to make predictions (from IModel)
///    - This is like a calculator that can process your data and give you answers
///    - For example, predicting house prices based on features like size and location
///
/// 2. The ability to save and load the model (from IModelSerializer and ICheckpointableModel)
///    - IModelSerializer: File-based saving/loading for long-term storage
///    - ICheckpointableModel: Stream-based checkpointing for training resumption and distillation
///    - This is like being able to save your work in a document and open it later
///    - It allows you to train a model once and then use it many times without retraining
///    - It also lets you share your trained model with others
///
/// 3. The ability to compute gradients explicitly (from IGradientComputable)
///    - This enables distributed training across multiple GPUs/machines
///    - Allows manual control over the training process for advanced users
///    - Supports meta-learning algorithms that need fine-grained gradient control
///
/// By implementing this interface, a model class provides everything needed for practical use:
/// you can train it, use it for predictions, save it to disk, load it back when needed,
/// checkpoint it during training, and even distribute training across multiple machines for faster learning on large datasets.
///
/// This is particularly useful for production environments where models need to be:
/// - Trained once (which might take a long time)
/// - Saved to disk for deployment
/// - Checkpointed during training to prevent data loss
/// - Loaded quickly when needed to make predictions
/// - Possibly trained in a distributed manner across multiple GPUs
/// - Updated with new data periodically
/// - Used in knowledge distillation as teacher or student models
/// </remarks>
public interface IFullModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>,
    IModelSerializer, ICheckpointableModel, IParameterizable<T, TInput, TOutput>, IFeatureAware, IFeatureImportance<T>,
    ICloneable<IFullModel<T, TInput, TOutput>>, IGradientComputable<T, TInput, TOutput>, IJitCompilable<T>
{
    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This loss function is used when calling <see cref="IGradientComputable{T, TInput, TOutput}.ComputeGradients"/>
    /// without explicitly providing a loss function. It represents the model's primary training objective.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// The loss function tells the model "what counts as a mistake". For example:
    /// - For regression (predicting numbers): Mean Squared Error measures how far predictions are from actual values
    /// - For classification (predicting categories): Cross Entropy measures how confident the model is in the right category
    ///
    /// This property provides a sensible default so you don't have to specify the loss function every time,
    /// but you can still override it if needed for special cases.
    /// </para>
    /// <para><b>Distributed Training:</b>
    /// In distributed training, all workers use the same loss function to ensure consistent gradient computation.
    /// The default loss function is automatically used when workers compute local gradients.
    /// </para>
    /// </remarks>
    /// <exception cref="System.InvalidOperationException">
    /// Thrown if accessed before the model has been configured with a loss function.
    /// </exception>
    ILossFunction<T> DefaultLossFunction { get; }
}
