using AiDotNet.LossFunctions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a complete machine learning model that combines prediction capabilities with serialization and checkpointing support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface combines all the important capabilities that a complete AI model needs.
///
/// Think of IFullModel as the core contract for a machine learning model. It provides:
///
/// 1. The ability to make predictions (from IModel)
///    - Process input data and produce output predictions
///
/// 2. The ability to save and load the model (from IModelSerializer and ICheckpointableModel)
///    - File-based saving/loading for deployment
///    - Stream-based checkpointing for training resumption
///
/// 3. Feature importance reporting (from IFeatureImportance)
///    - Understand which features contribute most to predictions
///
/// Optional capabilities (check with 'is' or InterfaceGuard before using):
/// - IParameterizable: Get/set model parameters (linear models, neural networks)
/// - IGradientComputable: Compute and apply gradients (gradient-based optimization)
/// - IFeatureAware: Feature selection and tracking
///
/// Not all models support all capabilities. Tree-based and ensemble models
/// may not implement IParameterizable or IGradientComputable.
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
    IModelSerializer, ICheckpointableModel, IFeatureImportance<T>,
    ICloneable<IFullModel<T, TInput, TOutput>>
{
    /// <summary>
    /// Gets the default loss function used by this model for gradient computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This loss function is used when computing gradients (on models that implement
    /// <see cref="IGradientComputable{T, TInput, TOutput}"/>) without explicitly providing one.
    /// It represents the model's primary training objective.
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
