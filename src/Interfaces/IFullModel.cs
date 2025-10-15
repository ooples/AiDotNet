namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a complete machine learning model that combines prediction capabilities with serialization support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface combines two important capabilities that a complete AI model needs.
/// 
/// Think of IFullModel as a "complete package" for a machine learning model. It combines:
/// 
/// 1. The ability to make predictions (from IModel)
///    - This is like a calculator that can process your data and give you answers
///    - For example, predicting house prices based on features like size and location
/// 
/// 2. The ability to save and load the model (from IModelSerializer)
///    - This is like being able to save your work in a document and open it later
///    - It allows you to train a model once and then use it many times without retraining
///    - It also lets you share your trained model with others
/// 
/// By implementing this interface, a model class provides everything needed for practical use:
/// you can train it, use it for predictions, save it to disk, and load it back when needed.
/// 
/// This is particularly useful for production environments where models need to be:
/// - Trained once (which might take a long time)
/// - Saved to disk
/// - Loaded quickly when needed to make predictions
/// - Possibly updated with new data periodically
/// </remarks>
public interface IFullModel<T, TInput, TOutput> : IInterpretableModel<T, TInput, TOutput>, 
    IModelSerializer, IParameterizable<T, TInput, TOutput>, IFeatureAware, ICloneable<IFullModel<T, TInput, TOutput>>
{
    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    int ParameterCount { get; }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    void SaveModel(string filePath);

    /// <summary>
    /// Gets the feature importance scores for the model.
    /// </summary>
    /// <returns>A dictionary mapping feature indices/names to their importance scores.</returns>
    Dictionary<string, T> GetFeatureImportance();
}