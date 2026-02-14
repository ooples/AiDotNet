using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for data loaders that provide standard input-output (X, Y) data for supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// This interface is for standard supervised learning scenarios where you have:
/// - Input features (X): The data used to make predictions
/// - Output labels (Y): The correct answers the model should learn to predict
/// </para>
/// <para><b>For Beginners:</b> Most machine learning tasks fall into this pattern:
///
/// **Example: House Price Prediction**
/// - X (inputs): Square footage, number of bedrooms, location, age
/// - Y (outputs): The actual house price
///
/// **Example: Email Spam Detection**
/// - X (inputs): Email text features (word counts, sender info, etc.)
/// - Y (outputs): Label (spam=1, not spam=0)
///
/// The data loader loads this data from files, databases, or other sources
/// and provides it in the format your model needs for training.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("InputOutputDataLoader")]
public interface IInputOutputDataLoader<T, TInput, TOutput> :
    IDataLoader<T>,
    IBatchIterable<(TInput Features, TOutput Labels)>,
    IShuffleable
{
    /// <summary>
    /// Gets all input features as a single data structure.
    /// </summary>
    /// <remarks>
    /// This provides access to the complete feature set. For large datasets,
    /// prefer using batch iteration methods instead of loading everything at once.
    /// </remarks>
    TInput Features { get; }

    /// <summary>
    /// Gets all output labels as a single data structure.
    /// </summary>
    /// <remarks>
    /// This provides access to all labels. For large datasets,
    /// prefer using batch iteration methods instead of loading everything at once.
    /// </remarks>
    TOutput Labels { get; }

    /// <summary>
    /// Gets the number of features per sample.
    /// </summary>
    int FeatureCount { get; }

    /// <summary>
    /// Gets the number of output dimensions (1 for regression/binary classification,
    /// N for multi-class with N classes).
    /// </summary>
    int OutputDimension { get; }

    /// <summary>
    /// Creates a train/validation/test split of the data.
    /// </summary>
    /// <param name="trainRatio">Fraction of data for training (0.0 to 1.0).</param>
    /// <param name="validationRatio">Fraction of data for validation (0.0 to 1.0).</param>
    /// <param name="seed">Optional random seed for reproducible splits.</param>
    /// <returns>A tuple containing three data loaders: (train, validation, test).</returns>
    /// <remarks>
    /// <para>
    /// The test ratio is implicitly 1 - trainRatio - validationRatio.
    /// </para>
    /// <para><b>For Beginners:</b> Splitting data is crucial for evaluating your model:
    /// - **Training set**: Data the model learns from
    /// - **Validation set**: Data used to tune hyperparameters and prevent overfitting
    /// - **Test set**: Data used only once at the end to get an unbiased performance estimate
    ///
    /// Common splits are 60/20/20 or 70/15/15 (train/validation/test).
    /// </para>
    /// </remarks>
    (IInputOutputDataLoader<T, TInput, TOutput> Train,
     IInputOutputDataLoader<T, TInput, TOutput> Validation,
     IInputOutputDataLoader<T, TInput, TOutput> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null);
}
