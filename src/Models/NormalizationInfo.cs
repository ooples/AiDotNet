namespace AiDotNet.Models;

/// <summary>
/// Represents information about how data normalization is performed for a model, including the normalizer and parameters.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the information needed to normalize input features and denormalize predictions for a machine 
/// learning model. Normalization is a preprocessing technique that scales data to a standard range, which can improve the 
/// performance and stability of many machine learning algorithms. This class stores the normalizer object that performs the 
/// actual normalization operations, as well as the parameters that describe how each feature and the target variable were 
/// normalized during training.
/// </para>
/// <para><b>For Beginners:</b> This class stores information about how data is scaled before and after prediction.
/// 
/// When working with machine learning models:
/// - Input data often needs to be scaled to a standard range (like 0-1)
/// - After prediction, the results may need to be scaled back to the original range
/// - You need to store how this scaling was done during training
/// 
/// This class stores all that information, including:
/// - The normalizer object that performs the actual scaling operations
/// - Parameters for how each input feature should be normalized
/// - Parameters for how the target variable (what you're predicting) should be denormalized
/// 
/// This ensures that new data is processed consistently with how the model was trained,
/// which is essential for accurate predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NormalizationInfo<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the normalizer used to normalize input data and denormalize predictions.
    /// </summary>
    /// <value>An implementation of INormalizer&lt;T&gt; that performs normalization operations.</value>
    /// <remarks>
    /// <para>
    /// This property contains the normalizer object that performs the actual normalization and denormalization operations. 
    /// The normalizer implements the INormalizer&lt;T&gt; interface, which provides methods for normalizing matrices and vectors 
    /// and denormalizing the results. Different implementations of this interface might use different normalization techniques, 
    /// such as min-max scaling, z-score normalization, or robust scaling. The property is nullable because it might not be 
    /// initialized immediately, but it must be set before normalization operations can be performed.
    /// </para>
    /// <para><b>For Beginners:</b> This is the object that actually performs the scaling operations.
    /// 
    /// The normalizer:
    /// - Implements the INormalizer interface
    /// - Contains methods to scale data to a standard range (normalize)
    /// - Contains methods to convert scaled data back to original values (denormalize)
    /// - Could use different techniques like min-max scaling or z-score normalization
    /// 
    /// This property is marked as nullable (with the ? symbol) because:
    /// - It might not be initialized immediately
    /// - When deserializing from storage, it might be populated later
    /// 
    /// However, it must be set before you can normalize or denormalize data,
    /// or you'll get an error when trying to use it.
    /// </para>
    /// </remarks>
    public INormalizer<T, TInput, TOutput>? Normalizer { get; set; }

    /// <summary>
    /// Gets or sets the normalization parameters for each input feature.
    /// </summary>
    /// <value>A list of NormalizationParameters&lt;T&gt; objects, one for each input feature.</value>
    /// <remarks>
    /// <para>
    /// This property contains a list of normalization parameters, one for each input feature in the model. Each 
    /// NormalizationParameters&lt;T&gt; object contains the information needed to normalize a single feature, such as the 
    /// minimum and maximum values, mean and standard deviation, or other statistics depending on the normalization method. 
    /// These parameters are typically calculated during training based on the training data and are then used to normalize 
    /// new data in the same way. The order of the parameters in the list corresponds to the order of the features in the 
    /// input data.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the scaling information for each input feature.
    /// 
    /// The X parameters list:
    /// - Contains one set of parameters for each input feature
    /// - Stores information like minimum/maximum values or mean/standard deviation
    /// - Is used to ensure new data is scaled the same way as training data
    /// 
    /// For example, if your model has three input features:
    /// - The first element might contain parameters for normalizing "square_footage"
    /// - The second element might contain parameters for normalizing "num_bedrooms"
    /// - The third element might contain parameters for normalizing "location_score"
    /// 
    /// This information is essential because:
    /// - Each feature might have a different range of values
    /// - Each feature needs to be normalized consistently
    /// - The normalizer uses these parameters to perform the scaling
    /// </para>
    /// </remarks>
    public List<NormalizationParameters<T>> XParams { get; set; } = new List<NormalizationParameters<T>>();

    /// <summary>
    /// Gets or sets the normalization parameters for the target variable.
    /// </summary>
    /// <value>A NormalizationParameters&lt;T&gt; object for the target variable.</value>
    /// <remarks>
    /// <para>
    /// This property contains the normalization parameters for the target variable (the variable being predicted). The 
    /// NormalizationParameters&lt;T&gt; object contains the information needed to normalize the target variable during training 
    /// and to denormalize predictions to obtain the final results in the original scale. These parameters are typically 
    /// calculated during training based on the training data and are then used to denormalize predictions made on new data.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the scaling information for the target variable (what you're predicting).
    /// 
    /// The Y parameters:
    /// - Store information about how the target variable was normalized during training
    /// - Are used to convert predictions back to their original scale
    /// - Contain values like minimum/maximum or mean/standard deviation
    /// 
    /// For example, if your model predicts house prices:
    /// - The YParams might store that prices ranged from $100,000 to $1,500,000
    /// - This information is used to convert a normalized prediction like 0.25 back to $450,000
    /// 
    /// This is important because:
    /// - The model works with normalized values internally
    /// - But users need predictions in the original, meaningful units
    /// - Consistent denormalization ensures accurate final predictions
    /// </para>
    /// </remarks>
    public NormalizationParameters<T> YParams { get; set; } = new();

    /// <summary>
    /// Creates a deep copy of this NormalizationInfo instance.
    /// </summary>
    /// <returns>A new NormalizationInfo with copied values.</returns>
    public NormalizationInfo<T, TInput, TOutput> DeepCopy()
    {
        return new NormalizationInfo<T, TInput, TOutput>
        {
            Normalizer = Normalizer,
            XParams = new List<NormalizationParameters<T>>(XParams),
            YParams = YParams
        };
    }

    /// <summary>
    /// Creates a new NormalizationInfo instance. Since normalization parameters are independent of model parameters,
    /// this returns a copy with the same normalization settings.
    /// </summary>
    /// <param name="parameters">The model parameters (not used for normalization info).</param>
    /// <returns>A new NormalizationInfo with the same normalization settings.</returns>
    public NormalizationInfo<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return DeepCopy();
    }
}
