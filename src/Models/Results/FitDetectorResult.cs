namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the result of a model fit detection analysis, which evaluates how well a model fits the data
/// and provides recommendations for improvement.
/// </summary>
/// <remarks>
/// <para>
/// When building statistical or machine learning models, it's important to assess how well the model fits the data. 
/// This class stores the results of such an assessment, including the type of fit detected (e.g., good fit, 
/// underfitting, overfitting), a confidence level for that assessment, and recommendations for improving the model. 
/// It also provides a flexible dictionary for storing additional information specific to different types of models 
/// or detection algorithms. This information helps data scientists and developers understand the quality of their 
/// models and take appropriate actions to improve them.
/// </para>
/// <para><b>For Beginners:</b> This class helps you understand how well your model fits your data.
/// 
/// When building statistical or machine learning models:
/// - You need to know if your model is a good match for your data
/// - Models can underfit (too simple) or overfit (too complex)
/// - Different types of models have different fit characteristics
/// 
/// This class stores:
/// - What type of fit was detected (good, underfit, overfit, etc.)
/// - How confident the detector is in its assessment
/// - Specific recommendations to improve your model
/// - Additional information that might be useful for diagnosis
/// 
/// This information helps you make informed decisions about how to adjust your model
/// to achieve better performance on both training and new data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for confidence values, typically float or double.</typeparam>
public class FitDetectorResult<T>
{
    /// <summary>
    /// Gets or sets the type of fit detected for the model.
    /// </summary>
    /// <value>A value from the FitType enumeration indicating the type of fit.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the type of fit detected for the model, such as good fit, underfitting, or overfitting. 
    /// A good fit indicates that the model appropriately captures the patterns in the data without being too simple 
    /// or too complex. Underfitting occurs when the model is too simple to capture the underlying patterns in the data, 
    /// resulting in poor performance on both training and test data. Overfitting occurs when the model is too complex 
    /// and captures noise in the training data, resulting in good performance on training data but poor performance on 
    /// test data. Other fit types might include multicollinearity (when predictor variables are highly correlated) or 
    /// heteroscedasticity (when the variance of errors varies across the range of a predictor).
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you what kind of fit was detected for your model.
    /// 
    /// The fit type:
    /// - Identifies the relationship between your model and your data
    /// - Helps you understand if your model needs to be adjusted
    /// - Guides your next steps in the modeling process
    /// 
    /// Common fit types include:
    /// - Good Fit: Your model appropriately captures the patterns in the data
    /// - Underfitting: Your model is too simple and misses important patterns
    /// - Overfitting: Your model is too complex and captures noise along with patterns
    /// - Multicollinearity: Your predictor variables are too highly correlated
    /// - Heteroscedasticity: The variance of errors changes across your data
    /// 
    /// For example, if this property is set to FitType.Overfitting, it means your model
    /// is likely too complex and needs to be simplified to generalize better.
    /// </para>
    /// </remarks>
    public FitType FitType { get; set; }

    /// <summary>
    /// Gets or sets the confidence level for the fit type assessment.
    /// </summary>
    /// <value>A numeric value representing the confidence level, or null if not applicable.</value>
    /// <remarks>
    /// <para>
    /// This property represents the confidence level or certainty with which the fit type was determined. Higher values 
    /// typically indicate greater confidence in the assessment. The exact interpretation of this value depends on the 
    /// specific detection algorithm used, but it generally provides a measure of how strongly the evidence supports the 
    /// identified fit type. This can be useful for prioritizing which issues to address first, focusing on those with 
    /// higher confidence levels. The property is nullable, allowing for cases where a confidence level is not applicable 
    /// or cannot be determined.
    /// </para>
    /// <para><b>For Beginners:</b> This value indicates how certain the detector is about the fit type assessment.
    /// 
    /// The confidence level:
    /// - Measures how sure the detector is about its assessment
    /// - Higher values indicate stronger evidence for the detected fit type
    /// - Can be null if confidence couldn't be determined
    /// 
    /// This value is important because:
    /// - It helps you prioritize which issues to address first
    /// - It indicates when a fit assessment might be borderline
    /// - It can guide how aggressively you should apply the recommendations
    /// 
    /// For example, a high confidence level for an overfitting detection suggests
    /// you should definitely take steps to reduce model complexity, while a low
    /// confidence might suggest more cautious adjustments.
    /// </para>
    /// </remarks>
    public T? ConfidenceLevel { get; set; }

    /// <summary>
    /// Gets or sets a list of recommendations for improving the model fit.
    /// </summary>
    /// <value>A list of string recommendations, initialized as an empty list.</value>
    /// <remarks>
    /// <para>
    /// This property provides specific recommendations for improving the model fit based on the detected fit type. 
    /// These recommendations are actionable suggestions that can help address issues with the model. For example, 
    /// if overfitting is detected, recommendations might include adding regularization, reducing model complexity, 
    /// or collecting more training data. If underfitting is detected, recommendations might include increasing model 
    /// complexity, adding more features, or using a different type of model. The recommendations are provided as 
    /// human-readable strings that can be directly presented to the user or logged for later review.
    /// </para>
    /// <para><b>For Beginners:</b> This list contains specific suggestions for improving your model.
    /// 
    /// The recommendations:
    /// - Provide actionable steps to address the detected fit issues
    /// - Are tailored to the specific fit type that was detected
    /// - Can be directly presented to users or logged for later review
    /// 
    /// Common recommendations might include:
    /// - For overfitting: "Add regularization to reduce model complexity"
    /// - For underfitting: "Try a more complex model architecture"
    /// - For multicollinearity: "Remove highly correlated features"
    /// 
    /// These recommendations are valuable because:
    /// - They translate the technical diagnosis into practical actions
    /// - They guide you toward better model performance
    /// - They help you learn about best practices in modeling
    /// 
    /// For example, if your model is overfitting, the recommendations might include
    /// "Add L2 regularization" and "Increase dropout rate to 0.5".
    /// </para>
    /// </remarks>
    public List<string> Recommendations { get; set; }

    /// <summary>
    /// Gets or sets additional information about the fit detection result.
    /// </summary>
    /// <value>A dictionary mapping string keys to object values, initialized as an empty dictionary.</value>
    /// <remarks>
    /// <para>
    /// This property provides a flexible way to store additional information about the fit detection result that 
    /// doesn't fit into the other properties. The dictionary can contain any type of information that might be useful 
    /// for understanding or addressing the fit issues. For example, it might include specific metrics like R-squared 
    /// values, variance inflation factors for multicollinearity detection, or plots showing residual patterns for 
    /// heteroscedasticity detection. The keys are string identifiers for the information, and the values can be any 
    /// object type, allowing for maximum flexibility in what can be stored.
    /// </para>
    /// <para><b>For Beginners:</b> This dictionary stores extra information that might be helpful for diagnosis.
    /// 
    /// The additional information:
    /// - Contains any extra details that don't fit in the other properties
    /// - Can store different types of data (numbers, strings, arrays, etc.)
    /// - Varies depending on the type of model and detection algorithm
    /// 
    /// Common types of additional information:
    /// - Specific metrics like R-squared or mean squared error
    /// - Threshold values used in the detection
    /// - Technical details about the model or data
    /// - References to related documentation or resources
    /// 
    /// This information is useful because:
    /// - It provides context for the fit assessment
    /// - It can help with more detailed diagnosis
    /// - It might be needed for implementing the recommendations
    /// 
    /// For example, for a multicollinearity detection, this might include
    /// the specific variance inflation factors for each feature.
    /// </para>
    /// </remarks>
    public Dictionary<string, object> AdditionalInfo { get; set; } = new Dictionary<string, object>();


    /// <summary>
    /// Initializes a new instance of the FitDetectorResult class with an empty list of recommendations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new FitDetectorResult instance with default values for FitType and ConfidenceLevel, 
    /// and initializes the Recommendations property to an empty list and AdditionalInfo to an empty dictionary. This 
    /// provides a clean starting point for storing fit detection results. The FitType and ConfidenceLevel properties 
    /// would typically be set after construction based on the results of the fit detection algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with default values.
    /// 
    /// When a new FitDetectorResult is created with this constructor:
    /// - The fit type is set to its default value
    /// - The confidence level is set to its default value
    /// - The recommendations list is initialized as empty
    /// - The additional info dictionary is initialized as empty
    /// 
    /// This initialization is important because:
    /// - It ensures consistent behavior regardless of how the object is created
    /// - It prevents potential issues with uninitialized collections
    /// - It provides a clean slate for the fit detector to populate
    /// 
    /// You typically won't need to call this constructor directly, as it will be
    /// used internally by the fit detection process.
    /// </para>
    /// </remarks>
    public FitDetectorResult()
    {
        Recommendations = new List<string>();
    }

    /// <summary>
    /// Initializes a new instance of the FitDetectorResult class with the specified fit type and confidence level.
    /// </summary>
    /// <param name="fitType">The type of fit detected for the model.</param>
    /// <param name="confidenceLevel">The confidence level for the fit type assessment.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new FitDetectorResult instance with the specified FitType and ConfidenceLevel, and 
    /// initializes the Recommendations property to an empty list and AdditionalInfo to an empty dictionary. This 
    /// constructor is useful when the fit type and confidence level are known at the time of creation, such as when 
    /// the result is being created directly after a fit detection algorithm has completed. The Recommendations and 
    /// AdditionalInfo properties can be populated after construction as needed.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with specific fit type and confidence values.
    /// 
    /// When a new FitDetectorResult is created with this constructor:
    /// - The fit type is set to the value you specify
    /// - The confidence level is set to the value you specify
    /// - The recommendations list is initialized as empty
    /// - The additional info dictionary is initialized as empty
    /// 
    /// This constructor is useful when:
    /// - You already know the fit type and confidence level
    /// - You want to create a result object in one step
    /// - You plan to add recommendations and additional info later
    /// 
    /// For example, you might use this constructor when you've determined that a model is
    /// overfitting with 85% confidence, and you'll add specific recommendations afterward.
    /// </para>
    /// </remarks>
    public FitDetectorResult(FitType fitType, T confidenceLevel)
    {
        FitType = fitType;
        ConfidenceLevel = confidenceLevel;
        Recommendations = new List<string>();
    }
}
