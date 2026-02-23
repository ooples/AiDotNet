namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Hybrid Fit Detector, which combines multiple model evaluation techniques
/// to provide a comprehensive assessment of model quality.
/// </summary>
/// <remarks>
/// <para>
/// The Hybrid Fit Detector uses a combination of approaches to evaluate model fit, including comparing
/// training and validation performance, analyzing residuals, and examining model complexity. This
/// comprehensive approach provides a more robust assessment than any single method alone, helping to
/// identify overfitting, underfitting, and good fit conditions with greater confidence.
/// </para>
/// <para><b>For Beginners:</b> This detector is like having a team of experts evaluate your machine
/// learning model from different angles. Instead of relying on just one way to check if your model is
/// learning properly, it uses several different methods and combines their insights.
/// 
/// Think of it like a comprehensive health checkup that includes multiple tests (blood work, physical
/// exam, imaging, etc.) rather than just checking your temperature. By looking at your model from
/// multiple perspectives, the Hybrid Fit Detector can give you a more complete picture of how well
/// your model is learning and where it might be having problems.
/// 
/// The detector helps identify three common scenarios:
/// - Overfitting: Your model has "memorized" the training data but doesn't generalize well to new data
/// - Underfitting: Your model is too simple and isn't capturing important patterns in the data
/// - Good Fit: Your model has found the right balance, learning meaningful patterns that generalize well
/// 
/// This hybrid approach is especially useful when individual detection methods might give conflicting
/// signals or when you want extra confidence in your model quality assessment.</para>
/// </remarks>
public class HybridFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on a composite score from multiple evaluation methods.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be overfitting based on a composite score
    /// that combines multiple evaluation metrics. The exact calculation depends on the implementation,
    /// but typically includes factors like the gap between training and validation performance, the
    /// complexity of the model relative to the amount of training data, and patterns in the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is "memorizing" the training
    /// data instead of learning general patterns. With the default value of 0.2, if the composite overfitting
    /// score exceeds 20%, your model is flagged as overfitting.
    /// 
    /// The composite score looks at several warning signs of overfitting, such as:
    /// - How much better your model performs on training data than on validation data
    /// - Whether your model is unnecessarily complex for the amount of data you have
    /// - If your model's errors show suspicious patterns
    /// 
    /// For example, a model might have only a small gap between training and validation performance
    /// (which alone wouldn't trigger an overfitting warning), but if it's also very complex and shows
    /// certain patterns in its errors, the combined evidence might push the composite score above the
    /// threshold, indicating overfitting.
    /// 
    /// When overfitting is detected, you might want to:
    /// - Use more regularization to penalize complexity
    /// - Reduce model complexity (fewer features, simpler model)
    /// - Gather more training data
    /// - Use techniques like early stopping or dropout</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on a composite score from multiple evaluation methods.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.5 (50%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to be underfitting based on a composite score
    /// that combines multiple evaluation metrics. The exact calculation depends on the implementation,
    /// but typically includes factors like the absolute performance on both training and validation data,
    /// the simplicity of the model relative to the complexity of the problem, and patterns in the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is "not learning enough"
    /// from the training data. With the default value of 0.5, if the composite underfitting score exceeds
    /// 50%, your model is flagged as underfitting.
    /// 
    /// The composite score looks at several warning signs of underfitting, such as:
    /// - Poor performance on both training and validation data
    /// - Whether your model is too simple for the complexity of the problem
    /// - If your model's errors show patterns that suggest missed relationships
    /// 
    /// For example, a model might have mediocre performance that alone wouldn't definitively indicate
    /// underfitting, but if it's also very simple and shows strong patterns in its errors that suggest
    /// missed relationships, the combined evidence might push the composite score above the threshold,
    /// indicating underfitting.
    /// 
    /// When underfitting is detected, you might want to:
    /// - Increase model complexity (more features, more complex model)
    /// - Train for more iterations or epochs
    /// - Reduce regularization strength
    /// - Engineer better features
    /// - Try a different type of model altogether</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the threshold for confirming good fit based on a composite score from multiple evaluation methods.
    /// </summary>
    /// <value>The good fit threshold, defaulting to 0.8 (80%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to have a good fit based on a composite score
    /// that combines multiple evaluation metrics. The exact calculation depends on the implementation,
    /// but typically includes factors like balanced performance between training and validation data,
    /// appropriate model complexity for the problem, and the absence of concerning patterns in the residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model has achieved the right
    /// balance - learning meaningful patterns without memorizing the training data. With the default value
    /// of 0.8, if the composite good fit score exceeds 80%, your model is considered to have a good fit.
    /// 
    /// The composite score looks at several indicators of good fit, such as:
    /// - Strong performance on both training and validation data, with the gap between them not being too large
    /// - Appropriate model complexity for the problem and amount of data
    /// - Residuals (errors) that look random rather than showing patterns
    /// - Consistent performance across different subsets of the data
    /// 
    /// For example, a model with good validation performance might not be flagged as having a good fit if
    /// there are other concerning signs like highly non-random errors. Conversely, a model with slightly
    /// lower performance but very clean residuals and appropriate complexity might be recognized as having
    /// a good fit.
    /// 
    /// A good fit means your model has struck the right balance between underfitting and overfitting - it's
    /// complex enough to learn from the data but not so complex that it just memorizes it. This is the
    /// "Goldilocks zone" we aim for in machine learning.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.8;
}
