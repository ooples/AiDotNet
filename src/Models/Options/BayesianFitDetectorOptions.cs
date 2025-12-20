namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Bayesian model fit detector, which evaluates how well a model fits the data.
/// </summary>
/// <remarks>
/// <para>
/// This class provides threshold values used to interpret Bayesian Information Criterion (BIC) or similar
/// Bayesian metrics that assess model fit. These thresholds help determine if a model is a good fit,
/// overfit (too complex), or underfit (too simple) for the given data.
/// </para>
/// <para><b>For Beginners:</b> When building AI models, it's important to know if your model is "just right" 
/// for your data. Think of it like Goldilocks choosing a bed - one can be too soft (overfit), one too hard (underfit), 
/// and one just right (good fit). This class helps set the thresholds for determining which category your model falls into.
/// </para>
/// <para>
/// An overfit model is like memorizing exam answers without understanding the concepts - it works perfectly for the 
/// practice questions but fails on the actual exam. An underfit model is too simple, like using a straight line to 
/// predict stock prices that go up and down. A good fit balances complexity and generalization, capturing the important 
/// patterns without getting distracted by random noise in the data.
/// </para>
/// </remarks>
public class BayesianFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for determining a good model fit.
    /// </summary>
    /// <value>The good fit threshold value, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// This threshold is used to identify models that have an appropriate balance between complexity and fit to the data.
    /// Models with Bayesian metric values around this threshold are considered to have a good fit.
    /// </para>
    /// <para><b>For Beginners:</b> This value (default: 5) represents the sweet spot for your model's performance. 
    /// When your model's score is close to this number, it suggests your model has found the right balance - it's 
    /// capturing the important patterns in your data without being either too simple or too complex. Think of it like 
    /// the "just right" porridge temperature in Goldilocks - not too hot, not too cold. Models with scores near this 
    /// threshold are likely to perform well not just on your training data but also on new, unseen data.</para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 5;

    /// <summary>
    /// Gets or sets the threshold for detecting model overfitting.
    /// </summary>
    /// <value>The overfit threshold value, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// Bayesian metric values above this threshold suggest that the model may be overfitting the data,
    /// meaning it is too complex and may be capturing noise rather than true patterns.
    /// </para>
    /// <para><b>For Beginners:</b> When your model's score exceeds this value (default: 10), it's a warning sign 
    /// that your model might be "memorizing" your training data instead of learning general patterns. This is called 
    /// overfitting. It's like a student who memorizes specific test questions but can't solve similar problems on the 
    /// actual exam. An overfit model performs extremely well on the data it was trained on but fails when given new data. 
    /// If your model exceeds this threshold, consider simplifying it by reducing features or using regularization techniques.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 10;

    /// <summary>
    /// Gets or sets the threshold for detecting model underfitting.
    /// </summary>
    /// <value>The underfit threshold value, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// Bayesian metric values below this threshold suggest that the model may be underfitting the data,
    /// meaning it is too simple and fails to capture important patterns.
    /// </para>
    /// <para><b>For Beginners:</b> When your model's score falls below this value (default: 2), it suggests your 
    /// model is too simple to capture the important patterns in your data. This is called underfitting. Imagine trying 
    /// to predict house prices using only the number of bedrooms, while ignoring location, size, age, and other important 
    /// factors. An underfit model makes oversimplified predictions that miss key relationships in the data. If your model 
    /// falls below this threshold, try adding more features, using a more complex model type, or reducing regularization.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 2;
}
