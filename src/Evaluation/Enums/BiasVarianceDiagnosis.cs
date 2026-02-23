namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the diagnosed bias-variance condition of a model.
/// </summary>
/// <remarks>
/// <para>
/// Bias-variance tradeoff is fundamental to understanding model performance:
/// <list type="bullet">
/// <item><b>Bias:</b> Error from overly simplistic assumptions (underfitting)</item>
/// <item><b>Variance:</b> Error from sensitivity to training data fluctuations (overfitting)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of it like throwing darts:
/// <list type="bullet">
/// <item><b>High bias:</b> Darts consistently miss the bullseye in the same direction (systematic error)</item>
/// <item><b>High variance:</b> Darts scattered all over (inconsistent)</item>
/// <item><b>Good fit:</b> Darts clustered around the bullseye</item>
/// </list>
/// </para>
/// </remarks>
public enum BiasVarianceDiagnosis
{
    /// <summary>
    /// Model exhibits high bias (underfitting): Poor performance on both training and test data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model is too simple to capture the underlying patterns.
    /// Both training and test errors are high and similar.</para>
    /// <para><b>Symptoms:</b></para>
    /// <list type="bullet">
    /// <item>High training error</item>
    /// <item>High test error (similar to training error)</item>
    /// <item>Training and test curves plateau early</item>
    /// </list>
    /// <para><b>Solutions:</b></para>
    /// <list type="bullet">
    /// <item>Use a more complex model</item>
    /// <item>Add more features</item>
    /// <item>Reduce regularization</item>
    /// <item>Train longer (if using iterative methods)</item>
    /// </list>
    /// </remarks>
    HighBias = 0,

    /// <summary>
    /// Model exhibits high variance (overfitting): Good training performance but poor test performance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model memorizes training data instead of learning
    /// general patterns. It performs great on training data but poorly on new data.</para>
    /// <para><b>Symptoms:</b></para>
    /// <list type="bullet">
    /// <item>Low training error</item>
    /// <item>High test error (much higher than training)</item>
    /// <item>Large gap between training and test curves</item>
    /// </list>
    /// <para><b>Solutions:</b></para>
    /// <list type="bullet">
    /// <item>Get more training data</item>
    /// <item>Use a simpler model</item>
    /// <item>Add regularization (L1, L2, dropout)</item>
    /// <item>Use data augmentation</item>
    /// <item>Apply early stopping</item>
    /// </list>
    /// </remarks>
    HighVariance = 1,

    /// <summary>
    /// Model has a good bias-variance balance: Good performance on both training and test data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model has found the sweet spot - it captures the
    /// underlying patterns without memorizing noise. Both training and test errors are low.</para>
    /// <para><b>Characteristics:</b></para>
    /// <list type="bullet">
    /// <item>Low training error</item>
    /// <item>Low test error (close to training error)</item>
    /// <item>Small gap between training and test curves</item>
    /// </list>
    /// </remarks>
    GoodFit = 2,

    /// <summary>
    /// Unable to determine diagnosis: Insufficient data or ambiguous results.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sometimes the data doesn't clearly indicate whether
    /// the problem is bias or variance. This might happen with very small datasets
    /// or unusual error patterns.</para>
    /// </remarks>
    Undetermined = 3,

    /// <summary>
    /// Both high bias and high variance: Rare case indicating fundamental model issues.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This unusual situation suggests the model is both
    /// too simple in some ways and overfitting in others. May indicate feature engineering
    /// issues or data quality problems.</para>
    /// <para><b>Solutions:</b></para>
    /// <list type="bullet">
    /// <item>Review feature engineering</item>
    /// <item>Check for data quality issues</item>
    /// <item>Consider different model architectures</item>
    /// </list>
    /// </remarks>
    HighBiasHighVariance = 4,

    /// <summary>
    /// Model is severely underfitting: Training error is extremely high.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model fails to learn even the training data.
    /// This is worse than regular high bias - the model may be completely inappropriate
    /// for the task.</para>
    /// </remarks>
    SevereUnderfit = 5,

    /// <summary>
    /// Model is severely overfitting: Perfect training but random test performance.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model has completely memorized the training data
    /// (near-zero training error) but performs almost randomly on test data. Often seen
    /// with very complex models on small datasets.</para>
    /// </remarks>
    SevereOverfit = 6,

    /// <summary>
    /// Model performance is still improving with more data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning curve shows that validation performance
    /// is still improving as more training data is added. Getting more data would likely
    /// improve the model further.</para>
    /// </remarks>
    NeedsMoreData = 7,

    /// <summary>
    /// Unable to determine diagnosis: Alias for Undetermined.
    /// </summary>
    Unknown = Undetermined
}
