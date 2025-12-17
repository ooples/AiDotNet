namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the fitness calculator, which determines how model performance is evaluated.
/// </summary>
/// <remarks>
/// <para>
/// The fitness calculator is responsible for computing a score that represents how well a model fits the data
/// or makes predictions. Different metrics emphasize different aspects of model performance, such as overall
/// fit, error magnitude, or prediction accuracy.
/// </para>
/// <para><b>For Beginners:</b> Think of the fitness calculator as a judge that scores how well your AI model
/// is performing. Just like different sports have different scoring systems (points in basketball, goals in
/// soccer), AI models can be evaluated using different metrics. Some metrics focus on how close your predictions
/// are to the actual values, while others might focus on whether your model captures the overall patterns in
/// the data. These options let you choose which scoring system to use and how to interpret the scores.</para>
/// </remarks>
public class FitnessCalculatorOptions
{
    /// <summary>
    /// Gets or sets the type of metric used to calculate the fitness score.
    /// </summary>
    /// <value>The score type, defaulting to R-squared.</value>
    /// <remarks>
    /// <para>
    /// Different metrics evaluate different aspects of model performance:
    /// <list type="bullet">
    ///   <item><description>R-squared measures the proportion of variance explained by the model (higher is better, max 1.0)</description></item>
    ///   <item><description>MeanSquaredError measures the average squared difference between predictions and actual values (lower is better)</description></item>
    ///   <item><description>MeanAbsoluteError measures the average absolute difference between predictions and actual values (lower is better)</description></item>
    ///   <item><description>RootMeanSquaredError is the square root of MeanSquaredError, which gives errors in the same units as the target variable (lower is better)</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> This determines which method is used to score your model's performance.
    /// The default, R-squared (also written as R²), measures how well your model explains the variations in your data.
    /// An R-squared of 1.0 means your model perfectly predicts every value, while 0.0 means it's no better than
    /// just guessing the average value every time. Other options include:
    /// <list type="bullet">
    ///   <item><description>Mean Squared Error: Measures the average of the squared differences between predictions and actual values. It penalizes larger errors more heavily.</description></item>
    ///   <item><description>Mean Absolute Error: Measures the average of the absolute differences between predictions and actual values. It treats all sizes of errors equally.</description></item>
    ///   <item><description>Root Mean Squared Error: The square root of Mean Squared Error, which gives you an error value in the same units as your original data.</description></item>
    /// </list>
    /// Choose R-squared when you want to understand how much of the data variation your model explains.
    /// Choose one of the error metrics when you want to focus on the size of prediction errors.</para>
    /// </remarks>
    public FitnessCalculatorType ScoreType { get; set; } = FitnessCalculatorType.RSquared;

    /// <summary>
    /// Gets or sets whether higher values indicate better fitness.
    /// </summary>
    /// <value>True if higher values are better, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// Some metrics like R-squared are better when higher (with a maximum of 1.0), while others like
    /// Mean Squared Error are better when lower (with a minimum of 0.0). This property determines how
    /// the scores should be interpreted when comparing models or evaluating performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the system whether a higher score means better performance
    /// or worse performance. With the default value of true, the system assumes that higher scores are better
    /// (which is correct for R-squared). If you switch to an error-based metric like Mean Squared Error,
    /// you should set this to false because lower errors mean better performance. Think of it like golf
    /// versus basketball scoring - in golf, lower scores are better, but in basketball, higher scores win.
    /// This setting helps the system know which direction is "better" when comparing models.</para>
    /// </remarks>
    public bool UseMaximumValue { get; set; } = true;
}
