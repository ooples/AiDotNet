namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Cook's Distance fit detector, which helps identify influential data points
/// and detect potential overfitting or underfitting in regression models.
/// </summary>
/// <remarks>
/// <para>
/// Cook's Distance is a statistical measure that identifies how much influence each data point has on a regression model.
/// Points with high Cook's Distance values have a disproportionate effect on the model's predictions and parameters.
/// This detector analyzes the distribution of Cook's Distance values across your dataset to identify potential
/// fitting problems.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're trying to draw a straight line through a set of points. Some points
/// might have a big effect on where you place that line - if you removed them, the line would move significantly.
/// These are called "influential points." Cook's Distance measures how influential each point is. This detector
/// looks at all your data points and checks if too many (or too few) are highly influential, which could indicate
/// problems with how well your model fits the data. If many points are influential, your model might be too simple
/// (underfitting). If very few points are influential, your model might be too complex (overfitting).</para>
/// </remarks>
public class CookDistanceFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for determining when a data point is considered influential.
    /// </summary>
    /// <value>The influential threshold, defaulting to 0.04 (4/100).</value>
    /// <remarks>
    /// <para>
    /// Cook's Distance values above this threshold indicate that a data point has significant influence on the model.
    /// The traditional rule of thumb is to use 4/n, where n is the sample size. The default value of 0.04 is
    /// appropriate for datasets with around 100 observations.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when a data point is considered "influential" enough
    /// to potentially affect your model's accuracy. The default value (0.04) works well for most datasets with
    /// around 100 data points. For larger datasets, you might want to use a smaller value (like 0.01 for 400 data points),
    /// and for smaller datasets, a larger value (like 0.1 for 40 data points). The general formula is 4 divided by
    /// the number of data points you have. Points with influence above this threshold might be outliers or important
    /// edge cases in your data.</para>
    /// </remarks>
    public double InfluentialThreshold { get; set; } = 4.0 / 100; // 4/n, where n is typically the sample size

    /// <summary>
    /// Gets or sets the threshold for the proportion of influential points that suggests overfitting.
    /// </summary>
    /// <value>The overfit threshold as a proportion, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// If the proportion of data points with Cook's Distance values above the InfluentialThreshold exceeds this value,
    /// the model may be overfitting the data. Overfitting occurs when a model learns the training data too well,
    /// including its noise and outliers, resulting in poor generalization to new data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect if your model is "overfitting" - meaning it's too
    /// complex and is essentially memorizing your training data rather than learning general patterns. The default
    /// value (0.1 or 10%) means that if more than 10% of your data points are highly influential, your model might
    /// be overfitting. This could happen if you're using too many features or a model that's too complex for your
    /// data. Overfitting models perform well on training data but poorly on new data they haven't seen before.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1; // 10% of points being influential suggests overfitting

    /// <summary>
    /// Gets or sets the threshold for the proportion of influential points that suggests underfitting.
    /// </summary>
    /// <value>The underfit threshold as a proportion, defaulting to 0.01 (1%).</value>
    /// <remarks>
    /// <para>
    /// If the proportion of data points with Cook's Distance values above the InfluentialThreshold is below this value,
    /// the model may be underfitting the data. Underfitting occurs when a model is too simple to capture the underlying
    /// patterns in the data, resulting in poor performance on both training and new data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps detect if your model is "underfitting" - meaning it's too
    /// simple and isn't capturing important patterns in your data. The default value (0.01 or 1%) means that if
    /// fewer than 1% of your data points are highly influential, your model might be underfitting. This could happen
    /// if you're using too few features or a model that's too simple. Underfitting models perform poorly on both
    /// training data and new data because they haven't learned the important relationships in the data.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.01; // 1% of points being influential suggests underfitting
}
