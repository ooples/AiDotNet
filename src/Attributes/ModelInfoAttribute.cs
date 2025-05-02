namespace AiDotNet.Attributes;

/// <summary>
/// Provides metadata about machine learning models in the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This attribute is used to attach important information to each model type in the library.
/// It specifies what category the model belongs to (like Regression or Classification), which metrics can be used
/// to evaluate its performance, and provides a description of the model. This information helps the library
/// automatically configure appropriate settings for each model type and provides useful documentation to users.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Field)]
public class ModelInfoAttribute : Attribute
{
    /// <summary>
    /// Gets the category that the model belongs to (e.g., Regression, Classification, TimeSeries).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property indicates what type of machine learning task the model is designed for.
    /// For example, a Linear Regression model would have the Regression category, while a Decision Tree might be
    /// categorized as Classification or Regression depending on its configuration.
    /// </para>
    /// </remarks>
    public ModelCategory Category { get; }

    /// <summary>
    /// Gets the metrics that can be used to evaluate this model's performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different models need different ways to measure how well they're performing.
    /// For example, classification models might use accuracy or F1 score, while regression models might use
    /// mean squared error. This property lists all the valid metrics that make sense for evaluating this
    /// particular model type.
    /// </para>
    /// </remarks>
    public MetricGroups[] ValidMetrics { get; }

    /// <summary>
    /// Gets a human-readable description of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property provides a brief explanation of what the model does, how it works,
    /// and what kinds of problems it's good at solving. This helps users understand the model's purpose and
    /// decide whether it's appropriate for their specific task.
    /// </para>
    /// </remarks>
    public string Description { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelInfoAttribute"/> class.
    /// </summary>
    /// <param name="category">The category that the model belongs to.</param>
    /// <param name="validMetrics">The metrics that can be used to evaluate this model's performance.</param>
    /// <param name="description">A human-readable description of the model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new ModelInfoAttribute with the specified category,
    /// valid metrics, and description. It's used when defining the attributes for each model type in the library.
    /// </para>
    /// </remarks>
    public ModelInfoAttribute(ModelCategory category, MetricGroups[] validMetrics, string description)
    {
        Category = category;
        ValidMetrics = validMetrics;
        Description = description;
    }
}