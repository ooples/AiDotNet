namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a machine learning experiment that groups related training runs.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> An experiment is a container for organizing related training runs.
/// Think of it like a folder that groups all the attempts you make at solving a particular
/// machine learning problem. For example, you might have an experiment called "Customer Churn Prediction"
/// that contains all your different attempts at building a churn prediction model.
/// </remarks>
public interface IExperiment
{
    /// <summary>
    /// Gets the unique identifier for this experiment.
    /// </summary>
    string ExperimentId { get; }

    /// <summary>
    /// Gets or sets the name of the experiment.
    /// </summary>
    string Name { get; set; }

    /// <summary>
    /// Gets or sets the description of the experiment.
    /// </summary>
    string? Description { get; set; }

    /// <summary>
    /// Gets the timestamp when the experiment was created.
    /// </summary>
    DateTime CreatedAt { get; }

    /// <summary>
    /// Gets the timestamp of the last update to the experiment.
    /// </summary>
    DateTime LastUpdatedAt { get; }

    /// <summary>
    /// Gets or sets tags associated with the experiment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tags are key-value pairs that help you organize and find experiments.
    /// For example, you might tag experiments with "team=data-science" or "priority=high".
    /// </remarks>
    Dictionary<string, string> Tags { get; set; }

    /// <summary>
    /// Gets the current status of the experiment.
    /// </summary>
    string Status { get; }

    /// <summary>
    /// Archives this experiment, making it read-only.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Archiving an experiment prevents accidental modifications
    /// while keeping the data available for reference.
    /// </remarks>
    void Archive();

    /// <summary>
    /// Restores this experiment from archived status.
    /// </summary>
    void Restore();
}
