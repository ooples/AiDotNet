using AiDotNet.Interfaces;

namespace AiDotNet.Models;

/// <summary>
/// Represents a machine learning experiment that groups related training runs.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> An experiment is a container for organizing related training runs.
/// It helps you group all attempts at solving a particular ML problem together.
/// </remarks>
public class Experiment : IExperiment
{
    /// <summary>
    /// Gets the unique identifier for this experiment.
    /// </summary>
    public string ExperimentId { get; private set; }

    /// <summary>
    /// Gets or sets the name of the experiment.
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// Gets or sets the description of the experiment.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets the timestamp when the experiment was created.
    /// </summary>
    public DateTime CreatedAt { get; private set; }

    /// <summary>
    /// Gets the timestamp of the last update to the experiment.
    /// </summary>
    public DateTime LastUpdatedAt { get; private set; }

    /// <summary>
    /// Gets or sets tags associated with the experiment.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; }

    /// <summary>
    /// Gets the current status of the experiment.
    /// </summary>
    public string Status { get; private set; }

    /// <summary>
    /// Initializes a new instance of the Experiment class.
    /// </summary>
    /// <param name="name">The name of the experiment.</param>
    /// <param name="description">Optional description.</param>
    /// <param name="tags">Optional tags.</param>
    public Experiment(string name, string? description = null, Dictionary<string, string>? tags = null)
    {
        ExperimentId = Guid.NewGuid().ToString();
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Description = description;
        CreatedAt = DateTime.UtcNow;
        LastUpdatedAt = DateTime.UtcNow;
        Tags = tags ?? new Dictionary<string, string>();
        Status = "Active";
    }

    /// <summary>
    /// Archives this experiment, making it read-only.
    /// </summary>
    public void Archive()
    {
        Status = "Archived";
        LastUpdatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Restores this experiment from archived status.
    /// </summary>
    public void Restore()
    {
        Status = "Active";
        LastUpdatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Updates the last updated timestamp.
    /// </summary>
    internal void Touch()
    {
        LastUpdatedAt = DateTime.UtcNow;
    }
}
