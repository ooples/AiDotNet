using AiDotNet.Interfaces;
using Newtonsoft.Json;
using AiDotNet.Validation;

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
    private string _name = string.Empty;

    /// <summary>
    /// Gets the unique identifier for this experiment.
    /// </summary>
    [JsonProperty]
    public string ExperimentId { get; private set; }

    /// <summary>
    /// Gets or sets the name of the experiment.
    /// </summary>
    /// <exception cref="ArgumentNullException">Thrown when value is null.</exception>
    /// <exception cref="ArgumentException">Thrown when value is empty or whitespace.</exception>
    public string Name
    {
        get => _name;
        set
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value), "Experiment name cannot be null.");
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("Experiment name cannot be empty or whitespace.", nameof(value));
            _name = value;
        }
    }

    /// <summary>
    /// Gets or sets the description of the experiment.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets the timestamp when the experiment was created.
    /// </summary>
    [JsonProperty]
    public DateTime CreatedAt { get; private set; }

    /// <summary>
    /// Gets the timestamp of the last update to the experiment.
    /// </summary>
    [JsonProperty]
    public DateTime LastUpdatedAt { get; private set; }

    /// <summary>
    /// Gets or sets tags associated with the experiment.
    /// </summary>
    public Dictionary<string, string> Tags { get; set; }

    /// <summary>
    /// Gets the current status of the experiment.
    /// </summary>
    [JsonProperty]
    public string Status { get; private set; }

    /// <summary>
    /// Private constructor for JSON deserialization.
    /// </summary>
    [JsonConstructor]
    private Experiment()
    {
        ExperimentId = string.Empty;
        _name = string.Empty; // Use backing field to allow deserialization
        Tags = new Dictionary<string, string>();
        Status = "Active";
    }

    /// <summary>
    /// Initializes a new instance of the Experiment class.
    /// </summary>
    /// <param name="name">The name of the experiment.</param>
    /// <param name="description">Optional description.</param>
    /// <param name="tags">Optional tags.</param>
    public Experiment(string name, string? description = null, Dictionary<string, string>? tags = null)
    {
        ExperimentId = Guid.NewGuid().ToString();
        Guard.NotNull(name);
        Name = name;
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
