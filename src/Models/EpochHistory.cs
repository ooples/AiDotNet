namespace AiDotNet.Models;


/// <summary>
/// Represents the training history for a single epoch or iteration.
/// </summary>
public class EpochHistory
{
    /// <summary>
    /// Gets or sets the epoch number.
    /// </summary>
    public int EpochNumber { get; set; }

    /// <summary>
    /// Gets or sets the loss value for this epoch.
    /// </summary>
    public double Loss { get; set; }

    /// <summary>
    /// Gets or sets the accuracy value for this epoch.
    /// </summary>
    public double Accuracy { get; set; }
}
