namespace AiDotNet.Interfaces;

/// <summary>
/// Defines capability to report dataset size and iteration progress.
/// </summary>
/// <remarks>
/// <para>
/// Data loaders that implement this interface provide information about
/// their size and current position, useful for progress tracking and
/// determining when an epoch is complete.
/// </para>
/// <para><b>For Beginners:</b> Knowing how much data you have and where you are
/// in processing is essential for:
/// - Showing progress bars
/// - Knowing when an epoch (full pass through data) is complete
/// - Calculating metrics like "samples processed per second"
/// </para>
/// </remarks>
public interface ICountable
{
    /// <summary>
    /// Gets the total number of samples in the dataset.
    /// </summary>
    int TotalCount { get; }

    /// <summary>
    /// Gets the current sample index in the iteration (0-based).
    /// </summary>
    int CurrentIndex { get; }

    /// <summary>
    /// Gets the total number of batches based on current batch size.
    /// </summary>
    /// <remarks>
    /// This is typically ceil(TotalCount / BatchSize).
    /// </remarks>
    int BatchCount { get; }

    /// <summary>
    /// Gets the current batch index in the iteration (0-based).
    /// </summary>
    int CurrentBatchIndex { get; }

    /// <summary>
    /// Gets the progress through the current epoch as a value from 0.0 to 1.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns CurrentIndex / TotalCount, useful for progress displays.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you a percentage of how far through
    /// the data you are:
    /// - 0.0 = just started
    /// - 0.5 = halfway through
    /// - 1.0 = finished (time to reset for next epoch)
    /// </para>
    /// </remarks>
    double Progress { get; }
}
