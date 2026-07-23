namespace AiDotNet.Interfaces;

/// <summary>
/// Defines capability to reset iteration state back to the beginning.
/// </summary>
/// <remarks>
/// <para>
/// Data loaders that implement this interface can be reset to their initial state,
/// allowing iteration to start over from the beginning of the dataset.
/// </para>
/// <para><b>For Beginners:</b> Think of this like rewinding a video back to the start.
/// After processing all your data once (one epoch), you can reset and go through it again
/// for another epoch of training.
/// </para>
/// </remarks>
public interface IResettable
{
    /// <summary>
    /// Resets the iteration state to the beginning of the data.
    /// </summary>
    /// <remarks>
    /// After calling Reset(), the next call to GetNextBatch() will return the first batch
    /// from the beginning of the dataset.
    /// </remarks>
    void Reset();
}
