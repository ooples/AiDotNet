namespace AiDotNet.Data.Video.Sampling;

/// <summary>
/// Interface for video frame sampling strategies.
/// </summary>
public interface IVideoFrameSampler
{
    /// <summary>
    /// Selects frame indices from a video with the given total frame count.
    /// </summary>
    /// <param name="totalFrames">Total number of frames in the video.</param>
    /// <param name="numFramesToSample">Number of frames to select.</param>
    /// <returns>Array of selected frame indices.</returns>
    int[] SampleFrameIndices(int totalFrames, int numFramesToSample);
}
