namespace AiDotNet.Data.Video.Sampling;

/// <summary>
/// Dense sampling: uniformly samples frames at regular intervals from the video.
/// </summary>
public class DenseSampler : IVideoFrameSampler
{
    public int[] SampleFrameIndices(int totalFrames, int numFramesToSample)
    {
        if (totalFrames <= 0 || numFramesToSample <= 0)
            return Array.Empty<int>();

        var indices = new int[numFramesToSample];

        if (numFramesToSample == 1)
        {
            indices[0] = totalFrames / 2; // Sample the middle frame
            return indices;
        }

        double step = (double)(totalFrames - 1) / (numFramesToSample - 1);

        for (int i = 0; i < numFramesToSample; i++)
        {
            indices[i] = Math.Min((int)(i * step), totalFrames - 1);
        }

        return indices;
    }
}
