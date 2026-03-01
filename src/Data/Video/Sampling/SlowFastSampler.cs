namespace AiDotNet.Data.Video.Sampling;

/// <summary>
/// SlowFast sampling: produces two sets of frame indices at different temporal resolutions
/// for SlowFast networks (Feichtenhofer et al., 2019).
/// </summary>
public class SlowFastSampler : IVideoFrameSampler
{
    private readonly int _alpha;

    /// <summary>
    /// Creates a new SlowFast sampler.
    /// </summary>
    /// <param name="alpha">Temporal stride ratio between slow and fast pathways. Default is 8.</param>
    public SlowFastSampler(int alpha = 8)
    {
        if (alpha <= 0) throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be positive.");
        _alpha = alpha;
    }

    /// <summary>
    /// Samples frames for the slow pathway. The fast pathway uses alpha times more frames.
    /// </summary>
    public int[] SampleFrameIndices(int totalFrames, int numFramesToSample)
    {
        if (totalFrames <= 0 || numFramesToSample <= 0)
            return Array.Empty<int>();

        // Slow pathway: sample numFramesToSample frames uniformly
        int slowCount = numFramesToSample;
        int fastCount = numFramesToSample * _alpha;

        // Total indices: slow + fast pathway frames
        var indices = new int[slowCount + fastCount];

        // Fill slow pathway
        if (slowCount == 1)
        {
            indices[0] = totalFrames / 2;
        }
        else
        {
            double slowStep = (double)(totalFrames - 1) / (slowCount - 1);
            for (int i = 0; i < slowCount; i++)
                indices[i] = Math.Min((int)(i * slowStep), totalFrames - 1);
        }

        // Fill fast pathway
        if (fastCount == 1)
        {
            indices[slowCount] = totalFrames / 2;
        }
        else
        {
            double fastStep = (double)(totalFrames - 1) / (fastCount - 1);
            for (int i = 0; i < fastCount; i++)
                indices[slowCount + i] = Math.Min((int)(i * fastStep), totalFrames - 1);
        }

        return indices;
    }
}
