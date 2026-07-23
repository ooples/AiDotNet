namespace AiDotNet.Data.Video.Sampling;

/// <summary>
/// Temporal Segment Networks (TSN) sampling: divides video into equal segments and samples one frame per segment.
/// </summary>
public class TemporalSegmentSampler : IVideoFrameSampler
{
    private readonly Random _random;

    public TemporalSegmentSampler(int? seed = null)
    {
        _random = seed.HasValue ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value) : Tensors.Helpers.RandomHelper.CreateSecureRandom();
    }

    public int[] SampleFrameIndices(int totalFrames, int numFramesToSample)
    {
        if (totalFrames <= 0 || numFramesToSample <= 0)
            return Array.Empty<int>();

        var indices = new int[numFramesToSample];
        double segmentLength = (double)totalFrames / numFramesToSample;

        for (int i = 0; i < numFramesToSample; i++)
        {
            int segStart = (int)(i * segmentLength);
            int segEnd = (int)((i + 1) * segmentLength);
            segEnd = Math.Min(segEnd, totalFrames);
            indices[i] = segStart + _random.Next(Math.Max(1, segEnd - segStart));
        }

        return indices;
    }
}
