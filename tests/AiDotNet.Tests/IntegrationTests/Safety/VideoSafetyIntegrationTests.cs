#nullable disable
using AiDotNet.Safety.Video;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for video safety modules.
/// Tests FrameSamplingVideoModerator, TemporalConsistencyDetector, and MultimodalVideoModerator
/// with frame sequences containing motion, static frames, and varying frame counts.
/// </summary>
public class VideoSafetyIntegrationTests
{
    #region FrameSamplingVideoModerator Tests

    [Fact]
    public void FrameSampling_StandardFrames_ProcessesWithoutError()
    {
        var moderator = new FrameSamplingVideoModerator<double>(samplingRate: 1.0);
        var frames = GenerateTestFrames(30, 8, 8);
        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void FrameSampling_FewFrames_HandlesGracefully()
    {
        var moderator = new FrameSamplingVideoModerator<double>(samplingRate: 1.0);
        var frames = GenerateTestFrames(3, 8, 8);
        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void FrameSampling_LargerFrames_ProcessesWithoutError()
    {
        var moderator = new FrameSamplingVideoModerator<double>(samplingRate: 0.5);
        var frames = GenerateTestFrames(20, 16, 16);
        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void FrameSampling_SingleFrame_HandlesGracefully()
    {
        var moderator = new FrameSamplingVideoModerator<double>(samplingRate: 1.0);
        var frames = GenerateTestFrames(1, 8, 8);
        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    #endregion

    #region TemporalConsistencyDetector Tests

    [Fact]
    public void Temporal_MotionSequence_ProcessesWithoutError()
    {
        var detector = new TemporalConsistencyDetector<double>();
        var frames = GenerateTestFrames(10, 8, 8);
        var findings = detector.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Temporal_StaticFrames_ProcessesWithoutError()
    {
        var detector = new TemporalConsistencyDetector<double>();
        var frames = GenerateStaticFrames(10, 8, 8);
        var findings = detector.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Temporal_ManyFrames_ProcessesWithoutError()
    {
        var detector = new TemporalConsistencyDetector<double>();
        var frames = GenerateTestFrames(60, 8, 8);
        var findings = detector.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Temporal_TwoFrames_HandlesGracefully()
    {
        var detector = new TemporalConsistencyDetector<double>();
        var frames = GenerateTestFrames(2, 8, 8);
        var findings = detector.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    #endregion

    #region MultimodalVideoModerator Tests

    [Fact]
    public void Multimodal_StandardFrames_ProcessesWithoutError()
    {
        var moderator = new MultimodalVideoModerator<double>();
        var frames = GenerateTestFrames(10, 8, 8);
        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Multimodal_LargerFrames_ProcessesWithoutError()
    {
        var moderator = new MultimodalVideoModerator<double>();
        var frames = GenerateTestFrames(15, 16, 16);
        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Multimodal_DifferentFrameRates_Work()
    {
        var moderator = new MultimodalVideoModerator<double>();
        var frames = GenerateTestFrames(10, 8, 8);

        foreach (var fps in new[] { 24.0, 30.0, 60.0 })
        {
            var findings = moderator.EvaluateVideo(frames, fps);
            Assert.NotNull(findings);
        }
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllModerators_SameFrames_ProduceResults()
    {
        var frames = GenerateTestFrames(10, 8, 8);

        Assert.NotNull(new FrameSamplingVideoModerator<double>(samplingRate: 1.0)
            .EvaluateVideo(frames, 30.0));
        Assert.NotNull(new TemporalConsistencyDetector<double>()
            .EvaluateVideo(frames, 30.0));
        Assert.NotNull(new MultimodalVideoModerator<double>()
            .EvaluateVideo(frames, 30.0));
    }

    #endregion

    #region Helpers

    private static IReadOnlyList<Tensor<double>> GenerateTestFrames(
        int numFrames, int height, int width)
    {
        var frames = new List<Tensor<double>>();
        var rng = new Random(42);

        for (int f = 0; f < numFrames; f++)
        {
            var data = new double[3 * height * width];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (rng.NextDouble() * 0.1 + f * 0.01) * 255;
            }

            frames.Add(new Tensor<double>(data, new[] { 3, height, width }));
        }

        return frames;
    }

    private static IReadOnlyList<Tensor<double>> GenerateStaticFrames(
        int numFrames, int height, int width)
    {
        var frames = new List<Tensor<double>>();
        var data = new double[3 * height * width];
        var rng = new Random(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextDouble() * 255;
        }

        for (int f = 0; f < numFrames; f++)
        {
            frames.Add(new Tensor<double>((double[])data.Clone(), new[] { 3, height, width }));
        }

        return frames;
    }

    #endregion
}
