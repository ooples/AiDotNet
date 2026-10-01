using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Safety.Video;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for video safety modules, judged on clips through <c>EvaluateVideo</c> (#2139).
/// </summary>
/// <remarks>
/// <para>
/// The generic safety fixture evaluated a single content vector, and the video bridge wraps a vector as one
/// frame. A module that judges motion between frames could therefore never produce a finding there, and its
/// confidence invariant failed on an empty result. This base gives every video module a real clip:
/// <see cref="FrameCount"/> frames of [3, <see cref="FrameSize"/>, <see cref="FrameSize"/>] pixels, the image
/// layout the moderators' per-frame classifiers read.
/// </para>
/// <para>
/// The vector bridge is held to its contract. A module that one frame suffices for answers through it; a
/// module that needs more (<c>MinimumFrames</c> &gt; 1) refuses with <see cref="NotSupportedException"/>
/// instead of returning no findings, which a caller would read as "safe".
/// </para>
/// </remarks>
public abstract class VideoSafetyModuleTestBase
{
    /// <summary>Subclasses return their concrete module.</summary>
    protected abstract IVideoSafetyModule<double> CreateModule();

    /// <summary>Frames per test clip.</summary>
    protected virtual int FrameCount => 8;

    /// <summary>Height and width of each frame.</summary>
    protected virtual int FrameSize => 32;

    /// <summary>Frames per second of the test clip.</summary>
    protected virtual double FrameRate => 30.0;

    /// <summary>A seeded clip whose frames change abruptly, so a temporal module has motion to judge.</summary>
    private List<Tensor<double>> CreateClip(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var frames = new List<Tensor<double>>(FrameCount);
        for (int f = 0; f < FrameCount; f++)
        {
            var frame = new Tensor<double>(new[] { 3, FrameSize, FrameSize });
            for (int i = 0; i < frame.Length; i++) frame[i] = rng.NextDouble();
            frames.Add(frame);
        }

        return frames;
    }

    private static void AssertConfidencesInUnitInterval(IReadOnlyList<SafetyFinding> findings)
    {
        foreach (var finding in findings)
        {
            Assert.True(finding.Confidence >= 0.0 && finding.Confidence <= 1.0,
                $"Confidence {finding.Confidence} for {finding.Category} is outside [0, 1]; it is a probability.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateVideo_ReturnsFindingsWithUnitIntervalConfidences()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var module = CreateModule();
        Assert.True(module.IsReady, $"Module '{module.ModuleName}' reports IsReady=false.");

        var findings = module.EvaluateVideo(CreateClip(seed: 1), FrameRate);

        Assert.NotNull(findings);
        AssertConfidencesInUnitInterval(findings);
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateVideo_IsDeterministic()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var module = CreateModule();

        var first = module.EvaluateVideo(CreateClip(seed: 2), FrameRate);
        var second = module.EvaluateVideo(CreateClip(seed: 2), FrameRate);

        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.Equal(first[i].Category, second[i].Category);
            Assert.Equal(first[i].Confidence, second[i].Confidence);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Evaluate_OnAVector_AnswersOrRefusesPerMinimumFrames()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var module = CreateModule();
        var single = CreateClip(seed: 3)[0];
        var content = new Vector<double>(single.ToArray());

        if (module is VideoSafetyModuleBase<double> { MinimumFrames: > 1 })
        {
            Assert.Throws<NotSupportedException>(() => module.Evaluate(content));
            return;
        }

        var findings = module.Evaluate(content);
        Assert.NotNull(findings);
        AssertConfidencesInUnitInterval(findings);
    }
}
