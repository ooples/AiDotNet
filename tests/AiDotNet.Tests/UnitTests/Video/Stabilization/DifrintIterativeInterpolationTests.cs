using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Stabilization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.Stabilization;

/// <summary>
/// Verifies DIFRINT's actual mechanism — DEEP ITERATIVE FRAME INTERPOLATION — against the paper
/// (Choi and Kweon, ACM TOG 39(1) / SIGGRAPH Asia 2019, arXiv:1909.02641).
/// </summary>
/// <remarks>
/// <para>
/// Each test targets one of three gaps that were MEASURED in the previous implementation, not guessed
/// at. All three shared a property that made them invisible: the class constructed, ran, and returned
/// finite output of the right shape the whole time, so every shape/finiteness/construction test passed.
/// </para>
/// <list type="number">
/// <item>Stabilize() ran a single pass; the iteration count reached only the layer builder.</item>
/// <item>The synthesis input was [prev, CURRENT, next], letting the network copy the jitter through.</item>
/// <item>ComputeSmoothPath's result was assigned and never read.</item>
/// </list>
/// <para>
/// These assert on STRUCTURE (how many passes, how many channels reach the network, whether alignment
/// moves the source toward the target) rather than on trained output quality, because the mechanism is
/// a property of the wiring and is therefore checkable on an untrained model.
/// </para>
/// </remarks>
public class DifrintIterativeInterpolationTests
{
    private const int Channels = 3;

    /// <summary>A textured frame. Flat frames are degenerate for block matching — every candidate
    /// offset ties at SAD 0, so the search reports whichever offset it happened to try first.</summary>
    private static Tensor<double> Textured(int h, int w, int seed)
    {
        var t = new Tensor<double>(new[] { Channels, h, w });
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
        return t;
    }

    /// <summary>Copies <paramref name="src"/> with its content displaced by (dx, dy) on screen.</summary>
    private static Tensor<double> Shifted(Tensor<double> src, int dx, int dy)
    {
        int h = src.Shape[1], w = src.Shape[2];
        var t = new Tensor<double>(new[] { Channels, h, w });
        for (int c = 0; c < Channels; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int sy = y - dy, sx = x - dx;
                    double v = (sy >= 0 && sy < h && sx >= 0 && sx < w)
                        ? src[(c * h * w) + (sy * w) + sx]
                        : 0.0;
                    t[(c * h * w) + (y * w) + x] = v;
                }
        return t;
    }

    private static double SumAbsDiff(Tensor<double> a, Tensor<double> b)
    {
        double s = 0.0;
        for (int i = 0; i < a.Length; i++) s += Math.Abs(a[i] - b[i]);
        return s;
    }

    private static double MaxAbsDiff(Tensor<double> a, Tensor<double> b)
    {
        double d = 0.0;
        for (int i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i]));
        return d;
    }

    private static NeuralNetworkArchitecture<double> Arch(int h, int w, List<ILayer<double>>? layers = null) =>
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: h, inputWidth: w, inputDepth: Channels,
            outputSize: Channels,
            layers: layers);

    /// <summary>Applies exactly ONE interpolation pass over the sequence, as the reference.</summary>
    private static List<Tensor<double>> OnePass(DIFRINT<double> model, List<Tensor<double>> frames)
    {
        var pass = new List<Tensor<double>>(frames.Count);
        for (int i = 0; i < frames.Count; i++)
        {
            var prev = i > 0 ? frames[i - 1] : frames[i];
            var next = i < frames.Count - 1 ? frames[i + 1] : frames[i];
            pass.Add(model.StabilizeFrame(prev, frames[i], next));
        }
        return pass;
    }

    [Fact]
    public void TheIterationCountReachesTheSequencePass()
    {
        // The defect: _numIterations was passed to the layer builder, the metadata and serialization,
        // but Stabilize() looped over the frames exactly once regardless of its value. So the "Deep
        // Iterative" mechanism — the paper's whole contribution and the source of its name — was
        // absent at the level where the paper applies it.
        //
        // Compared on ONE model instance, so the two results share identical weights. Two separately
        // constructed models get different random inits, and that difference would swamp the thing
        // being measured.
        const int size = 32;
        var model = new DIFRINT<double>(Arch(size, size), numIterations: 3);

        var frames = new List<Tensor<double>>
        {
            Textured(size, size, 1), Textured(size, size, 2),
            Textured(size, size, 3), Textured(size, size, 4),
        };

        var onePass = OnePass(model, frames);
        var threePasses = model.Stabilize(frames);

        Assert.Equal(frames.Count, threePasses.Count);

        double moved = MaxAbsDiff(onePass[1], threePasses[1]);
        Assert.True(moved > 0.0,
            "Stabilize() with NumIterations=3 produced exactly the same frame as a single manual pass, "
            + "so the iteration count is not reaching the sequence pass and the method is running as a "
            + "one-shot interpolator.");
    }

    [Fact]
    public void OneIterationEqualsExactlyOnePass()
    {
        // Pins the other end of the semantics. Without this, "3 differs from 1" could also be satisfied
        // by an off-by-one that runs 4 passes, or by iterating something unrelated.
        const int size = 32;
        var model = new DIFRINT<double>(Arch(size, size), numIterations: 1);

        var frames = new List<Tensor<double>>
        {
            Textured(size, size, 11), Textured(size, size, 12), Textured(size, size, 13),
        };

        var expected = OnePass(model, frames);
        var actual = model.Stabilize(frames);

        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
            Assert.Equal(0.0, MaxAbsDiff(expected[i], actual[i]), 12);
    }

    [Fact]
    public void StabilizeReturnsAFreshListAndNeverTheCallersOwn()
    {
        // Zero iterations is the edge that would otherwise hand the caller its own list back, so a
        // later mutation of the result would silently rewrite the input.
        const int size = 32;
        var model = new DIFRINT<double>(Arch(size, size), numIterations: 0);
        var frames = new List<Tensor<double>> { Textured(size, size, 21), Textured(size, size, 22) };

        var result = model.Stabilize(frames);

        Assert.False(ReferenceEquals(frames, result));
        Assert.Equal(frames.Count, result.Count);
    }

    [Fact]
    public void TheSynthesisInputExcludesTheCurrentFrame()
    {
        // The defect that makes the method unable to do its job: with [prev, CURRENT, next] as the
        // network input, any reconstruction objective is satisfiable by copying the current frame
        // straight through — which reproduces exactly the jitter the method exists to remove.
        //
        // Measured through the resolved input depth of the first layer, which is what actually
        // determines how many frames reach the network. A single 1x1 convolution is used so the count
        // is fully determined by THIS test rather than by the default architecture: weights are
        // outputDepth * inputDepth * 1 * 1, plus outputDepth biases.
        const int size = 32;
        var probe = new ConvolutionalLayer<double>(outputDepth: Channels, kernelSize: 1);
        var model = new DIFRINT<double>(Arch(size, size, new List<ILayer<double>> { probe }));

        var prev = Textured(size, size, 31);
        var curr = Textured(size, size, 32);
        var next = Textured(size, size, 33);

        // Resolves the lazy input depth from the real stacked input.
        _ = model.StabilizeFrame(prev, curr, next);

        long twoFrames = (Channels * (2 * Channels)) + Channels;    // 3*6 + 3 = 21
        long threeFrames = (Channels * (3 * Channels)) + Channels;  // 3*9 + 3 = 30

        Assert.Equal(twoFrames, model.ParameterCount);
        Assert.NotEqual(threeFrames, model.ParameterCount);
    }

    [Fact]
    public void WarpingAlignsTheSourceOntoTheTarget()
    {
        // The sign convention is the part of this most likely to be silently wrong, and getting it
        // backwards DOUBLES the misalignment instead of removing it while still returning a
        // correctly-shaped, finite, plausible-looking frame. EstimateMotionBetweenFrames reports how
        // far content travels from source to target; FlowWarpHelper.Warp backward-samples
        // (out[y][x] = src[y+fy][x+fx]); so the flow must be the NEGATED displacement.
        //
        // 96x96 because block matching only samples rows/cols in [blockSize+searchRange,
        // extent-blockSize-searchRange) = [24, 72) here. At 32x32 that range is empty, the estimate
        // falls back to (0,0), and the warp becomes an identity that would pass any sign convention.
        const int size = 96;
        var model = new DIFRINT<double>(Arch(size, size));

        var target = Textured(size, size, 41);
        var source = Shifted(target, dx: 3, dy: 2);   // within the +/-8 search range

        double before = SumAbsDiff(source, target);
        var aligned = model.WarpTowardTarget(source, target);
        double after = SumAbsDiff(aligned, target);

        Assert.Equal(source.Shape.ToArray(), aligned.Shape.ToArray());
        Assert.True(after < before,
            $"Warping moved the source AWAY from the target (SAD {before:F3} -> {after:F3}). The flow "
            + "sign is inverted: the neighbours are being pushed further out of alignment, so the "
            + "interpolation is averaging misaligned frames.");
    }

    [Fact]
    public void AFrameIsNotWarpedIntoItself()
    {
        // The sequence ends duplicate the frame (frame 0 has no predecessor), and estimating motion
        // between a frame and itself then warping by the result can only inject error.
        const int size = 96;
        var model = new DIFRINT<double>(Arch(size, size));
        var frame = Textured(size, size, 51);

        Assert.True(ReferenceEquals(frame, model.WarpTowardTarget(frame, frame)));
    }
}
