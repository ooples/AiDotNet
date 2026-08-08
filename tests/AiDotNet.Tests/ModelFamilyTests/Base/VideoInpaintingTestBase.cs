using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Video;
using Xunit;
using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video inpainting models. Inherits video NN invariants
/// and adds inpainting-specific: output same size as input and bounded values.
/// </summary>
/// <typeparam name="T">
/// Numeric precision of the generated fixture. Generic so a training-bound inpainting model can be
/// routed to <c>&lt;float&gt;</c> (the scaffold's Fp32 selection) instead of being deferred out of the
/// shard; the non-generic <see cref="VideoInpaintingTestBase"/> alias below keeps every existing
/// <c>&lt;double&gt;</c> fixture source-compatible.
/// </typeparam>
public abstract class VideoInpaintingTestBase<T> : VideoNNModelTestBase<T>
{
    [Fact(Timeout = 120000)]
    public async Task InpaintedOutput_SameSizeAsInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.Equal(input.Length, output.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task InpaintedValues_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double o = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(o), $"Inpainted output[{i}] is NaN.");
            Assert.True(Math.Abs(o) < 1e6,
                $"Inpainted output[{i}] = {o:E4} is unbounded.");
        }
    }

    /// <summary>
    /// Inpainting-specific invariant that goes beyond the generic cross-family suite: it verifies the
    /// concatenated single-channel mask actually <b>conditions</b> the output. Two different hole masks
    /// over the SAME frames must produce different fills; if they don't, the mask channel is a dead input
    /// (the exact degenerate an all-zero training mask hides — the mask flows no gradient and inference
    /// ignores where the hole is). None of the generic invariants exercise the mask path, so this is the
    /// only guard that the model is genuinely mask-conditioned.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Inpainting_MaskShouldConditionOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        // A MISSING API IS A FAILURE, NOT A PASS. The early `return` marked this test green
        // whenever CreateNetwork() handed back anything other than a VideoInpaintingBase<T>,
        // so a fixture wired to the wrong type -- or a model that lost the mask-conditioned
        // path in a refactor -- silently received NO mask validation at all. This is the only
        // guard that the mask channel is not a dead input, and it was the one that quietly
        // stepped aside exactly when the fixture was broken.
        // NULL-SAFE MESSAGE. C# evaluates the message argument BEFORE calling Assert.True, so
        // network.GetType() threw NullReferenceException when CreateNetwork() returned null -- and a
        // null return is precisely the broken-fixture case this assertion was written to diagnose.
        // The fixture saw a NullReferenceException instead of the explanation.
        Assert.True(network is VideoInpaintingBase<T>,
            $"{network?.GetType().Name ?? "A null network"} does not expose " +
            "VideoInpaintingBase<T>.Inpaint, so mask " +
            "conditioning cannot be verified. An inpainting fixture must expose the " +
            "mask-conditioned path; fix the fixture rather than skipping the invariant.");
        var inpainter = (VideoInpaintingBase<T>)network;

        var frames = CreateRandomTensor(InputShape, rng);
        int n = frames.Shape[0];
        int h = frames.Shape[2];
        int w = frames.Shape[3];

        // Two clearly distinct masks over the same frames: a top-left-quadrant hole vs a full-frame hole.
        // (Distinct for any realistic frame; a partial box can never equal an all-ones mask.)
        var boxMask = BuildBoxMask(n, h, w, 0, 0, Math.Max(1, h / 2), Math.Max(1, w / 2));
        var fullMask = BuildBoxMask(n, h, w, 0, 0, h, w);

        var boxOut = inpainter.Inpaint(frames, boxMask);
        var fullOut = inpainter.Inpaint(frames, fullMask);

        double sumSquared = 0;
        int len = Math.Min(boxOut.Length, fullOut.Length);
        for (int i = 0; i < len; i++)
        {
            double d = ConvertToDouble(boxOut[i]) - ConvertToDouble(fullOut[i]);
            sumSquared += d * d;
        }
        double l2 = Math.Sqrt(sumSquared);

        Assert.True(l2 > 1e-9,
            $"Inpainting produced identical output for two distinct hole masks (L2={l2:E4}). "
            + "The concatenated mask channel is not conditioning the network — it is a dead input.");
    }

    /// <summary>Builds a single-channel <c>[n, 1, h, w]</c> mask with a rectangular hole (1 = hole).</summary>
    private Tensor<T> BuildBoxMask(int n, int h, int w, int top, int left, int boxH, int boxW)
    {
        var mask = new Tensor<T>([n, 1, h, w]);
        var one = NumOps.FromDouble(1.0);
        var span = mask.Data.Span;
        int plane = h * w;
        for (int b = 0; b < n; b++)
        {
            int baseOffset = b * plane;
            for (int y = top; y < top + boxH && y < h; y++)
            {
                int rowOffset = baseOffset + y * w;
                for (int x = left; x < left + boxW && x < w; x++)
                    span[rowOffset + x] = one;
            }
        }
        return mask;
    }
}

/// <summary>
/// Double-precision alias so existing generated inpainting fixtures keep deriving from a
/// non-generic base; <c>&lt;float&gt;</c> fixtures derive from <see cref="VideoInpaintingTestBase{T}"/>
/// directly.
/// </summary>
public abstract class VideoInpaintingTestBase : VideoInpaintingTestBase<double> { }
