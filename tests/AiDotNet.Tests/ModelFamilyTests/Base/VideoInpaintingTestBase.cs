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
public abstract class VideoInpaintingTestBase : VideoNNModelTestBase
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
            Assert.False(double.IsNaN(output[i]), $"Inpainted output[{i}] is NaN.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Inpainted output[{i}] = {output[i]:E4} is unbounded.");
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

        // Only models that expose the mask-conditioned Inpaint path are in scope.
        if (network is not VideoInpaintingBase<T> inpainter) return;

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
