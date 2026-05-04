using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Shared CPU-side tensor primitives reused by every detection backbone
/// (ResNet stem ReLU + MaxPool, EfficientNet swish, etc.). Replaces the
/// duplicated nested loops that lived in each backbone before
/// <c>BackboneBase</c> was deleted.
/// </summary>
internal static class BackboneOps<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    public static Tensor<T> MaxPool2D(Tensor<T> x, int kernelSize, int stride, int padding)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];
        int outH = (height + 2 * padding - kernelSize) / stride + 1;
        int outW = (width + 2 * padding - kernelSize) / stride + 1;
        var output = new Tensor<T>(new[] { batch, channels, outH, outW });

        for (int n = 0; n < batch; n++)
        for (int c = 0; c < channels; c++)
        for (int oh = 0; oh < outH; oh++)
        for (int ow = 0; ow < outW; ow++)
        {
            double maxVal = double.NegativeInfinity;
            for (int kh = 0; kh < kernelSize; kh++)
            for (int kw = 0; kw < kernelSize; kw++)
            {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                {
                    double v = Ops.ToDouble(x[n, c, ih, iw]);
                    if (v > maxVal) maxVal = v;
                }
            }
            output[n, c, oh, ow] = Ops.FromDouble(maxVal == double.NegativeInfinity ? 0 : maxVal);
        }
        return output;
    }

    /// <summary>
    /// Element-wise residual addition (a + b in-place into a fresh tensor of a's shape).
    /// Validates BOTH length and rank-by-rank shape so a same-element-count but
    /// different-rank mismatch (e.g. [1,16,8,8] vs [16,8,1,8]) is caught instead
    /// of silently producing semantically-wrong activations.
    /// </summary>
    public static Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length || a._shape.Length != b._shape.Length)
            throw new InvalidOperationException(
                $"BackboneOps.AddResidual shape mismatch: [{string.Join(",", a._shape)}] vs [{string.Join(",", b._shape)}].");
        for (int axis = 0; axis < a._shape.Length; axis++)
        {
            if (a._shape[axis] != b._shape[axis])
                throw new InvalidOperationException(
                    $"BackboneOps.AddResidual shape mismatch at axis {axis}: " +
                    $"[{string.Join(",", a._shape)}] vs [{string.Join(",", b._shape)}].");
        }
        var result = new Tensor<T>(a._shape);
        for (int i = 0; i < a.Length; i++)
            result[i] = Ops.Add(a[i], b[i]);
        return result;
    }

    // ApplyReLU / ApplySiLU / ApplySwish removed — backbones (ResNet, CSPDarknet,
    // EfficientNet, SwinTransformer) now accept a configurable IActivationFunction<T>?
    // ctor parameter that resolves to the paper-correct default when null.
}
