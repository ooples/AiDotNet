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

    public static Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        for (int i = 0; i < x.Length; i++)
        {
            double v = Ops.ToDouble(x[i]);
            result[i] = Ops.FromDouble(Math.Max(0, v));
        }
        return result;
    }

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
    /// Validates shape match — used by ResNet/CSPDarknet/SwinTransformer skip paths.
    /// </summary>
    public static Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length)
            throw new InvalidOperationException(
                $"BackboneOps.AddResidual length mismatch: {a.Length} vs {b.Length}.");
        var result = new Tensor<T>(a._shape);
        for (int i = 0; i < a.Length; i++)
            result[i] = Ops.Add(a[i], b[i]);
        return result;
    }
}
