using AiDotNet.ComputerVision;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Non-overlapping window partitioning of a row-major <c>[h*w, C]</c> token grid, as the windowed vision
/// transformers do (SAM ViTDet, Swin, DaViT).
/// </summary>
/// <remarks>
/// Windows are ordered (windowRow, windowColumn) and tokens inside each window row-major. Positions past the
/// map, used to pad it to whole windows, read an appended zero row. The reference implementations pad the
/// normalised tokens with zeros and attend over the padding unmasked.
/// </remarks>
internal static class WindowPartition
{
    /// <summary>
    /// Gather indices into <c>[tokens; zero]</c> that produce the windowed order, plus the inverse: each real
    /// token's position in that order.
    /// </summary>
    public static (int[] Gather, int[] Inverse, int Windows) Indices(int h, int w, int windowSize)
    {
        int ws = windowSize;
        int windowsY = (h + ws - 1) / ws, windowsX = (w + ws - 1) / ws, n = h * w;
        var gather = new int[windowsY * windowsX * ws * ws];
        var inverse = new int[n];
        int at = 0;
        for (int wy = 0; wy < windowsY; wy++)
            for (int wx = 0; wx < windowsX; wx++)
                for (int iy = 0; iy < ws; iy++)
                    for (int ix = 0; ix < ws; ix++)
                    {
                        int y = (wy * ws) + iy, x = (wx * ws) + ix;
                        bool real = y < h && x < w;
                        gather[at] = real ? (y * w) + x : n;
                        if (real) inverse[(y * w) + x] = at;
                        at++;
                    }
        return (gather, inverse, windowsY * windowsX);
    }

    /// <summary>Tokens <c>[h*w, C]</c> rearranged into windows <c>[windows * ws^2, C]</c>.</summary>
    public static Tensor<T> Partition<T>(Tensor<T> tokens, int[] gather)
    {
        var engine = AiDotNetEngine.Current;
        var padded = engine.TensorConcatenate(new[] { tokens, new Tensor<T>(new[] { 1, tokens.Shape[1] }) }, 0);
        return CvTensorOps<T>.Select(padded, gather, 0);
    }

    /// <summary>Windowed rows back in row-major order, dropping the padding.</summary>
    public static Tensor<T> Merge<T>(Tensor<T> windowed, int[] inverse) => CvTensorOps<T>.Select(windowed, inverse, 0);
}
