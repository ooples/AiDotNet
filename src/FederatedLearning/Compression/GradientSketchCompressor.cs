using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Count Sketch-based gradient compression for federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A Count Sketch is a compact data structure that approximately stores
/// a large vector by hashing its elements into a smaller table. Think of it as a lossy compression
/// where you can recover the most important elements (top-k) but lose small values to hash collisions.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Create a sketch table of size (depth x width), much smaller than the gradient.</description></item>
/// <item><description>For each gradient element, use multiple hash functions to map it to sketch positions.</description></item>
/// <item><description>Each hash also produces a random sign (+1 or -1) to prevent bias from collisions.</description></item>
/// <item><description>To recover: for each element, take the median of its hashed positions (robust to collisions).</description></item>
/// <item><description>Optionally recover only top-k elements for further compression.</description></item>
/// </list>
///
/// <para><b>Compression ratio:</b> depth * width / gradient_size. With depth=5 and width=gradient_size/100,
/// this achieves ~20x compression.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class GradientSketchCompressor<T> : FederatedLearningComponentBase<T>
{
    private readonly AdvancedCompressionOptions _options;
    private readonly int[] _hashSeeds;

    /// <summary>
    /// Initializes a new instance of <see cref="GradientSketchCompressor{T}"/>.
    /// </summary>
    /// <param name="options">Advanced compression configuration.</param>
    public GradientSketchCompressor(AdvancedCompressionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // Generate hash seeds
        var rng = RandomHelper.CreateSecureRandom();
        _hashSeeds = new int[_options.SketchDepth * 2]; // depth pairs of (position hash, sign hash)
        for (int i = 0; i < _hashSeeds.Length; i++)
        {
            _hashSeeds[i] = rng.Next();
        }
    }

    /// <summary>
    /// Compresses a gradient tensor into a Count Sketch.
    /// </summary>
    /// <param name="gradient">The gradient tensor to compress.</param>
    /// <returns>The sketch table and metadata for decompression.</returns>
    public (double[,] Sketch, int OriginalSize, int Width) Compress(Tensor<T> gradient)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        int originalSize = gradient.Shape[0];
        int depth = _options.SketchDepth;
        int width = _options.SketchWidth > 0
            ? _options.SketchWidth
            : Math.Max(100, originalSize / 20); // Auto: ~5% of original size

        var sketch = new double[depth, width];

        for (int i = 0; i < originalSize; i++)
        {
            double value = NumOps.ToDouble(gradient[i]);
            if (Math.Abs(value) < 1e-15) continue;

            for (int d = 0; d < depth; d++)
            {
                int pos = HashPosition(i, d, width);
                int sign = HashSign(i, d);
                sketch[d, pos] += sign * value;
            }
        }

        return (sketch, originalSize, width);
    }

    /// <summary>
    /// Decompresses a Count Sketch back to a gradient tensor using median estimation.
    /// </summary>
    /// <param name="sketch">The sketch table.</param>
    /// <param name="originalSize">Original gradient vector size.</param>
    /// <param name="width">Sketch width used during compression.</param>
    /// <returns>Reconstructed gradient tensor.</returns>
    public Tensor<T> Decompress(double[,] sketch, int originalSize, int width)
    {
        int depth = _options.SketchDepth;
        var result = new Tensor<T>(new[] { originalSize });
        var estimates = new double[depth];

        for (int i = 0; i < originalSize; i++)
        {
            for (int d = 0; d < depth; d++)
            {
                int pos = HashPosition(i, d, width);
                int sign = HashSign(i, d);
                estimates[d] = sign * sketch[d, pos];
            }

            // Use median for robustness against hash collisions
            double value = Median(estimates, depth);
            result[i] = NumOps.FromDouble(value);
        }

        return result;
    }

    /// <summary>
    /// Decompresses only the top-k largest elements from the sketch.
    /// </summary>
    /// <param name="sketch">The sketch table.</param>
    /// <param name="originalSize">Original gradient vector size.</param>
    /// <param name="width">Sketch width used during compression.</param>
    /// <param name="topK">Number of top elements to recover. 0 = recover all.</param>
    /// <returns>Sparse gradient tensor with only top-k elements non-zero.</returns>
    public Tensor<T> DecompressTopK(double[,] sketch, int originalSize, int width, int topK)
    {
        if (topK <= 0 || topK >= originalSize)
        {
            return Decompress(sketch, originalSize, width);
        }

        int depth = _options.SketchDepth;
        var result = new Tensor<T>(new[] { originalSize });
        var allEstimates = new (int Index, double Value)[originalSize];
        var estimates = new double[depth];

        for (int i = 0; i < originalSize; i++)
        {
            for (int d = 0; d < depth; d++)
            {
                int pos = HashPosition(i, d, width);
                int sign = HashSign(i, d);
                estimates[d] = sign * sketch[d, pos];
            }

            allEstimates[i] = (i, Median(estimates, depth));
        }

        // Sort by absolute value descending, take top-k
        Array.Sort(allEstimates, (a, b) => Math.Abs(b.Value).CompareTo(Math.Abs(a.Value)));

        for (int k = 0; k < topK; k++)
        {
            result[allEstimates[k].Index] = NumOps.FromDouble(allEstimates[k].Value);
        }

        return result;
    }

    /// <summary>
    /// Gets the compression ratio achieved (sketch size / original size).
    /// </summary>
    public double GetCompressionRatio(int originalSize)
    {
        int width = _options.SketchWidth > 0
            ? _options.SketchWidth
            : Math.Max(100, originalSize / 20);
        double sketchSize = _options.SketchDepth * width;
        return sketchSize / originalSize;
    }

    private int HashPosition(int element, int depth, int width)
    {
        // Multiply-shift hash for position
        long seed = _hashSeeds[depth * 2];
        long hash = (seed * element + 0x9E3779B9L) & 0x7FFFFFFFL;
        return (int)(hash % width);
    }

    private int HashSign(int element, int depth)
    {
        // Sign hash: returns +1 or -1
        long seed = _hashSeeds[depth * 2 + 1];
        long hash = (seed * element + 0x517CC1B7L) & 0x7FFFFFFFL;
        return (hash % 2 == 0) ? 1 : -1;
    }

    private static double Median(double[] values, int count)
    {
        if (count == 0) return 0;
        if (count == 1) return values[0];

        var sorted = new double[count];
        Array.Copy(values, sorted, count);
        Array.Sort(sorted);

        if (count % 2 == 0)
        {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
        }

        return sorted[count / 2];
    }
}
