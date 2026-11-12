using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
    public (Matrix<T> Sketch, int OriginalSize, int Width) Compress(Tensor<T> gradient)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        int originalSize = gradient.Shape[0];
        int depth = _options.SketchDepth;
        int width = _options.SketchWidth > 0
            ? _options.SketchWidth
            : Math.Max(100, originalSize / 20); // Auto: ~5% of original size

        var values = Engine.TensorSlice(gradient.Reshape(gradient.Length), new[] { 0 }, new[] { originalSize });

        // Entries with |value| < 1e-15 are not inserted.
        var significant = Engine.TensorGreaterThan(Engine.TensorAbs(values), NumOps.FromDouble(1e-15));
        values = Engine.TensorWhere(significant, values, new Tensor<T>(new[] { originalSize }));

        var rows = new Tensor<T>[depth];
        for (int d = 0; d < depth; d++)
        {
            var (positions, signs) = HashRow(d, originalSize, width);
            rows[d] = Engine.TensorScatterAdd(
                new Tensor<T>(new[] { width }), positions, Engine.TensorMultiply(signs, values), 0);
        }

        var sketch = depth > 0 ? Engine.TensorStack(rows, 0).ToMatrix() : new Matrix<T>(0, width);
        return (sketch, originalSize, width);
    }

    /// <summary>
    /// Decompresses a Count Sketch back to a gradient tensor using median estimation.
    /// </summary>
    /// <param name="sketch">The sketch table.</param>
    /// <param name="originalSize">Original gradient vector size.</param>
    /// <param name="width">Sketch width used during compression.</param>
    /// <returns>Reconstructed gradient tensor.</returns>
    public Tensor<T> Decompress(Matrix<T> sketch, int originalSize, int width)
    {
        if (sketch is null) throw new ArgumentNullException(nameof(sketch));
        return MedianEstimates(sketch, originalSize, width);
    }

    /// <summary>
    /// Decompresses only the top-k largest elements from the sketch.
    /// </summary>
    /// <param name="sketch">The sketch table.</param>
    /// <param name="originalSize">Original gradient vector size.</param>
    /// <param name="width">Sketch width used during compression.</param>
    /// <param name="topK">Number of top elements to recover. 0 = recover all.</param>
    /// <returns>Sparse gradient tensor with only top-k elements non-zero.</returns>
    public Tensor<T> DecompressTopK(Matrix<T> sketch, int originalSize, int width, int topK)
    {
        if (sketch is null) throw new ArgumentNullException(nameof(sketch));
        if (topK <= 0 || topK >= originalSize)
        {
            return Decompress(sketch, originalSize, width);
        }

        var estimates = MedianEstimates(sketch, originalSize, width);
        Engine.TensorTopK(Engine.TensorAbs(estimates), topK, 0, out var topIndices);
        var mask = new int[originalSize];
        foreach (int index in topIndices.ToArray())
        {
            mask[index] = 1;
        }

        return Engine.TensorWhere(
            Vector<int>.Wrap(mask).AsTensor().Cast<T>(), estimates, new Tensor<T>(new[] { originalSize }));
    }

    /// <summary>
    /// Per-element estimate: the median over sketch rows of sign_d(i) * sketch[d, h_d(i)].
    /// </summary>
    private Tensor<T> MedianEstimates(Matrix<T> sketch, int originalSize, int width)
    {
        int depth = _options.SketchDepth;
        if (depth == 0)
        {
            return new Tensor<T>(new[] { originalSize });
        }

        var table = Tensor<T>.FromMatrix(sketch);
        var rows = new Tensor<T>[depth];
        for (int d = 0; d < depth; d++)
        {
            var (positions, signs) = HashRow(d, originalSize, width);
            var row = Engine.TensorSlice(table, new[] { d, 0 }, new[] { 1, width }).Reshape(width);
            rows[d] = Engine.TensorMultiply(signs, Engine.TensorGather(row, positions, 0));
        }

        // Median across rows (mean of the two middle values for an even depth) for robustness to collisions.
        var sorted = Engine.TensorSort(Engine.TensorStack(rows, 0), 0, false).Values;
        var upper = Engine.TensorSlice(sorted, new[] { depth / 2, 0 }, new[] { 1, originalSize }).Reshape(originalSize);
        if (depth % 2 == 1)
        {
            return upper;
        }

        var lower = Engine.TensorSlice(sorted, new[] { (depth / 2) - 1, 0 }, new[] { 1, originalSize }).Reshape(originalSize);
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(lower, upper), NumOps.FromDouble(0.5));
    }

    /// <summary>Hash bucket indices and +-1 signs of elements 0..count-1 for sketch row <paramref name="depth"/>.</summary>
    private (Tensor<int> Positions, Tensor<T> Signs) HashRow(int depth, int count, int width)
    {
        var positions = new int[count];
        var signs = new int[count];
        for (int i = 0; i < count; i++)
        {
            positions[i] = HashPosition(i, depth, width);
            signs[i] = HashSign(i, depth);
        }

        return (Vector<int>.Wrap(positions).AsTensor(), Vector<int>.Wrap(signs).AsTensor().Cast<T>());
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
}
