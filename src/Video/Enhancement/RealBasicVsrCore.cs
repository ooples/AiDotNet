using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RealBasicVSRNet (Chan et al., CVPR 2022; open-mmlab/mmagic, models/editors/real_basicvsr): an image-cleaning module
/// (residual blocks with an input convolution, then a 3x3 convolution to RGB) applied as a residual up to three
/// times, followed by the BasicVSR backbone. Shared by RealBasicVSR and RealBasicVSR-Sharp, the release's two training
/// stages of the same network.
/// </summary>
internal sealed class RealBasicVsrCore<T>
{
    private const int MaxCleaningPasses = 3;
    private readonly ResidualTrunkBlock<T> _cleaning;
    private readonly int _cleaningOut;

    internal RealBasicVsrCore(List<ILayer<T>> layers, int midChannels, int propagationBlocks, int cleaningBlocks)
    {
        First = layers.Count;
        _cleaning = new ResidualTrunkBlock<T>(layers, midChannels, cleaningBlocks);
        _cleaningOut = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3, 3, null));
        Backbone = new BasicVsrBackbone<T>(layers, midChannels, propagationBlocks);
    }

    /// <summary>Index of the first layer in the owner's list.</summary>
    internal int First { get; }

    internal BasicVsrBackbone<T> Backbone { get; }

    /// <summary>
    /// RealBasicVSRNet.forward on clips [N, T, 3, H, W] in [0, 1]: clean the frames (time folded into the batch) until the
    /// mean absolute residue falls below <paramref name="threshold"/> or three passes ran, then super-resolve them.
    /// Returns the output and the cleaned frames the cleaning loss scores.
    /// </summary>
    internal (Tensor<T> Output, Tensor<T> Cleaned) Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips, double threshold)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = clips.Shape[0], t = clips.Shape[1], c = clips.Shape[2], h = clips.Shape[3], w = clips.Shape[4];
        var frames = engine.Reshape(clips, new[] { n * t, c, h, w });
        LastCleaningPasses = 0;
        for (int pass = 0; pass < MaxCleaningPasses; pass++)
        {
            LastCleaningPasses++;
            var residues = BasicVsrBlocks.Run(layers, _cleaningOut, _cleaning.Forward(engine, layers, frames));
            frames = engine.TensorAdd(frames, residues);
            double meanAbs = 0;
            var span = residues.Data.Span;
            for (int i = 0; i < span.Length; i++) meanAbs += Math.Abs(ops.ToDouble(span[i]));
            if (meanAbs / Math.Max(1, span.Length) < threshold) break;
        }
        var cleaned = engine.Reshape(frames, new[] { n, t, c, h, w });
        return (Backbone.Run(engine, layers, cleaned), cleaned);
    }

    /// <summary>How many cleaning passes the last <see cref="Run"/> applied.</summary>
    internal int LastCleaningPasses { get; private set; }

    /// <summary>The SPyNet parameters, which RealBasicVSRNet never trains.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers) => Backbone.FlowParameters(layers);

    /// <summary>F.interpolate(scale_factor=0.25, mode='area') of frames [..., H, W]: 4x4 means.</summary>
    internal static Tensor<T> AreaDownsample4(IEngine engine, Tensor<T> clips)
    {
        int rank = clips.Rank, h = clips.Shape[rank - 2], w = clips.Shape[rank - 1];
        int lead = clips.Length / (h * w);
        var pooled = engine.AvgPool2D(engine.Reshape(clips, new[] { lead, 1, h, w }), 4, 4, 0, false);
        var shape = clips.Shape.ToArray();
        shape[rank - 2] = h / 4;
        shape[rank - 1] = w / 4;
        return engine.Reshape(pooled, shape);
    }

    /// <summary>
    /// mmagic's UnsharpMasking on frames [..., H, W] in [0, 1]: residue = img - G(img) with a normalized Gaussian of
    /// size <paramref name="kernelSize"/> (cv2 sigma 0, i.e. 0.3((k-1)/2-1)+0.8), mask = |residue|·255 &gt; threshold,
    /// soft mask = G(mask), result = soft·clip(img + weight·residue, 0, 1) + (1 - soft)·img. Borders reflect as
    /// cv2.filter2D's BORDER_REFLECT_101.
    /// </summary>
    internal static Tensor<T> UnsharpMask(Tensor<T> frames, double weight, int kernelSize, double threshold)
    {
        if (kernelSize < 1) throw new ArgumentOutOfRangeException(nameof(kernelSize), "Kernel size must be positive.");
        var ops = MathHelper.GetNumericOperations<T>();
        int rank = frames.Rank, h = frames.Shape[rank - 2], w = frames.Shape[rank - 1];
        int planes = frames.Length / (h * w);
        double sigma = 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
        var kernel = new double[kernelSize];
        double total = 0;
        for (int i = 0; i < kernelSize; i++)
        {
            double d = i - (kernelSize - 1) / 2.0;
            kernel[i] = Math.Exp(-d * d / (2 * sigma * sigma));
            total += kernel[i];
        }
        for (int i = 0; i < kernelSize; i++) kernel[i] /= total;

        var result = new Tensor<T>(frames.Shape.ToArray());
        var source = frames.Data.Span;
        var target = result.Data.Span;
        var image = new double[h * w];
        var mask = new double[h * w];
        for (int p = 0; p < planes; p++)
        {
            for (int i = 0; i < h * w; i++) image[i] = ops.ToDouble(source[p * h * w + i]);
            var blurred = Blur(image, h, w, kernel);
            for (int i = 0; i < h * w; i++) mask[i] = Math.Abs(image[i] - blurred[i]) * 255 > threshold ? 1.0 : 0.0;
            var soft = Blur(mask, h, w, kernel);
            for (int i = 0; i < h * w; i++)
            {
                double sharpened = Math.Min(1.0, Math.Max(0.0, image[i] + weight * (image[i] - blurred[i])));
                target[p * h * w + i] = ops.FromDouble(soft[i] * sharpened + (1 - soft[i]) * image[i]);
            }
        }
        return result;
    }

    private static double[] Blur(double[] image, int h, int w, double[] kernel)
    {
        int radius = kernel.Length / 2;
        var rows = new double[h * w];
        var output = new double[h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double s = 0;
                for (int k = 0; k < kernel.Length; k++) s += kernel[k] * image[y * w + Reflect101(x + k - radius, w)];
                rows[y * w + x] = s;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double s = 0;
                for (int k = 0; k < kernel.Length; k++) s += kernel[k] * rows[Reflect101(y + k - radius, h) * w + x];
                output[y * w + x] = s;
            }
        return output;
    }

    // cv2 BORDER_REFLECT_101: ... 2 1 | 0 1 2 ... n-1 | n-2 n-3 ...
    internal static int Reflect101(int index, int length)
    {
        if (length == 1) return 0;
        int period = 2 * (length - 1);
        int k = ((index % period) + period) % period;
        return k < length ? k : period - k;
    }
}

/// <summary>
/// mmagic's ExponentialMovingAverage of a generator (averaged = (1 - momentum)·averaged + momentum·current after every
/// step), whose weights the release evaluates with.
/// </summary>
internal sealed class ParameterAverage<T>
{
    private readonly double _momentum;
    private List<T[]>? _average;
    private Vector<T>? _restored;

    internal ParameterAverage(double momentum) => _momentum = momentum;

    /// <summary>Whether an average exists yet (it starts as a copy of the weights at the first update).</summary>
    internal bool HasAverage => _average is not null || _restored is not null;

    /// <summary>The average as one flat vector in parameter order, for serialization; null before the first update.</summary>
    internal Vector<T>? ToVector()
    {
        if (_average is null) return _restored;
        var flat = new Vector<T>(_average.Sum(a => a.Length));
        int offset = 0;
        foreach (var block in _average)
            for (int i = 0; i < block.Length; i++) flat[offset++] = block[i];
        return flat;
    }

    /// <summary>
    /// Restores a serialized average. The per-tensor split needs the parameters, which a deserialized model creates
    /// lazily, so the vector is kept until the next <see cref="Update"/> or <see cref="WithAverage"/> splits it.
    /// </summary>
    internal void Restore(Vector<T>? flat)
    {
        _average = null;
        _restored = flat;
    }

    private void Split(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_restored is not { } flat) return;
        _restored = null;
        if (parameters.Sum(p => p.Length) != flat.Length) return;
        _average = new List<T[]>(parameters.Count);
        int offset = 0;
        foreach (var p in parameters)
        {
            var block = new T[p.Length];
            for (int i = 0; i < block.Length; i++) block[i] = flat[offset++];
            _average.Add(block);
        }
    }

    internal void Update(IReadOnlyList<Tensor<T>> parameters)
    {
        Split(parameters);
        var ops = MathHelper.GetNumericOperations<T>();
        if (_average is null || _average.Count != parameters.Count)
        {
            _average = parameters.Select(p => p.ToArray()).ToList();
            return;
        }
        for (int k = 0; k < parameters.Count; k++)
        {
            var current = parameters[k].Data.Span;
            var average = _average[k];
            for (int i = 0; i < average.Length; i++)
                average[i] = ops.FromDouble((1 - _momentum) * ops.ToDouble(average[i]) + _momentum * ops.ToDouble(current[i]));
        }
    }

    /// <summary>Runs <paramref name="action"/> with the averaged weights in place, restoring the trained ones after.</summary>
    internal TResult WithAverage<TResult>(IReadOnlyList<Tensor<T>> parameters, Func<TResult> action)
    {
        Split(parameters);
        if (_average is null || _average.Count != parameters.Count) return action();
        var saved = parameters.Select(p => p.ToArray()).ToList();
        try
        {
            for (int k = 0; k < parameters.Count; k++) _average[k].AsSpan().CopyTo(parameters[k].Data.Span);
            return action();
        }
        finally
        {
            for (int k = 0; k < parameters.Count; k++) saved[k].AsSpan().CopyTo(parameters[k].Data.Span);
        }
    }
}
