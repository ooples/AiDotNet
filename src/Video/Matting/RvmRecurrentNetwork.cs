using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Matting;

/// <summary>
/// RVM's encoder, LR-ASPP and recurrent decoder (Lin et al., "Robust High-Resolution Video Matting with Temporal
/// Guidance", WACV 2022, Sec. 3 and Fig. 2), run over the layers <c>LayerHelper.CreateDefaultRVMLayers</c> builds.
/// </summary>
/// <remarks>
/// <para>
/// A view over <c>RVM.Layers</c>, bound per forward, so a deserialized model (whose layers are replaced) always runs
/// its own weights. The recurrent state is four ConvGRU hidden tensors, bottleneck first; a missing or mismatched
/// state starts from zeros as in the reference (<c>h = zeros_like(x)</c>).
/// </para>
/// <para>
/// ConvGRU (reference <c>model/decoder.py</c>): <c>r, z = sigmoid(conv_ih([x, h]))</c>,
/// <c>c = tanh(conv_hh([x, r * h]))</c>, <c>h' = (1 - z) * h + z * c</c>, applied to the second half of each
/// decoder block's channels. The output is <c>[alpha, foreground]</c> with <c>alpha = clamp(pha, 0, 1)</c> and
/// <c>foreground = clamp(residual + source, 0, 1)</c>.
/// </para>
/// </remarks>
internal sealed class RvmRecurrentNetwork<T>
{
    /// <summary>Number of layers the network binds.</summary>
    internal const int LayerCount = 26;

    /// <summary>Number of ConvGRU hidden states (bottleneck, 1/8, 1/4, 1/2).</summary>
    internal const int StateCount = 4;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IEngine _engine;
    private readonly ConvolutionalLayer<T>[] _encoder;
    private readonly ConvolutionalLayer<T> _asppConv;
    private readonly BatchNormalizationLayer<T> _asppNorm;
    private readonly ConvolutionalLayer<T> _asppGate;
    private readonly (ConvolutionalLayer<T> Ih, ConvolutionalLayer<T> Hh) _bottleneck;
    private readonly (ConvolutionalLayer<T> Conv, BatchNormalizationLayer<T> Norm,
        ConvolutionalLayer<T> Ih, ConvolutionalLayer<T> Hh)[] _upsampling;
    private readonly ConvolutionalLayer<T> _outputConv1;
    private readonly BatchNormalizationLayer<T> _outputNorm1;
    private readonly ConvolutionalLayer<T> _outputConv2;
    private readonly BatchNormalizationLayer<T> _outputNorm2;
    private readonly ConvolutionalLayer<T> _projection;

    internal RvmRecurrentNetwork(IEngine engine, IReadOnlyList<ILayer<T>> layers)
    {
        _engine = engine;
        if (layers.Count != LayerCount)
            throw new InvalidOperationException(
                $"RVM binds {LayerCount} layers in LayerHelper.CreateDefaultRVMLayers order; the network has {layers.Count}.");

        int index = 0;
        ConvolutionalLayer<T> Conv() => layers[index++] as ConvolutionalLayer<T>
            ?? throw new InvalidOperationException($"RVM layer {index - 1} must be a ConvolutionalLayer.");
        BatchNormalizationLayer<T> Norm() => layers[index++] as BatchNormalizationLayer<T>
            ?? throw new InvalidOperationException($"RVM layer {index - 1} must be a BatchNormalizationLayer.");

        _encoder = new[] { Conv(), Conv(), Conv(), Conv() };
        _asppConv = Conv();
        _asppNorm = Norm();
        _asppGate = Conv();
        _bottleneck = (Conv(), Conv());
        _upsampling = new (ConvolutionalLayer<T>, BatchNormalizationLayer<T>, ConvolutionalLayer<T>, ConvolutionalLayer<T>)[3];
        for (int i = 0; i < _upsampling.Length; i++)
        {
            var conv = Conv();
            var norm = Norm();
            _upsampling[i] = (conv, norm, Conv(), Conv());
        }

        _outputConv1 = Conv();
        _outputNorm1 = Norm();
        _outputConv2 = Conv();
        _outputNorm2 = Norm();
        _projection = Conv();
    }

    /// <summary>
    /// Mattes one frame <c>[B, C, H, W]</c> given the previous hidden states, returning <c>[B, 4, H, W]</c>
    /// (alpha, then foreground RGB) and the next states.
    /// </summary>
    internal (Tensor<T> Output, Tensor<T>[] State) Forward(Tensor<T> source, IReadOnlyList<Tensor<T>?> state)
    {
        var e = _engine;
        int channels = source.Shape[1];

        // The encoder reads ImageNet-normalized RGB, as the reference's backbone does
        var encoded = channels == 3 ? Normalize(source) : source;
        var f1 = _encoder[0].Forward(encoded);
        var f2 = _encoder[1].Forward(f1);
        var f3 = _encoder[2].Forward(f2);
        var f4 = _encoder[3].Forward(f3);

        // LR-ASPP
        var gate = _asppGate.Forward(e.ReduceMean(f4, new[] { 2, 3 }, true));
        var aspp = e.ReLU(_asppNorm.Forward(_asppConv.Forward(f4)));
        var x4 = e.TensorMultiply(aspp, e.TensorBroadcastTo(gate, aspp.Shape.ToArray()));

        // Image pyramid at 1/2, 1/4, 1/8 (AvgPool2d(2, 2, ceil_mode=True, count_include_pad=False))
        var s1 = Downsample(source);
        var s2 = Downsample(s1);
        var s3 = Downsample(s2);

        var next = new Tensor<T>[StateCount];
        Tensor<T>? Previous(int i) => i < state.Count ? state[i] : null;

        var (x, h) = RecurrentHalf(x4, Previous(0), _bottleneck.Ih, _bottleneck.Hh);
        next[0] = h;
        var skips = new[] { (f3, s3), (f2, s2), (f1, s1) };
        for (int i = 0; i < _upsampling.Length; i++)
        {
            var (skip, image) = skips[i];
            var block = _upsampling[i];
            var merged = e.TensorConcatenate(new[] { UpsampleTo(x, skip.Shape[2], skip.Shape[3]), skip, image }, 1);
            var features = e.ReLU(block.Norm.Forward(block.Conv.Forward(merged)));
            (x, h) = RecurrentHalf(features, Previous(i + 1), block.Ih, block.Hh);
            next[i + 1] = h;
        }

        // Output block and projection at full resolution
        int height = source.Shape[2], width = source.Shape[3];
        var full = e.TensorConcatenate(new[] { UpsampleTo(x, height, width), source }, 1);
        full = e.ReLU(_outputNorm1.Forward(_outputConv1.Forward(full)));
        full = e.ReLU(_outputNorm2.Forward(_outputConv2.Forward(full)));
        var projected = _projection.Forward(full);

        int rgb = Math.Min(3, channels);
        var residual = e.TensorNarrow(projected, 1, 0, rgb);
        var foreground = e.TensorClamp(e.TensorAdd(residual, e.TensorNarrow(source, 1, 0, rgb)), NumOps.Zero, NumOps.One);
        var alpha = e.TensorClamp(e.TensorNarrow(projected, 1, 3, 1), NumOps.Zero, NumOps.One);
        return (e.TensorConcatenate(new[] { alpha, foreground }, 1), next);
    }

    /// <summary>
    /// The reference's decoder split: the first half of the channels passes through, the second half is the input
    /// of a ConvGRU whose new hidden state replaces it.
    /// </summary>
    private (Tensor<T> Output, Tensor<T> State) RecurrentHalf(
        Tensor<T> x, Tensor<T>? hidden, ConvolutionalLayer<T> ih, ConvolutionalLayer<T> hh)
    {
        int half = x.Shape[1] / 2;
        var kept = _engine.TensorNarrow(x, 1, 0, x.Shape[1] - half);
        var recurrent = _engine.TensorNarrow(x, 1, x.Shape[1] - half, half);
        var h = ConvGru(recurrent, hidden, ih, hh);
        return (_engine.TensorConcatenate(new[] { kept, h }, 1), h);
    }

    /// <summary>One ConvGRU step: r, z = sigmoid(ih([x, h])); c = tanh(hh([x, r h])); h' = h + z (c - h).</summary>
    internal Tensor<T> ConvGru(Tensor<T> x, Tensor<T>? hidden, ConvolutionalLayer<T> ih, ConvolutionalLayer<T> hh)
    {
        var e = _engine;
        var h = hidden is not null && hidden.Shape.ToArray().SequenceEqual(x.Shape.ToArray()) ? hidden : e.TensorZerosLike(x);
        int c = x.Shape[1];
        var gates = ih.Forward(e.TensorConcatenate(new[] { x, h }, 1));
        var reset = e.TensorNarrow(gates, 1, 0, c);
        var update = e.TensorNarrow(gates, 1, c, c);
        var candidate = hh.Forward(e.TensorConcatenate(new[] { x, e.TensorMultiply(reset, h) }, 1));
        return e.TensorAdd(h, e.TensorMultiply(update, e.TensorSubtract(candidate, h)));
    }

    private Tensor<T> Normalize(Tensor<T> source)
    {
        var shape = source.Shape.ToArray();
        var mean = new Tensor<double>(new[] { 0.485, 0.456, 0.406 }, new[] { 1, 3, 1, 1 }).Cast<T>();
        var std = new Tensor<double>(new[] { 0.229, 0.224, 0.225 }, new[] { 1, 3, 1, 1 }).Cast<T>();
        return _engine.TensorDivide(
            _engine.TensorSubtract(source, _engine.TensorBroadcastTo(mean, shape)), _engine.TensorBroadcastTo(std, shape));
    }

    /// <summary>
    /// 2x2 average pooling with ceil_mode=True and count_include_pad=False: an odd trailing row or column averages
    /// only its real pixels.
    /// </summary>
    private Tensor<T> Downsample(Tensor<T> image)
    {
        int height = image.Shape[2], width = image.Shape[3];
        int padBottom = height % 2, padRight = width % 2;
        if (padBottom == 0 && padRight == 0)
            return _engine.AvgPool2D(image, 2, 2, 0, false);

        var padded = _engine.Pad(image, 0, padBottom, 0, padRight, NumOps.Zero);
        var coverage = _engine.Pad(Tensor<T>.CreateDefault(new[] { 1, 1, height, width }, NumOps.One),
            0, padBottom, 0, padRight, NumOps.Zero);
        var sums = _engine.AvgPool2D(padded, 2, 2, 0, false);
        var counts = _engine.AvgPool2D(coverage, 2, 2, 0, false);
        return _engine.TensorDivide(sums, _engine.TensorBroadcastTo(counts, sums.Shape.ToArray()));
    }

    // Bilinear 2x upsampling (align_corners=False) cropped to the skip connection's size
    private Tensor<T> UpsampleTo(Tensor<T> x, int height, int width)
    {
        var up = _engine.Interpolate(x, new[] { x.Shape[2] * 2, x.Shape[3] * 2 }, InterpolateMode.Bilinear, false);
        return _engine.TensorNarrow(_engine.TensorNarrow(up, 2, 0, height), 3, 0, width);
    }
}
