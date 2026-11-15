using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>The widths of the Panoptic-DeepLab semantic decoder and head kMaX-DeepLab trains its auxiliary loss on.</summary>
internal sealed class SemanticDecoderShape
{
    /// <param name="asppDim">Width of every ASPP branch and of its projection (<c>decoder.aspp_channels</c>, 256).</param>
    /// <param name="atrousRates">Dilation rates of the ASPP's 3x3 branches (<c>decoder.atrous_rates</c>, 6, 12, 18).</param>
    /// <param name="decoderDim">Width of the fusion convolutions (<c>decoder.decoder_channels</c>, 256).</param>
    /// <param name="lowLevelProjections">Widths of the 1x1 projections of the low-level features, coarsest first
    /// (<c>auxiliary_low_level</c>: res3 to 64, res2 to 32).</param>
    /// <param name="headDim">Width of the semantic head's 5x5 depthwise-separable convolution
    /// (<c>auxiliary_semantic_head.head_channels</c>, 256).</param>
    /// <param name="classes">Semantic head outputs.</param>
    /// <param name="dropRate">Dropout after the ASPP projection (0.1, deeplab2's <c>ASPP._proj_drop</c>).</param>
    internal SemanticDecoderShape(int asppDim, int[] atrousRates, int decoderDim, int[] lowLevelProjections, int headDim,
        int classes, double dropRate)
    {
        if (atrousRates is null) throw new ArgumentNullException(nameof(atrousRates));
        if (lowLevelProjections is null) throw new ArgumentNullException(nameof(lowLevelProjections));
        if (asppDim < 1 || decoderDim < 1 || headDim < 1 || classes < 1)
            throw new ArgumentException("The semantic decoder widths must be positive.");
        if (atrousRates.Any(rate => rate < 1)) throw new ArgumentException("The atrous rates must be positive.", nameof(atrousRates));
        if (lowLevelProjections.Any(width => width < 1))
            throw new ArgumentException("The low-level projection widths must be positive.", nameof(lowLevelProjections));
        if (!(dropRate >= 0 && dropRate < 1)) throw new ArgumentOutOfRangeException(nameof(dropRate), "The dropout rate must be in [0, 1).");
        AsppDim = asppDim;
        AtrousRates = (int[])atrousRates.Clone();
        DecoderDim = decoderDim;
        LowLevelProjections = (int[])lowLevelProjections.Clone();
        HeadDim = headDim;
        Classes = classes;
        DropRate = dropRate;
    }

    internal int AsppDim { get; }
    internal IReadOnlyList<int> AtrousRates { get; }
    internal int DecoderDim { get; }
    internal IReadOnlyList<int> LowLevelProjections { get; }
    internal int HeadDim { get; }
    internal int Classes { get; }
    internal double DropRate { get; }
}

/// <summary>
/// kMaX-DeepLab's auxiliary semantic decoder and head: deeplab2's <c>PanopticDeepLabSingleDecoder</c> (ASPP on the
/// stride-32 backbone features, then a fusion with each low-level backbone feature) followed by its
/// <c>PanopticDeepLabSingleHead</c>, with kMaX-DeepLab's GELU activations and BatchNorms, built from standard layers held
/// in the model's layer list.
/// </summary>
/// <remarks>
/// <para>
/// References: deeplab2 <c>model/kmax_deeplab.py</c> (the COCO / ADE20K predictor sets
/// <c>use_auxiliary_semantic_decoder=True</c>, activation <c>gelu</c>), <c>model/decoder/max_deeplab.py</c> (the
/// decoder runs on <c>feature_semantic</c>, which <c>model/transformer_decoder/kmax.py</c> sets to the backbone's
/// <c>stage5</c>), <c>model/decoder/panoptic_deeplab.py</c> and <c>model/decoder/aspp.py</c>; widths from
/// <c>configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto</c>.
/// </para>
/// <list type="number">
/// <item>ASPP: a 1x1 branch, one 3x3 branch per atrous rate and an image-pooling branch (global average, 1x1, then
/// broadcast back), each a bias-free convolution, BN and GELU; the concatenation is projected by a bias-free 1x1
/// convolution, BN and GELU, then dropout 0.1.</item>
/// <item>For each low-level feature (coarsest first): a bias-free 1x1 projection, BN and GELU; the running features are
/// bilinearly resized to it (align corners), concatenated with it and fused by a 5x5 depthwise-separable convolution
/// (depthwise, BN, GELU, then 1x1 to the decoder width, BN, GELU).</item>
/// <item>The head: a 5x5 depthwise-separable convolution as above to the head width, then a biased 1x1 convolution to
/// the classes.</item>
/// </list>
/// <para>
/// The atrous convolutions run as TensorFlow runs them (<c>tf.nn.atrous_conv2d</c>): the input is zero-padded to a
/// multiple of the rate, split into rate x rate sub-grids (space to batch), convolved by an ordinary 3x3 convolution
/// with padding 1 and interleaved back (batch to space), which is exactly the dilated convolution with "same"
/// padding.
/// </para>
/// </remarks>
internal static class PanopticDeepLabSemanticDecoder
{
    /// <summary>Kernel of the fusion and head depthwise convolutions (5).</summary>
    internal const int SeparableKernel = 5;

    /// <summary>Layers of the ASPP: the 1x1, atrous and pooling branches with their BNs, the projection, BN, dropout.</summary>
    internal static int AsppLayerCount(SemanticDecoderShape shape) => 2 * (shape.AtrousRates.Count + 3) + 1;

    /// <summary>Layers of one low-level fusion: projection + BN, depthwise + BN, pointwise + BN.</summary>
    internal const int FusionLayerCount = 6;

    /// <summary>Layers of the head: depthwise + BN, pointwise + BN, the final classifier.</summary>
    internal const int HeadLayerCount = 5;

    /// <summary>All layers of the decoder and head.</summary>
    internal static int LayerCount(SemanticDecoderShape shape)
        => AsppLayerCount(shape) + shape.LowLevelProjections.Count * FusionLayerCount + HeadLayerCount;

    /// <summary>Creates the layers, in the order <see cref="Run"/> reads them.</summary>
    internal static IEnumerable<ILayer<T>> CreateLayers<T>(SemanticDecoderShape shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        yield return Pointwise<T>(shape.AsppDim); yield return Norm<T>();                 // 1x1 branch
        foreach (var _ in shape.AtrousRates)
        {
            yield return new ConvolutionalLayer<T>(shape.AsppDim, 3, 1, 1, biasMode: BiasMode.Never); // atrous 3x3 branch
            yield return Norm<T>();
        }

        yield return Pointwise<T>(shape.AsppDim); yield return Norm<T>();                 // image-pooling branch
        yield return Pointwise<T>(shape.AsppDim); yield return Norm<T>();                 // projection
        yield return new DropoutLayer<T>(shape.DropRate);

        int running = shape.AsppDim;
        foreach (int projection in shape.LowLevelProjections)
        {
            yield return Pointwise<T>(projection); yield return Norm<T>();                // low-level projection
            foreach (var layer in Separable<T>(running + projection, shape.DecoderDim)) yield return layer;
            running = shape.DecoderDim;
        }

        foreach (var layer in Separable<T>(running, shape.HeadDim)) yield return layer;
        yield return new ConvolutionalLayer<T>(shape.Classes, 1, 1, 0);                    // final_conv (biased)
    }

    private static IEnumerable<ILayer<T>> Separable<T>(int inputChannels, int outputChannels)
    {
        yield return ConvolutionalLayer<T>.WithInputDepth(inputChannels, inputChannels, SeparableKernel, 1, SeparableKernel / 2,
            groups: inputChannels, biasMode: BiasMode.Never);
        yield return Norm<T>();
        yield return ConvolutionalLayer<T>.WithInputDepth(inputChannels, outputChannels, 1, 1, 0, biasMode: BiasMode.Never);
        yield return Norm<T>();
    }

    private static ConvolutionalLayer<T> Pointwise<T>(int width) => new ConvolutionalLayer<T>(width, 1, 1, 0, biasMode: BiasMode.Never);

    private static BatchNormalizationLayer<T> Norm<T>() => new BatchNormalizationLayer<T>();

    /// <summary>
    /// Runs the decoder and head.
    /// </summary>
    /// <param name="layers">The model's layers.</param>
    /// <param name="start">Index of the decoder's first layer.</param>
    /// <param name="shape">The decoder widths.</param>
    /// <param name="stages">Backbone features <c>[B, C, h, w]</c>, finest first. The ASPP reads the coarsest; the
    /// low-level projection <c>i</c> of <c>P</c> reads stage <c>P - 1 - i</c> (res3 then res2 with the default two and
    /// stride-4, 8, 16, 32 stages).</param>
    /// <returns>Semantic logits <c>[B, K, h, w]</c> at the resolution of the last low-level feature read.</returns>
    internal static Tensor<T> Run<T>(IReadOnlyList<ILayer<T>> layers, int start, SemanticDecoderShape shape, IReadOnlyList<Tensor<T>> stages)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (stages is null) throw new ArgumentNullException(nameof(stages));
        if (stages.Count == 0) throw new ArgumentException("At least one backbone stage is required.", nameof(stages));
        int projections = shape.LowLevelProjections.Count;
        if (projections > stages.Count)
            throw new ArgumentException(
                $"The semantic decoder fuses {projections} low-level features but the backbone has {stages.Count} stages.", nameof(stages));
        if (start < 0 || start + LayerCount(shape) > layers.Count)
            throw new ArgumentException("The layer list does not hold the semantic decoder's layers at the given start.", nameof(layers));

        var engine = AiDotNetEngine.Current;
        int index = start;
        var x = Aspp(layers, ref index, shape, stages[stages.Count - 1]);
        for (int i = 0; i < projections; i++)
        {
            var low = engine.TensorGELU(layers[index + 1].Forward(layers[index].Forward(stages[projections - 1 - i])));
            index += 2;
            int h = low.Shape[2], w = low.Shape[3];
            if (x.Shape[2] != h || x.Shape[3] != w)
                x = engine.Interpolate(x, new[] { h, w }, InterpolateMode.Bilinear, true);
            x = SeparableForward(layers, ref index, engine.TensorConcatenate(new[] { x, low }, 1));
        }

        x = SeparableForward(layers, ref index, x);
        return layers[index].Forward(x);
    }

    private static Tensor<T> Aspp<T>(IReadOnlyList<ILayer<T>> layers, ref int index, SemanticDecoderShape shape, Tensor<T> input)
    {
        var engine = AiDotNetEngine.Current;
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var branches = new List<Tensor<T>>(shape.AtrousRates.Count + 2)
        {
            engine.TensorGELU(layers[index + 1].Forward(layers[index].Forward(input))),
        };
        index += 2;
        foreach (int rate in shape.AtrousRates)
        {
            branches.Add(engine.TensorGELU(layers[index + 1].Forward(AtrousConvolution(layers[index], input, rate))));
            index += 2;
        }

        var pooled = engine.ReduceMean(input, new[] { 2, 3 }, true);                                      // [B, C, 1, 1]
        pooled = engine.TensorGELU(layers[index + 1].Forward(layers[index].Forward(pooled)));
        index += 2;
        branches.Add(engine.TensorBroadcastTo(pooled, new[] { b, pooled.Shape[1], h, w }));

        var x = engine.TensorGELU(layers[index + 1].Forward(layers[index].Forward(engine.TensorConcatenate(branches.ToArray(), 1))));
        x = layers[index + 2].Forward(x);                                                                 // dropout
        index += 3;
        return x;
    }

    private static Tensor<T> SeparableForward<T>(IReadOnlyList<ILayer<T>> layers, ref int index, Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        x = engine.TensorGELU(layers[index + 1].Forward(layers[index].Forward(x)));
        x = engine.TensorGELU(layers[index + 3].Forward(layers[index + 2].Forward(x)));
        index += 4;
        return x;
    }

    /// <summary>
    /// The 3x3 convolution <paramref name="convolution"/> (stride 1, padding 1) applied with dilation
    /// <paramref name="rate"/> and "same" padding to <c>[B, C, H, W]</c>, through space to batch and batch to space.
    /// </summary>
    internal static Tensor<T> AtrousConvolution<T>(ILayer<T> convolution, Tensor<T> input, int rate)
    {
        if (convolution is null) throw new ArgumentNullException(nameof(convolution));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (rate < 1) throw new ArgumentOutOfRangeException(nameof(rate), "The atrous rate must be positive.");
        if (rate == 1) return convolution.Forward(input);

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int b = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        int gh = (h + rate - 1) / rate, gw = (w + rate - 1) / rate;
        var padded = input;
        if (gh * rate > h)
            padded = engine.TensorConcatenate(new[] { padded, Tensor<T>.CreateDefault(new[] { b, c, gh * rate - h, w }, ops.Zero) }, 2);
        if (gw * rate > w)
            padded = engine.TensorConcatenate(new[] { padded, Tensor<T>.CreateDefault(new[] { b, c, gh * rate, gw * rate - w }, ops.Zero) }, 3);

        // [B, C, gh, r, gw, r] -> [r, r, B, C, gh, gw]: sub-grid (a, d) holds the pixels (i * r + a, j * r + d).
        var grids = engine.Reshape(
            engine.TensorPermute(engine.Reshape(padded, new[] { b, c, gh, rate, gw, rate }), new[] { 3, 5, 0, 1, 2, 4 }),
            new[] { rate * rate * b, c, gh, gw });
        var convolved = convolution.Forward(grids);                                                      // [r*r*B, O, gh, gw]
        int o = convolved.Shape[1];
        var merged = engine.Reshape(
            engine.TensorPermute(engine.Reshape(convolved, new[] { rate, rate, b, o, gh, gw }), new[] { 2, 3, 4, 0, 5, 1 }),
            new[] { b, o, gh * rate, gw * rate });
        return gh * rate == h && gw * rate == w ? merged : engine.TensorSlice(merged, new[] { 0, 0, 0, 0 }, new[] { b, o, h, w });
    }
}
