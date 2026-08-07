using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Paper-structured ViT-CoMer encoder and dense-prediction head with parallel ViT/CNN branches,
/// multi-receptive-field refinement, and bidirectional cross-branch interaction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The implementation follows Xia et al., "ViT-CoMer: Vision Transformer with Convolutional
/// Multi-scale Feature Interaction for Dense Predictions" (CVPR 2024): a plain ViT remains at
/// 1/16 resolution while a spatial-prior CNN builds a four-level pyramid. Before every ViT stage,
/// 3x3/5x5 depthwise MRFP paths refine the CNN pyramid and cross-attention injects its multi-scale
/// tokens into the ViT. At the stage end a second cross-attention sends the global ViT context back
/// to every CNN scale. A compact FPN-style head consumes all four fused levels.
/// </para>
/// <para>
/// The paper uses multi-scale deformable attention inside CTI. AiDotNet's native attention API does
/// not expose learned reference-point offsets, so this layer uses full multi-head cross-attention
/// over the same query/context token sets. It preserves CTI's directionality, trainable projections,
/// stage placement, and multi-scale information flow while sampling every context location instead
/// of a learned sparse subset.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Transformer)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High,
    TestInputShape = "1, 3, 32, 32",
    TestConstructorArgs = "3, 32, 32, 16, new int[] { 8, 12, 16, 24 }, new int[] { 1, 1, 1, 1 }, 8, 4, 0.0")]
[AutoParameters]
public partial class ViTCoMerSegmentationLayer<T> : LayerBase<T>
{
    private readonly int _inputChannels;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _embedDim;
    private readonly int[] _cnnChannels;
    private readonly int[] _depths;
    private readonly int _decoderDim;
    private readonly int _numClasses;
    private readonly double _dropRate;
    private readonly int _numHeads;
    private readonly int[] _levelHeights;
    private readonly int[] _levelWidths;

    private readonly ConvolutionalLayer<T> _cnnStem1;
    private readonly ConvolutionalLayer<T> _cnnStem2;
    private readonly ConvolutionalLayer<T>[] _cnnDownsamples;
    private readonly ConvolutionalLayer<T> _vitPatchEmbedding;
    private readonly TransformerEncoderBlock<T>[][] _vitStages;

    private readonly ConvolutionalLayer<T>[] _mrfp3;
    private readonly ConvolutionalLayer<T>[] _mrfp5;
    private readonly ConvolutionalLayer<T>[] _cnnToVit;
    private readonly ConvolutionalLayer<T>[] _vitToCnn;
    private readonly CrossAttentionLayer<T>[] _ctiToVit;
    private readonly CrossAttentionLayer<T>[] _ctiToCnn;

    private readonly ConvolutionalLayer<T>[] _decoderProjections;
    private readonly ConvolutionalLayer<T> _decoderFusion;
    private readonly ConvolutionalLayer<T> _classifier;

    /// <summary>Creates the native ViT-CoMer segmentation graph.</summary>
    public ViTCoMerSegmentationLayer(
        int inputChannels = 3,
        int inputHeight = 512,
        int inputWidth = 512,
        int embedDim = 384,
        int[]? cnnChannels = null,
        int[]? depths = null,
        int decoderDim = 256,
        int numClasses = 150,
        double dropRate = 0.1)
        : base(
            new[] { inputChannels, inputHeight, inputWidth },
            new[] { numClasses, Math.Max(1, inputHeight / 4), Math.Max(1, inputWidth / 4) })
    {
        cnnChannels ??= [64, 128, 320, 512];
        depths ??= [2, 2, 6, 2];
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (inputHeight < 16 || inputWidth < 16)
            throw new ArgumentOutOfRangeException(nameof(inputHeight), "ViT-CoMer requires spatial dimensions of at least 16.");
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (cnnChannels.Length != 4 || cnnChannels.Any(c => c <= 0))
            throw new ArgumentException("ViT-CoMer requires four positive CNN stage widths.", nameof(cnnChannels));
        if (depths.Length != 4 || depths.Any(d => d <= 0))
            throw new ArgumentException("ViT-CoMer requires four positive ViT stage depths.", nameof(depths));
        if (decoderDim <= 0) throw new ArgumentOutOfRangeException(nameof(decoderDim));
        if (numClasses <= 0) throw new ArgumentOutOfRangeException(nameof(numClasses));

        _inputChannels = inputChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _embedDim = embedDim;
        _cnnChannels = cnnChannels.ToArray();
        _depths = depths.ToArray();
        _decoderDim = decoderDim;
        _numClasses = numClasses;
        _dropRate = dropRate;
        _numHeads = ResolveHeadCount(embedDim);
        _levelHeights = new int[4];
        _levelWidths = new int[4];

        var gelu = new GELUActivation<T>();
        var identity = new IdentityActivation<T>();

        // Spatial-prior module: two stride-2 stem convolutions produce C1 at 1/4, followed by
        // three stride-2 stages for C2/C3/C4 at 1/8, 1/16 and 1/32.
        int h2 = ConvOutput(inputHeight, 3, 2, 1);
        int w2 = ConvOutput(inputWidth, 3, 2, 1);
        _cnnStem1 = CreateConv(inputChannels, cnnChannels[0], 3, 2, 1, inputHeight, inputWidth, gelu);
        _cnnStem2 = CreateConv(cnnChannels[0], cnnChannels[0], 3, 2, 1, h2, w2, gelu);
        RegisterSubLayer(_cnnStem1);
        RegisterSubLayer(_cnnStem2);

        _levelHeights[0] = ConvOutput(h2, 3, 2, 1);
        _levelWidths[0] = ConvOutput(w2, 3, 2, 1);
        _cnnDownsamples = new ConvolutionalLayer<T>[3];
        for (int i = 1; i < 4; i++)
        {
            _cnnDownsamples[i - 1] = CreateConv(
                cnnChannels[i - 1], cnnChannels[i], 3, 2, 1,
                _levelHeights[i - 1], _levelWidths[i - 1], gelu);
            RegisterSubLayer(_cnnDownsamples[i - 1]);
            _levelHeights[i] = ConvOutput(_levelHeights[i - 1], 3, 2, 1);
            _levelWidths[i] = ConvOutput(_levelWidths[i - 1], 3, 2, 1);
        }

        // Plain ViT patch embedding stays isotropic at 1/16 for all four interaction stages.
        _vitPatchEmbedding = CreateConv(
            inputChannels, embedDim, 16, 16, 0,
            inputHeight, inputWidth, identity);
        RegisterSubLayer(_vitPatchEmbedding);

        _vitStages = new TransformerEncoderBlock<T>[4][];
        for (int stage = 0; stage < 4; stage++)
        {
            _vitStages[stage] = new TransformerEncoderBlock<T>[depths[stage]];
            for (int block = 0; block < depths[stage]; block++)
            {
                var transformer = new TransformerEncoderBlock<T>(
                    embedDim, _numHeads, embedDim * 4, dropRate, gelu);
                // Materialize the block's lazy FFN at a one-token sequence. Parameters are
                // sequence-length independent and now serialize before any image warm-up.
                transformer.SetTrainingMode(false);
                _ = transformer.Forward(new Tensor<T>([1, 1, embedDim]));
                transformer.ResetState();
                _vitStages[stage][block] = transformer;
                RegisterSubLayer(transformer);
            }
        }

        _mrfp3 = new ConvolutionalLayer<T>[3];
        _mrfp5 = new ConvolutionalLayer<T>[3];
        _cnnToVit = new ConvolutionalLayer<T>[3];
        for (int level = 1; level < 4; level++)
        {
            int index = level - 1;
            int channels = cnnChannels[level];
            _mrfp3[index] = CreateConv(
                channels, channels, 3, 1, 1,
                _levelHeights[level], _levelWidths[level], gelu, groups: channels);
            _mrfp5[index] = CreateConv(
                channels, channels, 5, 1, 2,
                _levelHeights[level], _levelWidths[level], gelu, groups: channels);
            _cnnToVit[index] = CreateConv(
                channels, embedDim, 1, 1, 0,
                _levelHeights[level], _levelWidths[level], identity);
            RegisterSubLayer(_mrfp3[index]);
            RegisterSubLayer(_mrfp5[index]);
            RegisterSubLayer(_cnnToVit[index]);
        }

        int vitH = ConvOutput(inputHeight, 16, 16, 0);
        int vitW = ConvOutput(inputWidth, 16, 16, 0);
        _vitToCnn = new ConvolutionalLayer<T>[4];
        for (int level = 0; level < 4; level++)
        {
            _vitToCnn[level] = CreateConv(
                embedDim, cnnChannels[level], 1, 1, 0,
                _levelHeights[level], _levelWidths[level], identity);
            RegisterSubLayer(_vitToCnn[level]);
        }

        int contextLength = _levelHeights[1] * _levelWidths[1]
            + _levelHeights[2] * _levelWidths[2]
            + _levelHeights[3] * _levelWidths[3];
        _ctiToVit = new CrossAttentionLayer<T>[4];
        _ctiToCnn = new CrossAttentionLayer<T>[4];
        for (int stage = 0; stage < 4; stage++)
        {
            _ctiToVit[stage] = new CrossAttentionLayer<T>(embedDim, embedDim, _numHeads, vitH * vitW);
            _ctiToCnn[stage] = new CrossAttentionLayer<T>(embedDim, embedDim, _numHeads, contextLength);
            RegisterSubLayer(_ctiToVit[stage]);
            RegisterSubLayer(_ctiToCnn[stage]);
        }

        _decoderProjections = new ConvolutionalLayer<T>[4];
        for (int level = 0; level < 4; level++)
        {
            _decoderProjections[level] = CreateConv(
                cnnChannels[level], decoderDim, 1, 1, 0,
                _levelHeights[level], _levelWidths[level], gelu);
            RegisterSubLayer(_decoderProjections[level]);
        }
        _decoderFusion = CreateConv(
            decoderDim, decoderDim, 3, 1, 1,
            _levelHeights[0], _levelWidths[0], gelu);
        _classifier = CreateConv(
            decoderDim, numClasses, 1, 1, 0,
            _levelHeights[0], _levelWidths[0], identity);
        RegisterSubLayer(_decoderFusion);
        RegisterSubLayer(_classifier);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        if (!unbatched && input.Rank != 4)
            throw new ArgumentException("ViTCoMerSegmentationLayer requires [B,C,H,W] input.", nameof(input));
        int channelAxis = unbatched ? 0 : 1;
        int heightAxis = unbatched ? 1 : 2;
        int widthAxis = unbatched ? 2 : 3;
        if (input.Shape[channelAxis] != _inputChannels
            || input.Shape[heightAxis] != _inputHeight
            || input.Shape[widthAxis] != _inputWidth)
        {
            throw new ArgumentException(
                $"Expected [B,{_inputChannels},{_inputHeight},{_inputWidth}], got [{string.Join(",", input.Shape)}].",
                nameof(input));
        }
        if (unbatched)
            input = Engine.Reshape(input, new[] { 1, _inputChannels, _inputHeight, _inputWidth });

        var cnn = new Tensor<T>[4];
        cnn[0] = _cnnStem2.Forward(_cnnStem1.Forward(input));
        for (int level = 1; level < 4; level++)
            cnn[level] = _cnnDownsamples[level - 1].Forward(cnn[level - 1]);

        var vitMap = _vitPatchEmbedding.Forward(input);
        var vitTokens = ToTokens(vitMap);

        for (int stage = 0; stage < 4; stage++)
        {
            // MRFP: split 3x3/5x5 depthwise receptive fields and retain the spatial-prior residual.
            for (int level = 1; level < 4; level++)
            {
                int index = level - 1;
                var local3 = _mrfp3[index].Forward(cnn[level]);
                var local5 = _mrfp5[index].Forward(cnn[level]);
                var refined = Engine.TensorMultiplyScalar(
                    Engine.TensorAdd(local3, local5), NumOps.FromDouble(0.5));
                cnn[level] = Engine.TensorAdd(cnn[level], refined);
            }

            // CTI_toV: ViT queries attend to the concatenated multi-scale CNN context.
            var cnnContext = BuildCnnContext(cnn);
            var toVit = _ctiToVit[stage].Forward(vitTokens, cnnContext);
            vitTokens = Engine.TensorAdd(vitTokens, toVit);

            foreach (var transformer in _vitStages[stage])
                vitTokens = transformer.Forward(vitTokens);

            // CTI_toC: multi-scale CNN queries attend back to the global ViT representation.
            var toCnn = _ctiToCnn[stage].Forward(cnnContext, vitTokens);
            var enhancedContext = Engine.TensorAdd(cnnContext, toCnn);
            ScatterContextToCnn(enhancedContext, vitTokens, cnn);
        }

        // Dense prediction head: project and align every fused pyramid level at C1 (1/4) scale.
        Tensor<T>? fused = null;
        for (int level = 0; level < 4; level++)
        {
            var projected = _decoderProjections[level].Forward(cnn[level]);
            projected = Resize(projected, _levelHeights[0], _levelWidths[0]);
            fused = fused is null ? projected : Engine.TensorAdd(fused, projected);
        }
        if (fused is null) throw new InvalidOperationException("ViT-CoMer pyramid is empty.");
        fused = Engine.TensorMultiplyScalar(fused, NumOps.FromDouble(0.25));
        var logits = _classifier.Forward(_decoderFusion.Forward(fused));
        return unbatched
            ? Engine.Reshape(logits, new[] { _numClasses, _levelHeights[0], _levelWidths[0] })
            : logits;
    }

    private Tensor<T> BuildCnnContext(Tensor<T>[] cnn)
    {
        var tokenLevels = new Tensor<T>[3];
        for (int level = 1; level < 4; level++)
            tokenLevels[level - 1] = ToTokens(_cnnToVit[level - 1].Forward(cnn[level]));
        return Engine.TensorConcatenate(tokenLevels, axis: 1);
    }

    private void ScatterContextToCnn(Tensor<T> context, Tensor<T> vitTokens, Tensor<T>[] cnn)
    {
        int offset = 0;
        for (int level = 1; level < 4; level++)
        {
            int count = _levelHeights[level] * _levelWidths[level];
            var tokens = Engine.TensorSlice(
                context,
                new[] { 0, offset, 0 },
                new[] { context.Shape[0], count, _embedDim });
            var map = TokensToMap(tokens, _levelHeights[level], _levelWidths[level]);
            cnn[level] = Engine.TensorAdd(cnn[level], _vitToCnn[level].Forward(map));
            offset += count;
        }

        // C1 is not part of MRFP's C2-C4 token sequence; inject the ViT feature directly at 1/4.
        int vitH = _inputHeight / 16;
        int vitW = _inputWidth / 16;
        var c1Map = Resize(TokensToMap(vitTokens, vitH, vitW), _levelHeights[0], _levelWidths[0]);
        cnn[0] = Engine.TensorAdd(cnn[0], _vitToCnn[0].Forward(c1Map));
    }

    private Tensor<T> ToTokens(Tensor<T> map)
    {
        var permuted = Engine.TensorPermute(map, new[] { 0, 2, 3, 1 });
        return Engine.Reshape(permuted, new[] { map.Shape[0], map.Shape[2] * map.Shape[3], map.Shape[1] });
    }

    private Tensor<T> TokensToMap(Tensor<T> tokens, int height, int width)
    {
        var bhwc = Engine.Reshape(tokens, new[] { tokens.Shape[0], height, width, _embedDim });
        return Engine.TensorPermute(bhwc, new[] { 0, 3, 1, 2 });
    }

    private Tensor<T> Resize(Tensor<T> map, int height, int width)
    {
        if (map.Shape[2] == height && map.Shape[3] == width) return map;
        if (height % map.Shape[2] != 0 || width % map.Shape[3] != 0)
            throw new InvalidOperationException("ViT-CoMer pyramid levels must align by integer scale factors.");

        // RepeatInterleave is tape-aware and gives the paper decoder's spatial alignment without
        // passing a non-contiguous token-permute view into the current Interpolate kernel.
        // The scalar multiply materializes that view through a recorded identity operation first.
        var contiguousGraphValue = Engine.TensorMultiplyScalar(map, NumOps.One);
        int heightScale = height / map.Shape[2];
        int widthScale = width / map.Shape[3];
        var resized = heightScale == 1
            ? contiguousGraphValue
            : Engine.TensorRepeatInterleave(contiguousGraphValue, heightScale, dim: 2);
        return widthScale == 1
            ? resized
            : Engine.TensorRepeatInterleave(resized, widthScale, dim: 3);
    }

    private static int ConvOutput(int size, int kernel, int stride, int padding)
        => (size + 2 * padding - kernel) / stride + 1;

    private static int ResolveHeadCount(int embedDim)
    {
        int preferred = Math.Max(1, embedDim / 64);
        while (preferred > 1 && embedDim % preferred != 0) preferred--;
        return preferred;
    }

    private static ConvolutionalLayer<T> CreateConv(
        int inputChannels,
        int outputChannels,
        int kernel,
        int stride,
        int padding,
        int inputHeight,
        int inputWidth,
        IActivationFunction<T> activation,
        int groups = 1)
    {
        var layer = new ConvolutionalLayer<T>(
            outputChannels, kernel, stride, padding, activation, groups: groups);
        layer.ResolveFromShape(new[] { inputChannels, inputHeight, inputWidth });
        return layer;
    }

    private IEnumerable<ILayer<T>> OrderedLayers()
    {
        yield return _cnnStem1;
        yield return _cnnStem2;
        foreach (var layer in _cnnDownsamples) yield return layer;
        yield return _vitPatchEmbedding;
        foreach (var stage in _vitStages)
        foreach (var layer in stage) yield return layer;
        foreach (var layer in _mrfp3) yield return layer;
        foreach (var layer in _mrfp5) yield return layer;
        foreach (var layer in _cnnToVit) yield return layer;
        foreach (var layer in _vitToCnn) yield return layer;
        foreach (var layer in _ctiToVit) yield return layer;
        foreach (var layer in _ctiToCnn) yield return layer;
        foreach (var layer in _decoderProjections) yield return layer;
        yield return _decoderFusion;
        yield return _classifier;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var values = new List<T>((int)ParameterCount);
        foreach (var layer in OrderedLayers())
        {
            var gradients = layer.GetParameterGradients();
            for (int i = 0; i < gradients.Length; i++) values.Add(gradients[i]);
        }
        return new Vector<T>(values.ToArray());
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in OrderedLayers()) layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in OrderedLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in OrderedLayers()) layer.ResetState();
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputChannels"] = _inputChannels.ToString(ci);
        metadata["InputHeight"] = _inputHeight.ToString(ci);
        metadata["InputWidth"] = _inputWidth.ToString(ci);
        metadata["EmbedDim"] = _embedDim.ToString(ci);
        metadata["CnnChannels"] = string.Join(",", _cnnChannels);
        metadata["Depths"] = string.Join(",", _depths);
        metadata["DecoderDim"] = _decoderDim.ToString(ci);
        metadata["NumClasses"] = _numClasses.ToString(ci);
        metadata["DropRate"] = _dropRate.ToString("R", ci);
        return metadata;
    }
}
