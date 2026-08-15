using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the Spatial-Temporal Convolution (STC) connector from VideoLLaMA 2
/// (Cheng et al. 2024, arXiv:2406.07476).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The paper's connector is not a lone 3D convolution. Its published structure is
/// <c>RegStage -&gt; Conv3D downsampler -&gt; RegStage -&gt; MLP</c>. Each RegStage uses
/// depthwise RegNet-style bottleneck blocks with LayerNorm2d and SiLU. The default
/// constructor values reproduce the released connector: stage depth 4, a
/// <c>2 x 2 x 2</c> convolution with stride 2 and padding 1, and a two-layer GELU MLP.
/// </para>
/// <para>
/// Token input may be unbatched <c>[frames, Hp*Wp, visionDim]</c>, batched
/// <c>[batch, frames, Hp*Wp, visionDim]</c>, or an already-expanded grid
/// <c>[batch, frames, Hp, Wp, visionDim]</c>. The output is the flattened visual-token
/// sequence in the language decoder width.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "4, 4, 8", TestConstructorArgs = "8, 2, 2")]
// The three accepted forms are ForwardTraced's own rank switch, and nothing else is accepted (rank 6+
// and rank 2 throw "expects [T,L,C], [B,T,L,C], or [B,T,H,W,C] input"):
//   rank 3  [T, Hp*Wp, C]     - unbatched tokens
//   rank 4  [B, T, Hp*Wp, C]  - batched tokens
//   rank 5  [B, T, Hp, Wp, C] - already-expanded grid
// The video axis is Frames, which is what it literally is here and what TensorAxis.Frames is for
// ("video frames, when frames are a separate axis from Time or Batch"). The token form's middle axis is
// the patch grid ALREADY FLATTENED into one dimension, so it is Length - a sequence position with no
// temporal meaning - rather than Height or Width, neither of which is separately addressable there.
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Length, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input,
    Note = "Token form; the Length axis is the flattened Hp*Wp patch grid.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Height, TensorAxis.Width, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input, Note = "Expanded grid form.")]
// Output is the flattened visual-token sequence, [B, T'*H'*W', decoderDim], reshaped down to
// [T'*H'*W', decoderDim] when the input had no explicit batch. Time, not Length: this is the sequence
// the language decoder consumes, and [Batch, Time, Features] is what every transformer block in this
// codebase declares as its input, so naming it anything else would break chain validation at the
// connector's only real call site.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class STCConnectorLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Derived from the last three statements of <c>ForwardTraced</c>: the sampler's result is read as
    /// <c>outFrames = sampled.Shape[2]</c>, <c>outHeight = sampled.Shape[3]</c>,
    /// <c>outWidth = sampled.Shape[4]</c>, those are flattened by
    /// <c>Engine.Reshape(..., new[] { batch, outFrames * outHeight * outWidth, _decoderDim })</c>, and
    /// the readout MLP is <see cref="DenseLayer{T}"/>s built at <c>_decoderDim</c>, which change only
    /// the last axis. The RegStage blocks on either side are spatially neutral (1x1, depthwise 3x3 at
    /// stride 1 padding 1, 1x1, plus a residual add that could not otherwise line up), so they do not
    /// enter the contract either.
    /// </para>
    /// <para>
    /// THE TOKEN AXIS IS A PRODUCT OF THREE SAMPLED EXTENTS. <c>_sampler</c> is
    /// <c>new Conv3DLayer&lt;T&gt;(decoderDim, kernelSize, stride, padding, SiLU)</c> applied to
    /// <c>[B, C, T, Hp, Wp]</c>, and <c>Conv3DLayer.CalculateOutputShape</c> sizes every one of those
    /// three as <c>(in + 2*padding - kernel) / stride + 1</c> - the plain window formula. So each
    /// factor is a <see cref="AxisRelation.Window"/> and the token count is their product, which is
    /// what <see cref="AxisRelation.ProductOf"/> is for; <see cref="AxisRelation.Product"/> multiplies
    /// RAW input axes and cannot express the downsampling that happens first.
    /// </para>
    /// <para>
    /// WHY THE SPATIAL FACTORS ARE FIXED IN THE TOKEN FORMS. At ranks 3 and 4 the patch grid arrives
    /// pre-flattened and <c>ValidateTokenShape</c> REQUIRES it to equal
    /// <c>_patchesHeight * _patchesWidth</c>, so the pre-sampling spatial extents are the constructor's
    /// values rather than anything read off the input. Their sampled sizes are therefore computed here
    /// from <c>_patchesHeight</c>/<c>_patchesWidth</c> with the sampler's own formula - configuration,
    /// not an observed literal. At rank 5 the same two extents ARE addressable axes (and are checked
    /// equal to the same fields), so the honest form there is the window on the real axes.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_decoderDim <= 0 || _kernelSize <= 0 || _stride <= 0 || _padding < 0) return null;

        // Conv3DLayer.CalculateOutputShape: (in + 2 * padding - kernelSize) / stride + 1.
        int Sampled(int extent) => (extent + (2 * _padding) - _kernelSize) / _stride + 1;

        // The frame axis is the only sampled extent that varies with the input in every accepted form.
        var frames = AxisRelation.Window(
            TensorAxis.Frames, kernel: _kernelSize, stride: _stride, padding: _padding);

        AxisRelation tokenCount;
        switch (inputRank)
        {
            case 3:
            case 4:
            {
                int outHeight = Sampled(_patchesHeight);
                int outWidth = Sampled(_patchesWidth);
                // Matches Conv3DLayer's own guard, which throws when a sampled extent collapses.
                if (outHeight <= 0 || outWidth <= 0) return null;
                tokenCount = AxisRelation.ProductOf(
                    frames, AxisRelation.Fixed(outHeight), AxisRelation.Fixed(outWidth));
                break;
            }

            case 5:
                tokenCount = AxisRelation.ProductOf(
                    frames,
                    AxisRelation.Window(
                        TensorAxis.Height, kernel: _kernelSize, stride: _stride, padding: _padding),
                    AxisRelation.Window(
                        TensorAxis.Width, kernel: _kernelSize, stride: _stride, padding: _padding));
                break;

            default:
                return null;
        }

        var tokens = new OutputAxisContract(TensorAxis.Time, tokenCount);
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_decoderDim));

        // hadExplicitBatch is false only for the rank-3 form, which is reshaped back to two axes.
        return inputRank == 3
            ? new[] { tokens, features }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                tokens,
                features,
            };
    }

    private readonly int _visionDim;
    private readonly int _decoderDim;
    private readonly int _patchesHeight;
    private readonly int _patchesWidth;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _stageDepth;
    private readonly int _mlpDepth;

    private readonly RegStageBlock[] _stage1;
    private readonly Conv3DLayer<T> _sampler;
    private readonly RegStageBlock[] _stage2;
    private readonly DenseLayer<T>[] _readout;
    private readonly LayerBase<T>[] _parameterLayers;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a connector that preserves the input/output feature width.
    /// </summary>
    /// <remarks>
    /// This overload preserves source compatibility with the original AiDotNet connector API.
    /// New code that connects a vision tower to a differently-sized language decoder should use
    /// the overload that accepts both <c>visionDim</c> and <c>decoderDim</c>.
    /// </remarks>
    public STCConnectorLayer(
        int dim,
        int patchesHeight,
        int patchesWidth,
        int kernelSize = 2,
        int stride = 2,
        int padding = 1)
        : this(dim, dim, patchesHeight, patchesWidth, kernelSize, stride, padding, stageDepth: 4, mlpDepth: 2)
    {
    }

    /// <summary>
    /// Creates the paper's complete STC connector.
    /// </summary>
    /// <param name="visionDim">Feature width emitted by the vision encoder.</param>
    /// <param name="decoderDim">Feature width consumed by the language decoder.</param>
    /// <param name="patchesHeight">Vision patch-grid height.</param>
    /// <param name="patchesWidth">Vision patch-grid width.</param>
    /// <param name="kernelSize">Uniform temporal/spatial Conv3D kernel size. Paper default: 2.</param>
    /// <param name="stride">Uniform temporal/spatial Conv3D stride. Paper default: 2.</param>
    /// <param name="padding">Uniform temporal/spatial Conv3D padding. Released connector default: 1.</param>
    /// <param name="stageDepth">Number of RegNet bottleneck blocks in each spatial stage. Paper default: 4.</param>
    /// <param name="mlpDepth">Number of linear layers in the readout MLP. Paper default: 2.</param>
    public STCConnectorLayer(
        [LayerState] int visionDim,
        [LayerState] int decoderDim,
        [LayerState] int patchesHeight,
        [LayerState] int patchesWidth,
        [LayerState] int kernelSize = 2,
        [LayerState] int stride = 2,
        [LayerState] int padding = 1,
        [LayerState] int stageDepth = 4,
        [LayerState] int mlpDepth = 2)
        : base(
            new[] { -1, patchesHeight * patchesWidth, visionDim },
            new[] { -1, decoderDim })
    {
        if (visionDim <= 0) throw new ArgumentOutOfRangeException(nameof(visionDim));
        if (decoderDim <= 0) throw new ArgumentOutOfRangeException(nameof(decoderDim));
        if (patchesHeight <= 0) throw new ArgumentOutOfRangeException(nameof(patchesHeight));
        if (patchesWidth <= 0) throw new ArgumentOutOfRangeException(nameof(patchesWidth));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        if (stageDepth < 0) throw new ArgumentOutOfRangeException(nameof(stageDepth));
        if (mlpDepth <= 0) throw new ArgumentOutOfRangeException(nameof(mlpDepth));

        _visionDim = visionDim;
        _decoderDim = decoderDim;
        _patchesHeight = patchesHeight;
        _patchesWidth = patchesWidth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _stageDepth = stageDepth;
        _mlpDepth = mlpDepth;

        _stage1 = new RegStageBlock[stageDepth];
        for (int i = 0; i < stageDepth; i++)
        {
            int inputChannels = i == 0 ? visionDim : decoderDim;
            _stage1[i] = new RegStageBlock(inputChannels, decoderDim);
        }

        _sampler = new Conv3DLayer<T>(
            decoderDim,
            kernelSize,
            stride,
            padding,
            (IActivationFunction<T>)new SiLUActivation<T>());

        _stage2 = new RegStageBlock[stageDepth];
        for (int i = 0; i < stageDepth; i++)
            _stage2[i] = new RegStageBlock(decoderDim, decoderDim);

        _readout = new DenseLayer<T>[mlpDepth];
        for (int i = 0; i < mlpDepth; i++)
        {
            IActivationFunction<T> activation = i < mlpDepth - 1
                ? new GELUActivation<T>()
                : new IdentityActivation<T>();
            _readout[i] = new DenseLayer<T>(decoderDim, activation);
        }

        var parameterLayers = new List<LayerBase<T>>(stageDepth * 2 + mlpDepth + 1);
        parameterLayers.AddRange(_stage1);
        parameterLayers.Add(_sampler);
        parameterLayers.AddRange(_stage2);
        parameterLayers.AddRange(_readout);
        _parameterLayers = parameterLayers.ToArray();

        foreach (var layer in _parameterLayers)
            RegisterSubLayer(layer);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int rank = input.Rank;
        bool hadExplicitBatch;
        int batch;
        int frames;
        Tensor<T> grid;

        if (rank == 3)
        {
            // [T, Hp*Wp, C] -> [1, T, Hp, Wp, C]
            hadExplicitBatch = false;
            batch = 1;
            frames = input.Shape[0];
            ValidateTokenShape(input.Shape[1], input.Shape[2]);
            grid = Engine.Reshape(input,
                new[] { batch, frames, _patchesHeight, _patchesWidth, _visionDim });
        }
        else if (rank == 4)
        {
            // [B, T, Hp*Wp, C] -> [B, T, Hp, Wp, C]
            hadExplicitBatch = true;
            batch = input.Shape[0];
            frames = input.Shape[1];
            ValidateTokenShape(input.Shape[2], input.Shape[3]);
            grid = Engine.Reshape(input,
                new[] { batch, frames, _patchesHeight, _patchesWidth, _visionDim });
        }
        else if (rank == 5)
        {
            // [B, T, Hp, Wp, C]
            hadExplicitBatch = true;
            batch = input.Shape[0];
            frames = input.Shape[1];
            if (input.Shape[2] != _patchesHeight || input.Shape[3] != _patchesWidth || input.Shape[4] != _visionDim)
            {
                throw new ArgumentException(
                    $"Expected STC grid [B,T,{_patchesHeight},{_patchesWidth},{_visionDim}], " +
                    $"got [{string.Join(",", input.Shape)}].",
                    nameof(input));
            }
            grid = input;
        }
        else
        {
            throw new ArgumentException(
                "STCConnectorLayer expects [T,L,C], [B,T,L,C], or [B,T,H,W,C] input.",
                nameof(input));
        }

        // [B,T,H,W,C] -> [B*T,C,H,W], matching the paper's per-frame RegStage.
        var framesNchw = Engine.Reshape(
            Engine.TensorPermute(grid, new[] { 0, 1, 4, 2, 3 }),
            new[] { batch * frames, _visionDim, _patchesHeight, _patchesWidth });

        var stage1Output = framesNchw;
        foreach (var block in _stage1)
            stage1Output = block.Forward(stage1Output);

        // [B*T,C,H,W] -> [B,C,T,H,W] for the joint spatial-temporal sampler.
        var volume = Engine.TensorPermute(
            Engine.Reshape(stage1Output,
                new[] { batch, frames, _decoderDim, _patchesHeight, _patchesWidth }),
            new[] { 0, 2, 1, 3, 4 });
        var sampled = _sampler.Forward(volume);

        int outFrames = sampled.Shape[2];
        int outHeight = sampled.Shape[3];
        int outWidth = sampled.Shape[4];

        // Apply the second RegStage independently to every downsampled frame.
        var sampledFrames = Engine.Reshape(
            Engine.TensorPermute(sampled, new[] { 0, 2, 1, 3, 4 }),
            new[] { batch * outFrames, _decoderDim, outHeight, outWidth });
        var stage2Output = sampledFrames;
        foreach (var block in _stage2)
            stage2Output = block.Forward(stage2Output);

        // [B*T',C,H',W'] -> [B,T'*H'*W',C], then the published MLP readout.
        var tokens = Engine.Reshape(
            Engine.TensorPermute(
                Engine.Reshape(stage2Output,
                    new[] { batch, outFrames, _decoderDim, outHeight, outWidth }),
                new[] { 0, 1, 3, 4, 2 }),
            new[] { batch, outFrames * outHeight * outWidth, _decoderDim });

        var output = tokens;
        foreach (var layer in _readout)
            output = layer.Forward(output);

        return hadExplicitBatch
            ? output
            : Engine.Reshape(output, new[] { output.Shape[1], output.Shape[2] });
    }

    private void ValidateTokenShape(int tokenCount, int featureDim)
    {
        int expectedTokens = _patchesHeight * _patchesWidth;
        if (tokenCount != expectedTokens || featureDim != _visionDim)
        {
            throw new ArgumentException(
                $"STCConnectorLayer requires token shape [...,{expectedTokens},{_visionDim}] " +
                $"(height*width={_patchesHeight}*{_patchesWidth} tokens and width={_visionDim} features); " +
                $"got [...,{tokenCount},{featureDim}].");
        }
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in _parameterLayers)
            layer.SetTrainingMode(isTraining);
    }

    private void MaterializeSublayers()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            int frames = Math.Max(1, _kernelSize - 2 * _padding);
            _ = Forward(new Tensor<T>(new[]
            {
                frames,
                _patchesHeight * _patchesWidth,
                _visionDim
            }));
            ResetState();
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => Concatenate(_parameterLayers, gradients: true);

    private static Vector<T> Concatenate(IEnumerable<LayerBase<T>> layers, bool gradients)
    {
        var vectors = layers
            .Select(layer => gradients ? layer.GetParameterGradients() : layer.GetParameters())
            .ToArray();
        int total = checked(vectors.Sum(vector => vector.Length));
        var result = new Vector<T>(total);
        int offset = 0;
        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
                result[offset++] = vector[i];
        }
        return result;
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in _parameterLayers)
            layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _parameterLayers)
            layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in _parameterLayers)
            layer.ResetState();
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Dim"] = _visionDim.ToString(ci); // legacy key
        metadata["VisionDim"] = _visionDim.ToString(ci);
        metadata["DecoderDim"] = _decoderDim.ToString(ci);
        metadata["PatchesHeight"] = _patchesHeight.ToString(ci);
        metadata["PatchesWidth"] = _patchesWidth.ToString(ci);
        metadata["KernelSize"] = _kernelSize.ToString(ci);
        metadata["Stride"] = _stride.ToString(ci);
        metadata["Padding"] = _padding.ToString(ci);
        metadata["StageDepth"] = _stageDepth.ToString(ci);
        metadata["MlpDepth"] = _mlpDepth.ToString(ci);
        return metadata;
    }

    /// <summary>
    /// RegNet bottleneck used by the official connector's two spatial-interaction stages.
    /// It follows timm's default RegStage bottleneck ratio/group-width behavior: 1x1 projection,
    /// depthwise 3x3 spatial convolution, 1x1 projection, LayerNorm2d, SiLU, and a residual path.
    /// </summary>
    // Rank 4 only, and batch is NOT optional: ForwardTraced opens with
    // "if (input.Rank != 4 || input.Shape[1] != _inputChannels) throw", so [C,H,W] is rejected outright
    // and axis 1 is unambiguously Channels.
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        Direction = TensorLayoutDirection.Output)]
    private sealed partial class RegStageBlock : LayerBase<T>, IShapeContract
    {
        /// <inheritdoc />
        /// <remarks>
        /// <para>
        /// A RegNet bottleneck is spatially neutral by construction, and this one is written that way:
        /// <c>_conv1</c> and <c>_conv3</c> are <c>(outputChannels, 1, 1, 0)</c>, <c>_conv2</c> is
        /// <c>(outputChannels, 3, 1, 1, groups: outputChannels)</c> - kernel 3, stride 1, padding 1,
        /// which leaves H and W unchanged - and the block ends in
        /// <c>Engine.TensorAdd(output, residual)</c>, which could not line up at all if any of them
        /// resized the spatial axes. So Height and Width are <see cref="AxisRelation.Same"/> rather
        /// than windows.
        /// </para>
        /// <para>
        /// Channels is the only axis that moves: every convolution on both the main and the shortcut
        /// path is built with <c>outputChannels</c> filters, so the block emits <c>_outputChannels</c>
        /// whatever it was fed - which is exactly why the shortcut carries its own 1x1 projection when
        /// <c>inputChannels != outputChannels</c>. The interleaved
        /// <see cref="LayerNormalizationLayer{T}"/>s and <see cref="ActivationLayer{T}"/>s preserve
        /// shape and do not enter the contract.
        /// </para>
        /// </remarks>
        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        {
            if (inputRank != 4 || _outputChannels <= 0) return null;

            return new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outputChannels)),
                new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
                new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
            };
        }

        private readonly int _inputChannels;
        private readonly int _outputChannels;
        private readonly ConvolutionalLayer<T> _conv1;
        private readonly LayerNormalizationLayer<T> _norm1;
        private readonly ActivationLayer<T> _act1;
        private readonly ConvolutionalLayer<T> _conv2;
        private readonly LayerNormalizationLayer<T> _norm2;
        private readonly ActivationLayer<T> _act2;
        private readonly ConvolutionalLayer<T> _conv3;
        private readonly LayerNormalizationLayer<T> _norm3;
        private readonly ConvolutionalLayer<T>? _shortcutConv;
        private readonly LayerNormalizationLayer<T>? _shortcutNorm;
        private readonly ActivationLayer<T> _outputActivation;
        private readonly LayerBase<T>[] _parameterLayers;
        private readonly LayerBase<T>[] _allLayers;

        public RegStageBlock(int inputChannels, int outputChannels)
            : base(new[] { inputChannels, -1, -1 }, new[] { outputChannels, -1, -1 })
        {
            _inputChannels = inputChannels;
            _outputChannels = outputChannels;
            var identity = (IActivationFunction<T>)new IdentityActivation<T>();

            _conv1 = new ConvolutionalLayer<T>(outputChannels, 1, 1, 0, identity);
            _norm1 = new LayerNormalizationLayer<T>(outputChannels, epsilon: 1e-6);
            _act1 = new ActivationLayer<T>((IActivationFunction<T>)new SiLUActivation<T>());
            _conv2 = new ConvolutionalLayer<T>(
                outputChannels, 3, 1, 1, identity, groups: outputChannels);
            _norm2 = new LayerNormalizationLayer<T>(outputChannels, epsilon: 1e-6);
            _act2 = new ActivationLayer<T>((IActivationFunction<T>)new SiLUActivation<T>());
            _conv3 = new ConvolutionalLayer<T>(outputChannels, 1, 1, 0, identity);
            _norm3 = new LayerNormalizationLayer<T>(outputChannels, epsilon: 1e-6);
            _outputActivation = new ActivationLayer<T>((IActivationFunction<T>)new SiLUActivation<T>());

            if (inputChannels != outputChannels)
            {
                _shortcutConv = new ConvolutionalLayer<T>(outputChannels, 1, 1, 0, identity);
                _shortcutNorm = new LayerNormalizationLayer<T>(outputChannels, epsilon: 1e-6);
            }

            var parameterLayers = new List<LayerBase<T>>
            {
                _conv1, _norm1, _conv2, _norm2, _conv3, _norm3
            };
            if (_shortcutConv is not null) parameterLayers.Add(_shortcutConv);
            if (_shortcutNorm is not null) parameterLayers.Add(_shortcutNorm);
            _parameterLayers = parameterLayers.ToArray();

            var allLayers = new List<LayerBase<T>>
            {
                _conv1, _norm1, _act1, _conv2, _norm2, _act2, _conv3, _norm3
            };
            if (_shortcutConv is not null) allLayers.Add(_shortcutConv);
            if (_shortcutNorm is not null) allLayers.Add(_shortcutNorm);
            allLayers.Add(_outputActivation);
            _allLayers = allLayers.ToArray();

            foreach (var layer in _allLayers)
                RegisterSubLayer(layer);
        }

        public override bool SupportsTraining => true;

        protected override Tensor<T> ForwardTraced(Tensor<T> input)
        {
            if (input.Rank != 4 || input.Shape[1] != _inputChannels)
            {
                throw new ArgumentException(
                    $"RegStageBlock expects [B,{_inputChannels},H,W], got [{string.Join(",", input.Shape)}].",
                    nameof(input));
            }

            var residual = input;
            if (_shortcutConv is not null && _shortcutNorm is not null)
                residual = NormalizeChannels(_shortcutNorm, _shortcutConv.Forward(input));

            var output = _conv1.Forward(input);
            output = _act1.Forward(NormalizeChannels(_norm1, output));
            output = _conv2.Forward(output);
            output = _act2.Forward(NormalizeChannels(_norm2, output));
            output = NormalizeChannels(_norm3, _conv3.Forward(output));
            return _outputActivation.Forward(Engine.TensorAdd(output, residual));
        }

        private Tensor<T> NormalizeChannels(LayerNormalizationLayer<T> norm, Tensor<T> input)
        {
            var channelsLast = Engine.TensorPermute(input, new[] { 0, 2, 3, 1 });
            var normalized = norm.Forward(channelsLast);
            return Engine.TensorPermute(normalized, new[] { 0, 3, 1, 2 });
        }

        public override void SetTrainingMode(bool isTraining)
        {
            base.SetTrainingMode(isTraining);
            foreach (var layer in _allLayers)
                layer.SetTrainingMode(isTraining);
        }

        public override Vector<T> GetParameterGradients() => Concatenate(_parameterLayers, gradients: true);

        public override void ClearGradients()
        {
            base.ClearGradients();
            foreach (var layer in _parameterLayers)
                layer.ClearGradients();
        }

        public override void UpdateParameters(T learningRate)
        {
            foreach (var layer in _parameterLayers)
                layer.UpdateParameters(learningRate);
        }

        public override void ResetState()
        {
            foreach (var layer in _allLayers)
                layer.ResetState();
        }
    }
}
