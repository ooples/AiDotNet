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
public class STCConnectorLayer<T> : LayerBase<T>
{
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
        int visionDim,
        int decoderDim,
        int patchesHeight,
        int patchesWidth,
        int kernelSize = 2,
        int stride = 2,
        int padding = 1,
        int stageDepth = 4,
        int mlpDepth = 2)
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
    public override Tensor<T> Forward(Tensor<T> input)
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
                $"Expected STC tokens [...,{expectedTokens},{_visionDim}], got [...,{tokenCount},{featureDim}].");
        }
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in _parameterLayers)
            layer.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override long ParameterCount => _parameterLayers.Sum(layer => layer.ParameterCount);

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => Concatenate(_parameterLayers, gradients: false);

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Every convolution/dense sublayer is lazy. Reconstruct its concrete shapes before slicing
        // a serialized trained vector into the newly-created connector.
        if (_parameterLayers.Sum(layer => (long)layer.GetParameters().Length) != parameters.Length)
            MaterializeSublayers();

        int offset = 0;
        foreach (var layer in _parameterLayers)
        {
            int count = layer.GetParameters().Length;
            if (count == 0) continue;
            if (offset + count > parameters.Length)
                throw new ArgumentException("STCConnectorLayer parameter vector is shorter than its reconstructed structure.", nameof(parameters));
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }

        if (offset != parameters.Length)
            throw new ArgumentException($"Expected {offset} STC parameters, got {parameters.Length}.", nameof(parameters));
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
    private sealed class RegStageBlock : LayerBase<T>
    {
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

        public override Tensor<T> Forward(Tensor<T> input)
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

        public override long ParameterCount => _parameterLayers.Sum(layer => layer.ParameterCount);

        public override Vector<T> GetParameters() => Concatenate(_parameterLayers, gradients: false);

        public override void SetParameters(Vector<T> parameters)
        {
            int expected = _parameterLayers.Sum(layer => layer.GetParameters().Length);
            if (parameters.Length != expected)
                throw new ArgumentException($"Expected {expected} RegStage parameters, got {parameters.Length}.", nameof(parameters));

            int offset = 0;
            foreach (var layer in _parameterLayers)
            {
                int count = layer.GetParameters().Length;
                if (count == 0) continue;
                layer.SetParameters(parameters.Slice(offset, count));
                offset += count;
            }
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
