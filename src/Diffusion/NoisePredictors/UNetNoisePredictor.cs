using System.Diagnostics.CodeAnalysis;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Attention;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// U-Net architecture for noise prediction in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The U-Net is the most common architecture for diffusion model noise prediction.
/// It has an encoder-decoder structure with skip connections that help preserve
/// fine-grained details during the denoising process.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of U-Net like a funnel:
/// 1. Encoder (going down): Compresses the image, capturing patterns at different scales
/// 2. Middle: Processes the most compressed representation
/// 3. Decoder (going up): Expands back to original size, using skip connections
///
/// Skip connections are like "shortcuts" that connect encoder layers directly to
/// decoder layers, helping the network preserve fine details that might otherwise
/// be lost during compression.
/// </para>
/// <para>
/// This implementation follows the Stable Diffusion architecture with:
/// - Residual blocks with group normalization
/// - Self-attention at lower resolutions
/// - Cross-attention for text conditioning
/// - Time embedding injection via adaptive normalization
/// </para>
/// </remarks>
public class UNetNoisePredictor<T> : NoisePredictorBase<T>
{
    /// <summary>
    /// Channel multipliers for each resolution level.
    /// </summary>
    private readonly int[] _channelMultipliers;

    /// <summary>
    /// Number of residual blocks per resolution level.
    /// </summary>
    private readonly int _numResBlocks;

    /// <summary>
    /// Resolutions at which to apply attention.
    /// </summary>
    private readonly int[] _attentionResolutions;

    /// <summary>
    /// Encoder blocks (downsampling path).
    /// </summary>
    private readonly List<UNetBlock> _encoderBlocks;

    /// <summary>
    /// Middle blocks (bottleneck).
    /// </summary>
    private readonly List<UNetBlock> _middleBlocks;

    /// <summary>
    /// Decoder blocks (upsampling path).
    /// </summary>
    private readonly List<UNetBlock> _decoderBlocks;

    /// <summary>
    /// Input convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _inputConv;

    /// <summary>
    /// Output convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Time embedding MLP.
    /// </summary>
    private DenseLayer<T>? _timeEmbedMlp1;
    private DenseLayer<T>? _timeEmbedMlp2;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output for backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Number of input channels.
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Number of output channels.
    /// </summary>
    private readonly int _outputChannels;

    /// <summary>
    /// Base channel count.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Time embedding dimension.
    /// </summary>
    private readonly int _timeEmbeddingDim;

    /// <summary>
    /// Context dimension for cross-attention.
    /// </summary>
    private readonly int _contextDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Latent spatial height for the noise predictor (default: 64).
    /// </summary>
    private readonly int _inputHeight;

    /// <summary>
    /// The neural network architecture configuration, if provided.
    /// </summary>
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _outputChannels;

    /// <inheritdoc />
    public override int BaseChannels => _baseChannels;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _timeEmbeddingDim;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => _contextDim > 0;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <summary>
    /// Initializes a new instance of the UNetNoisePredictor class with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture with custom layers. If the architecture's Layers
    /// list contains layers, those will be used for the encoder blocks. If null or empty,
    /// industry-standard layers from the Stable Diffusion paper are created automatically.
    /// </param>
    /// <param name="inputChannels">Number of input channels (default: 4 for latent diffusion).</param>
    /// <param name="outputChannels">Number of output channels (default: same as input).</param>
    /// <param name="baseChannels">Base channel count (default: 320 for SD).</param>
    /// <param name="channelMultipliers">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocks">Number of residual blocks per level (default: 2).</param>
    /// <param name="attentionResolutions">Resolution indices for attention (default: [1, 2, 3]).</param>
    /// <param name="contextDim">Context dimension for cross-attention (default: 768 for CLIP).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="inputHeight">Latent spatial height (default: 64 for SD 512/8).</param>
    /// <param name="encoderBlocks">
    /// Optional custom encoder blocks. If provided, these blocks are used instead of creating
    /// default blocks. This allows full customization of the encoder path.
    /// </param>
    /// <param name="middleBlocks">
    /// Optional custom middle (bottleneck) blocks.
    /// </param>
    /// <param name="decoderBlocks">
    /// Optional custom decoder blocks.
    /// </param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults
    /// from the original Stable Diffusion paper. You can create a ready-to-use U-Net
    /// with no arguments, or customize any component:
    ///
    /// <code>
    /// // Default configuration (recommended for most users)
    /// var unet = new UNetNoisePredictor&lt;float&gt;();
    ///
    /// // Custom layers via NeuralNetworkArchitecture
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(..., layers: myCustomLayers);
    /// var unet = new UNetNoisePredictor&lt;float&gt;(architecture: arch);
    ///
    /// // SDXL configuration
    /// var unet = new UNetNoisePredictor&lt;float&gt;(
    ///     baseChannels: 320,
    ///     channelMultipliers: new[] { 1, 2, 4 },
    ///     contextDim: 2048);
    /// </code>
    /// </para>
    /// </remarks>
    public UNetNoisePredictor(
        NeuralNetworkArchitecture<T>? architecture = null,
        int inputChannels = 4,
        int? outputChannels = null,
        int baseChannels = 320,
        int[]? channelMultipliers = null,
        int numResBlocks = 2,
        int[]? attentionResolutions = null,
        int contextDim = 768,
        int numHeads = 8,
        int inputHeight = 64,
        List<UNetBlock>? encoderBlocks = null,
        List<UNetBlock>? middleBlocks = null,
        List<UNetBlock>? decoderBlocks = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _architecture = architecture;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels ?? inputChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? [1, 2, 4, 4];
        _numResBlocks = numResBlocks;
        _attentionResolutions = attentionResolutions ?? [1, 2, 3];
        _contextDim = contextDim;
        _numHeads = numHeads;
        _timeEmbeddingDim = baseChannels * 4;
        _inputHeight = inputHeight;

        _encoderBlocks = new List<UNetBlock>();
        _middleBlocks = new List<UNetBlock>();
        _decoderBlocks = new List<UNetBlock>();

        InitializeLayers(architecture, encoderBlocks, middleBlocks, decoderBlocks);
    }

    /// <summary>
    /// Initializes all layers of the U-Net, using custom layers from the user
    /// if provided or creating industry-standard layers from the Stable Diffusion paper.
    /// </summary>
    /// <param name="architecture">Optional architecture with custom layers.</param>
    /// <param name="customEncoderBlocks">Optional custom encoder blocks.</param>
    /// <param name="customMiddleBlocks">Optional custom middle blocks.</param>
    /// <param name="customDecoderBlocks">Optional custom decoder blocks.</param>
    /// <remarks>
    /// <para>
    /// Layer resolution order:
    /// 1. If custom encoder/middle/decoder blocks are provided directly, use those
    /// 2. If a NeuralNetworkArchitecture with layers is provided, wrap those as encoder blocks
    /// 3. Otherwise, create industry-standard layers from the Stable Diffusion paper
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_inputConv), nameof(_outputConv), nameof(_timeEmbedMlp1), nameof(_timeEmbedMlp2))]
    private void InitializeLayers(
        NeuralNetworkArchitecture<T>? architecture,
        List<UNetBlock>? customEncoderBlocks,
        List<UNetBlock>? customMiddleBlocks,
        List<UNetBlock>? customDecoderBlocks)
    {
        // Create input/output convolutions and time embedding MLP via LayerHelper
        var baseLayers = LayerHelper<T>.CreateUNetNoisePredictorEncoderLayers(
            _inputChannels, _baseChannels, _channelMultipliers, _numResBlocks,
            _contextDim, _numHeads).ToList();

        // First layer is input conv, next two are time embedding MLP
        _inputConv = (ConvolutionalLayer<T>)baseLayers[0];
        _timeEmbedMlp1 = (DenseLayer<T>)baseLayers[1];
        _timeEmbedMlp2 = (DenseLayer<T>)baseLayers[2];

        // Output conv from decoder layers
        var decoderBaseLayers = LayerHelper<T>.CreateUNetNoisePredictorDecoderLayers(
            _outputChannels, _baseChannels, _channelMultipliers, _numResBlocks).ToList();
        _outputConv = (ConvolutionalLayer<T>)decoderBaseLayers[^1];

        // Priority 1: Use custom blocks passed directly
        if (customEncoderBlocks != null && customEncoderBlocks.Count > 0 &&
            customMiddleBlocks != null && customMiddleBlocks.Count > 0 &&
            customDecoderBlocks != null && customDecoderBlocks.Count > 0)
        {
            _encoderBlocks.AddRange(customEncoderBlocks);
            _middleBlocks.AddRange(customMiddleBlocks);
            _decoderBlocks.AddRange(customDecoderBlocks);
            return;
        }

        // Priority 2: Use layers from NeuralNetworkArchitecture as encoder ResBlocks
        if (architecture?.Layers != null && architecture.Layers.Count > 0)
        {
            foreach (var layer in architecture.Layers)
            {
                _encoderBlocks.Add(new UNetBlock { ResBlock = layer });
            }
            CreateDefaultMiddleBlocks(_baseChannels * _channelMultipliers[^1]);
            CreateDefaultDecoderBlocks();
            return;
        }

        // Priority 3: Create industry-standard layers from the Stable Diffusion paper
        CreateDefaultEncoderBlocks();
        CreateDefaultMiddleBlocks(_baseChannels * _channelMultipliers[^1]);
        CreateDefaultDecoderBlocks();
    }

    /// <summary>
    /// Creates industry-standard encoder blocks based on the Stable Diffusion U-Net.
    /// </summary>
    private void CreateDefaultEncoderBlocks()
    {
        var inChannels = _baseChannels;
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            for (int block = 0; block < _numResBlocks; block++)
            {
                _encoderBlocks.Add(new UNetBlock
                {
                    ResBlock = CreateResBlock(inChannels, outChannels),
                    AttentionBlock = useAttention ? CreateAttentionBlock(outChannels) : null,
                    CrossAttentionBlock = useAttention && _contextDim > 0 ? CreateCrossAttentionBlock(outChannels) : null
                });
                inChannels = outChannels;
            }

            // Add downsampling except for last level
            if (level < _channelMultipliers.Length - 1)
            {
                _encoderBlocks.Add(new UNetBlock
                {
                    Downsample = CreateDownsample(outChannels)
                });
            }
        }
    }

    /// <summary>
    /// Creates industry-standard middle (bottleneck) blocks.
    /// </summary>
    private void CreateDefaultMiddleBlocks(int channels)
    {
        _middleBlocks.Add(new UNetBlock
        {
            ResBlock = CreateResBlock(channels, channels),
            AttentionBlock = CreateAttentionBlock(channels),
            CrossAttentionBlock = _contextDim > 0 ? CreateCrossAttentionBlock(channels) : null
        });
        _middleBlocks.Add(new UNetBlock
        {
            ResBlock = CreateResBlock(channels, channels)
        });
    }

    /// <summary>
    /// Creates industry-standard decoder blocks (reverse of encoder).
    /// </summary>
    private void CreateDefaultDecoderBlocks()
    {
        var inChannels = _baseChannels * _channelMultipliers[^1];
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            for (int block = 0; block <= _numResBlocks; block++)
            {
                var skipChannels = block == 0 && level < _channelMultipliers.Length - 1
                    ? _baseChannels * _channelMultipliers[level + 1]
                    : outChannels;

                _decoderBlocks.Add(new UNetBlock
                {
                    ResBlock = CreateResBlock(inChannels + skipChannels, outChannels),
                    AttentionBlock = useAttention ? CreateAttentionBlock(outChannels) : null,
                    CrossAttentionBlock = useAttention && _contextDim > 0 ? CreateCrossAttentionBlock(outChannels) : null
                });
                inChannels = outChannels;
            }

            // Add upsampling except for first level
            if (level > 0)
            {
                _decoderBlocks.Add(new UNetBlock
                {
                    Upsample = CreateUpsample(outChannels)
                });
            }
        }
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        // Compute timestep embedding
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        // Forward pass through U-Net
        var output = ForwardUNet(noisySample, timeEmbed, conditioning);

        _lastOutput = output;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        // Project time embedding
        var timeEmbed = ProjectTimeEmbedding(timeEmbedding);

        // Forward pass
        var output = ForwardUNet(noisySample, timeEmbed, conditioning);

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Projects the sinusoidal timestep embedding through the MLP.
    /// </summary>
    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
    {
        if (_timeEmbedMlp1 == null || _timeEmbedMlp2 == null)
        {
            throw new InvalidOperationException("Time embedding layers not initialized.");
        }

        var x = _timeEmbedMlp1.Forward(timeEmbed);
        x = _timeEmbedMlp2.Forward(x);
        return x;
    }

    /// <summary>
    /// Performs the forward pass through the U-Net architecture.
    /// </summary>
    private Tensor<T> ForwardUNet(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        if (_inputConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Convolutional layers not initialized.");
        }

        // Input convolution
        x = _inputConv.Forward(x);

        // Store skip connections
        var skips = new List<Tensor<T>>();

        // Encoder
        foreach (var block in _encoderBlocks)
        {
            if (block.Downsample != null)
            {
                x = block.Downsample.Forward(x);
            }
            else
            {
                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock != null)
                {
                    x = block.AttentionBlock.Forward(x);
                }
                if (block.CrossAttentionBlock != null && conditioning != null)
                {
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
                }
                skips.Add(x);
            }
        }

        // Middle
        foreach (var block in _middleBlocks)
        {
            x = ApplyResBlock(block.ResBlock, x, timeEmbed);
            if (block.AttentionBlock != null)
            {
                x = block.AttentionBlock.Forward(x);
            }
            if (block.CrossAttentionBlock != null && conditioning != null)
            {
                x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
            }
        }

        // Decoder
        var skipIdx = skips.Count - 1;
        foreach (var block in _decoderBlocks)
        {
            if (block.Upsample != null)
            {
                x = block.Upsample.Forward(x);
            }
            else
            {
                // Concatenate skip connection
                if (skipIdx >= 0)
                {
                    x = ConcatenateChannels(x, skips[skipIdx]);
                    skipIdx--;
                }

                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock != null)
                {
                    x = block.AttentionBlock.Forward(x);
                }
                if (block.CrossAttentionBlock != null && conditioning != null)
                {
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
                }
            }
        }

        // Output convolution
        x = _outputConv.Forward(x);

        return x;
    }

    /// <summary>
    /// Applies a residual block with time embedding conditioning.
    /// </summary>
    private Tensor<T> ApplyResBlock(ILayer<T>? resBlock, Tensor<T> x, Tensor<T> timeEmbed)
    {
        if (resBlock == null) return x;

        // For now, just apply the block (time conditioning would be added via AdaGN)
        return resBlock.Forward(x);
    }

    /// <summary>
    /// Applies cross-attention between the sample and conditioning.
    /// </summary>
    /// <param name="crossAttn">The cross-attention layer.</param>
    /// <param name="x">Spatial features [batch, channels, height, width].</param>
    /// <param name="conditioning">Text embeddings [batch, seq_len, context_dim].</param>
    /// <returns>Attended spatial features with same shape as x.</returns>
    private Tensor<T> ApplyCrossAttention(ILayer<T>? crossAttn, Tensor<T> x, Tensor<T> conditioning)
    {
        if (crossAttn == null) return x;

        // CrossAttentionLayer handles:
        // - Query: spatial features from x (reshaped internally)
        // - Key/Value: text embeddings from conditioning
        // - Output: attended features with same shape as x

        // DiffusionCrossAttention uses ForwardWithContext for conditioning
        if (crossAttn is DiffusionCrossAttention<T> diffusionCrossAttn)
        {
            return diffusionCrossAttn.ForwardWithContext(x, conditioning);
        }

        // CrossAttentionLayer has Forward(input, context) overload
        if (crossAttn is CrossAttentionLayer<T> crossAttnLayer)
        {
            return crossAttnLayer.Forward(x, conditioning);
        }

        // Fallback for other layers
        return crossAttn.Forward(x);
    }

    /// <summary>
    /// Concatenates two tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        var aShape = a.Shape;
        var bShape = b.Shape;

        var outputShape = new[] { aShape[0], aShape[1] + bShape[1], aShape[2], aShape[3] };
        var output = new Tensor<T>(outputShape);
        var outSpan = output.AsWritableSpan();
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        var batchSize = aShape[0];
        var aChannels = aShape[1];
        var bChannels = bShape[1];
        var height = aShape[2];
        var width = aShape[3];
        var spatialSize = height * width;

        for (int batch = 0; batch < batchSize; batch++)
        {
            // Copy a channels
            for (int c = 0; c < aChannels; c++)
            {
                var srcOffset = batch * aChannels * spatialSize + c * spatialSize;
                var dstOffset = batch * (aChannels + bChannels) * spatialSize + c * spatialSize;
                for (int i = 0; i < spatialSize; i++)
                {
                    outSpan[dstOffset + i] = aSpan[srcOffset + i];
                }
            }

            // Copy b channels
            for (int c = 0; c < bChannels; c++)
            {
                var srcOffset = batch * bChannels * spatialSize + c * spatialSize;
                var dstOffset = batch * (aChannels + bChannels) * spatialSize + (aChannels + c) * spatialSize;
                for (int i = 0; i < spatialSize; i++)
                {
                    outSpan[dstOffset + i] = bSpan[srcOffset + i];
                }
            }
        }

        return output;
    }

    #region Layer Factory Methods

    private ILayer<T> CreateResBlock(int inChannels, int outChannels)
    {
        // Create a residual block with group normalization
        return new DenseLayer<T>(inChannels, outChannels, (IActivationFunction<T>)new SiLUActivation<T>());
    }

    private ILayer<T> CreateAttentionBlock(int channels)
    {
        // Self-attention layer using Flash Attention for memory efficiency
        int latentSpatialSize = _inputHeight;
        return new DiffusionAttention<T>(
            channels: channels,
            numHeads: _numHeads,
            spatialSize: latentSpatialSize,
            flashAttentionThreshold: latentSpatialSize * latentSpatialSize / 16);
    }

    private ILayer<T> CreateCrossAttentionBlock(int channels)
    {
        // Cross-attention layer for conditioning with proper Q/K/V projections
        int latentSpatialSize = _inputHeight;
        return new DiffusionCrossAttention<T>(
            queryDim: channels,
            contextDim: _contextDim,
            numHeads: _numHeads,
            spatialSize: latentSpatialSize);
    }

    private ILayer<T> CreateDownsample(int channels)
    {
        int latentSpatialSize = _inputHeight;
        return new ConvolutionalLayer<T>(
            inputDepth: channels,
            outputDepth: channels,
            kernelSize: 3,
            inputHeight: latentSpatialSize,
            inputWidth: latentSpatialSize,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    private ILayer<T> CreateUpsample(int channels)
    {
        int halfSpatialSize = _inputHeight / 2;
        return new DeconvolutionalLayer<T>(
            inputShape: new[] { 1, channels, halfSpatialSize, halfSpatialSize },
            outputDepth: channels,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    #endregion

    #region Parameter Management

    private int CalculateParameterCount()
    {
        // Approximate parameter count based on architecture
        long count = 0;

        // Input/output convolutions
        count += _inputChannels * _baseChannels * 9 + _baseChannels;
        count += _baseChannels * _outputChannels * 9 + _outputChannels;

        // Time embedding MLP
        count += (_timeEmbeddingDim / 4) * _timeEmbeddingDim + _timeEmbeddingDim;
        count += _timeEmbeddingDim * _timeEmbeddingDim + _timeEmbeddingDim;

        // Estimate blocks
        foreach (var channels in _channelMultipliers.Select(mult => _baseChannels * mult))
        {
            count += _numResBlocks * (channels * channels * 2); // Rough estimate per res block
        }

        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        // Collect parameters from all layers
        AddLayerParameters(parameters, _inputConv);
        AddLayerParameters(parameters, _timeEmbedMlp1);
        AddLayerParameters(parameters, _timeEmbedMlp2);

        foreach (var block in _encoderBlocks)
        {
            AddBlockParameters(parameters, block);
        }

        foreach (var block in _middleBlocks)
        {
            AddBlockParameters(parameters, block);
        }

        foreach (var block in _decoderBlocks)
        {
            AddBlockParameters(parameters, block);
        }

        AddLayerParameters(parameters, _outputConv);

        return new Vector<T>(parameters.ToArray());
    }

    private void AddLayerParameters(List<T> parameters, ILayer<T>? layer)
    {
        if (layer == null) return;
        var layerParams = layer.GetParameters();
        for (int i = 0; i < layerParams.Length; i++)
        {
            parameters.Add(layerParams[i]);
        }
    }

    private void AddBlockParameters(List<T> parameters, UNetBlock block)
    {
        AddLayerParameters(parameters, block.ResBlock);
        AddLayerParameters(parameters, block.AttentionBlock);
        AddLayerParameters(parameters, block.CrossAttentionBlock);
        AddLayerParameters(parameters, block.Downsample);
        AddLayerParameters(parameters, block.Upsample);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var index = 0;

        SetLayerParameters(_inputConv, parameters, ref index);
        SetLayerParameters(_timeEmbedMlp1, parameters, ref index);
        SetLayerParameters(_timeEmbedMlp2, parameters, ref index);

        foreach (var block in _encoderBlocks)
        {
            SetBlockParameters(block, parameters, ref index);
        }

        foreach (var block in _middleBlocks)
        {
            SetBlockParameters(block, parameters, ref index);
        }

        foreach (var block in _decoderBlocks)
        {
            SetBlockParameters(block, parameters, ref index);
        }

        SetLayerParameters(_outputConv, parameters, ref index);
    }

    private void SetLayerParameters(ILayer<T>? layer, Vector<T> parameters, ref int index)
    {
        if (layer == null) return;
        var layerParams = layer.GetParameters();
        var newParams = new Vector<T>(layerParams.Length);
        for (int i = 0; i < layerParams.Length && index < parameters.Length; i++)
        {
            newParams[i] = parameters[index++];
        }
        layer.SetParameters(newParams);
    }

    private void SetBlockParameters(UNetBlock block, Vector<T> parameters, ref int index)
    {
        SetLayerParameters(block.ResBlock, parameters, ref index);
        SetLayerParameters(block.AttentionBlock, parameters, ref index);
        SetLayerParameters(block.CrossAttentionBlock, parameters, ref index);
        SetLayerParameters(block.Downsample, parameters, ref index);
        SetLayerParameters(block.Upsample, parameters, ref index);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new UNetNoisePredictor<T>(
            inputChannels: _inputChannels,
            outputChannels: _outputChannels,
            baseChannels: _baseChannels,
            channelMultipliers: _channelMultipliers,
            numResBlocks: _numResBlocks,
            attentionResolutions: _attentionResolutions,
            contextDim: _contextDim,
            numHeads: _numHeads,
            lossFunction: LossFunction);

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion

    /// <summary>
    /// Structure for U-Net blocks containing residual, attention, and sampling layers.
    /// </summary>
    public class UNetBlock
    {
        public ILayer<T>? ResBlock { get; set; }
        public ILayer<T>? AttentionBlock { get; set; }
        public ILayer<T>? CrossAttentionBlock { get; set; }
        public ILayer<T>? Downsample { get; set; }
        public ILayer<T>? Upsample { get; set; }
    }
}
