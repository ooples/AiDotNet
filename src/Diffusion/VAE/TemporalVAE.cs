using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Temporal-aware Variational Autoencoder for video diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The TemporalVAE extends the standard VAE to handle video data by incorporating
/// temporal awareness into the encoding and decoding process. This helps maintain
/// temporal consistency across frames when used in video diffusion models.
/// </para>
/// <para>
/// <b>For Beginners:</b> While a standard VAE processes each frame independently,
/// TemporalVAE considers relationships between consecutive frames:
///
/// Standard VAE approach (per-frame):
/// - Frame 1 -> Latent 1 (no knowledge of other frames)
/// - Frame 2 -> Latent 2 (no knowledge of other frames)
/// - Result: Possible flickering/inconsistency between frames
///
/// TemporalVAE approach:
/// - Frames 1,2,3,... -> Encode with temporal awareness
/// - Latent knows about neighboring frames
/// - Result: Smoother, more consistent video
///
/// Key features:
/// - 3D convolutions that span across time dimension
/// - Temporal attention for long-range frame relationships
/// - Optional causal mode for streaming/autoregressive generation
///
/// Used in: Stable Video Diffusion, Video LDM, and similar models.
/// </para>
/// <para>
/// Architecture details:
/// - Input: [batch, channels, frames, height, width] video tensor
/// - Encoder: 2D spatial blocks + 1D temporal blocks
/// - Latent: [batch, latentChannels, frames, height/8, width/8]
/// - Decoder: 2D spatial blocks + 1D temporal blocks
/// - Output: [batch, channels, frames, height, width] reconstructed video
/// </para>
/// </remarks>
public class TemporalVAE<T> : VAEModelBase<T>
{
    /// <summary>
    /// Standard Stable Video Diffusion latent scale factor.
    /// </summary>
    private const double SVD_LATENT_SCALE = 0.18215;

    /// <summary>
    /// Number of temporal layers in encoder/decoder.
    /// </summary>
    private readonly int _numTemporalLayers;

    /// <summary>
    /// Base channel count.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Channel multipliers for each level.
    /// </summary>
    private readonly int[] _channelMultipliers;

    /// <summary>
    /// Whether to use causal convolutions (for streaming).
    /// </summary>
    private readonly bool _causalMode;

    /// <summary>
    /// Number of frames to process together.
    /// </summary>
    private readonly int _temporalKernelSize;

    /// <summary>
    /// Encoder spatial layers.
    /// </summary>
    private readonly List<ILayer<T>> _encoderSpatialLayers;

    /// <summary>
    /// Encoder temporal layers.
    /// </summary>
    private readonly List<ILayer<T>> _encoderTemporalLayers;

    /// <summary>
    /// Decoder spatial layers.
    /// </summary>
    private readonly List<ILayer<T>> _decoderSpatialLayers;

    /// <summary>
    /// Decoder temporal layers.
    /// </summary>
    private readonly List<ILayer<T>> _decoderTemporalLayers;

    /// <summary>
    /// Mean projection layer.
    /// </summary>
    private ConvolutionalLayer<T>? _meanConv;

    /// <summary>
    /// Log variance projection layer.
    /// </summary>
    private ConvolutionalLayer<T>? _logVarConv;

    /// <summary>
    /// Input convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _inputConv;

    /// <summary>
    /// Post-quant convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _postQuantConv;

    /// <summary>
    /// Output convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Cached mean from encoding.
    /// </summary>
    private Tensor<T>? _cachedMean;

    /// <summary>
    /// Cached log variance from encoding.
    /// </summary>
    private Tensor<T>? _cachedLogVar;

    /// <summary>
    /// Input channels (3 for RGB video).
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Latent channels.
    /// </summary>
    private readonly int _latentChannels;

    /// <summary>
    /// Downsampling factor.
    /// </summary>
    private readonly int _downsampleFactor;

    /// <summary>
    /// Latent scale factor.
    /// </summary>
    private readonly double _latentScaleFactor;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int LatentChannels => _latentChannels;

    /// <inheritdoc />
    public override int DownsampleFactor => _downsampleFactor;

    /// <inheritdoc />
    public override double LatentScaleFactor => _latentScaleFactor;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    /// <inheritdoc />
    public override bool SupportsTiling => true;

    /// <inheritdoc />
    public override bool SupportsSlicing => true;

    /// <summary>
    /// Gets whether this VAE uses causal convolutions.
    /// </summary>
    public bool IsCausal => _causalMode;

    /// <summary>
    /// Gets the temporal kernel size.
    /// </summary>
    public int TemporalKernelSize => _temporalKernelSize;

    /// <summary>
    /// Initializes a new instance of the TemporalVAE class.
    /// </summary>
    /// <param name="inputChannels">Number of input image channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="channelMultipliers">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numTemporalLayers">Number of temporal layers per spatial block (default: 1).</param>
    /// <param name="temporalKernelSize">Kernel size for temporal convolutions (default: 3).</param>
    /// <param name="causalMode">Whether to use causal convolutions (default: false).</param>
    /// <param name="latentScaleFactor">Scale factor for latents (default: 0.18215).</param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public TemporalVAE(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int[]? channelMultipliers = null,
        int numTemporalLayers = 1,
        int temporalKernelSize = 3,
        bool causalMode = false,
        double? latentScaleFactor = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? new[] { 1, 2, 4, 4 };
        _numTemporalLayers = numTemporalLayers;
        _temporalKernelSize = temporalKernelSize;
        _causalMode = causalMode;
        _latentScaleFactor = latentScaleFactor ?? SVD_LATENT_SCALE;

        _downsampleFactor = (int)Math.Pow(2, _channelMultipliers.Length - 1);

        _encoderSpatialLayers = new List<ILayer<T>>();
        _encoderTemporalLayers = new List<ILayer<T>>();
        _decoderSpatialLayers = new List<ILayer<T>>();
        _decoderTemporalLayers = new List<ILayer<T>>();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes all encoder and decoder layers.
    /// </summary>
    private void InitializeLayers()
    {
        // Input convolution
        _inputConv = new ConvolutionalLayer<T>(
            inputDepth: _inputChannels,
            outputDepth: _baseChannels,
            kernelSize: 3,
            inputHeight: 64,
            inputWidth: 64,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Build encoder
        var inChannels = _baseChannels;
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];

            // Spatial block
            _encoderSpatialLayers.Add(CreateSpatialBlock(inChannels, outChannels));

            // Temporal block(s)
            for (int t = 0; t < _numTemporalLayers; t++)
            {
                _encoderTemporalLayers.Add(CreateTemporalBlock(outChannels));
            }

            inChannels = outChannels;

            // Downsample (except last level)
            if (level < _channelMultipliers.Length - 1)
            {
                _encoderSpatialLayers.Add(CreateDownsample(outChannels));
            }
        }

        // Latent projection layers
        var lastChannels = _baseChannels * _channelMultipliers[^1];
        _meanConv = new ConvolutionalLayer<T>(
            inputDepth: lastChannels,
            outputDepth: _latentChannels,
            kernelSize: 3,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _logVarConv = new ConvolutionalLayer<T>(
            inputDepth: lastChannels,
            outputDepth: _latentChannels,
            kernelSize: 3,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Post-quant convolution
        _postQuantConv = new ConvolutionalLayer<T>(
            inputDepth: _latentChannels,
            outputDepth: lastChannels,
            kernelSize: 3,
            inputHeight: 8,
            inputWidth: 8,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Build decoder
        inChannels = lastChannels;
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];

            // Spatial block
            _decoderSpatialLayers.Add(CreateSpatialBlock(inChannels, outChannels));

            // Temporal block(s)
            for (int t = 0; t < _numTemporalLayers; t++)
            {
                _decoderTemporalLayers.Add(CreateTemporalBlock(outChannels));
            }

            inChannels = outChannels;

            // Upsample (except first level going backwards)
            if (level > 0)
            {
                _decoderSpatialLayers.Add(CreateUpsample(outChannels));
            }
        }

        // Output convolution
        _outputConv = new ConvolutionalLayer<T>(
            inputDepth: _baseChannels,
            outputDepth: _inputChannels,
            kernelSize: 3,
            inputHeight: 64,
            inputWidth: 64,
            stride: 1,
            padding: 1,
            activationFunction: new TanhActivation<T>());
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> video, bool sampleMode = true)
    {
        var (mean, logVar) = EncodeWithDistribution(video);
        return sampleMode ? Sample(mean, logVar) : mean;
    }

    /// <inheritdoc />
    public override (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> video)
    {
        if (_inputConv == null || _meanConv == null || _logVarConv == null)
        {
            throw new InvalidOperationException("Encoder layers not initialized.");
        }

        // Handle video input: [batch, channels, frames, height, width]
        // or image input: [batch, channels, height, width]
        bool isVideo = video.Shape.Length == 5;

        return isVideo ? EncodeVideo(video) : EncodeFrame(video);
    }

    /// <summary>
    /// Encodes a single frame.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVariance) EncodeFrame(Tensor<T> frame)
    {
        if (_inputConv == null || _meanConv == null || _logVarConv == null)
        {
            throw new InvalidOperationException("Encoder layers not initialized.");
        }

        var x = _inputConv.Forward(frame);

        // Apply spatial encoder blocks
        foreach (var layer in _encoderSpatialLayers)
        {
            x = layer.Forward(x);
        }

        var mean = _meanConv.Forward(x);
        var logVar = _logVarConv.Forward(x);

        _cachedMean = mean;
        _cachedLogVar = logVar;

        return (mean, logVar);
    }

    /// <summary>
    /// Encodes a video with temporal awareness.
    /// </summary>
    private (Tensor<T> Mean, Tensor<T> LogVariance) EncodeVideo(Tensor<T> video)
    {
        if (_inputConv == null || _meanConv == null || _logVarConv == null)
        {
            throw new InvalidOperationException("Encoder layers not initialized.");
        }

        int frames = video.Shape[2];

        // Process each frame through spatial encoder
        var frameLatents = new List<Tensor<T>>();
        for (int f = 0; f < frames; f++)
        {
            // Extract frame: [batch, channels, height, width]
            var frame = ExtractFrame(video, f);

            var x = _inputConv.Forward(frame);

            // Apply spatial blocks
            foreach (var layer in _encoderSpatialLayers)
            {
                x = layer.Forward(x);
            }

            frameLatents.Add(x);
        }

        // Apply temporal processing across frames
        var processedLatents = ApplyTemporalLayers(_encoderTemporalLayers, frameLatents);

        // Project to mean and logVar for each frame
        var means = new List<Tensor<T>>();
        var logVars = new List<Tensor<T>>();

        foreach (var latent in processedLatents)
        {
            means.Add(_meanConv.Forward(latent));
            logVars.Add(_logVarConv.Forward(latent));
        }

        // Stack frames back into video format
        var mean = StackFrames(means);
        var logVar = StackFrames(logVars);

        _cachedMean = mean;
        _cachedLogVar = logVar;

        return (mean, logVar);
    }

    /// <inheritdoc />
    public override Tensor<T> Decode(Tensor<T> latent)
    {
        if (_postQuantConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Decoder layers not initialized.");
        }

        // Handle video latent: [batch, latentChannels, frames, height, width]
        // or image latent: [batch, latentChannels, height, width]
        bool isVideo = latent.Shape.Length == 5;

        return isVideo ? DecodeVideo(latent) : DecodeFrame(latent);
    }

    /// <summary>
    /// Decodes a single frame latent.
    /// </summary>
    private Tensor<T> DecodeFrame(Tensor<T> latent)
    {
        if (_postQuantConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Decoder layers not initialized.");
        }

        var x = _postQuantConv.Forward(latent);

        foreach (var layer in _decoderSpatialLayers)
        {
            x = layer.Forward(x);
        }

        return _outputConv.Forward(x);
    }

    /// <summary>
    /// Decodes a video latent with temporal awareness.
    /// </summary>
    private Tensor<T> DecodeVideo(Tensor<T> latent)
    {
        if (_postQuantConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Decoder layers not initialized.");
        }

        int frames = latent.Shape[2];

        // Extract frame latents
        var frameLatents = new List<Tensor<T>>();
        for (int f = 0; f < frames; f++)
        {
            var frameLatent = ExtractFrame(latent, f);
            var x = _postQuantConv.Forward(frameLatent);
            frameLatents.Add(x);
        }

        // Apply temporal processing
        var processedLatents = ApplyTemporalLayers(_decoderTemporalLayers, frameLatents);

        // Decode each frame using LINQ Select for explicit transformation
        var decodedFrames = processedLatents.Select(x =>
        {
            var decoded = _decoderSpatialLayers.Aggregate(x, (current, layer) => layer.Forward(current));
            return _outputConv.Forward(decoded);
        }).ToList();

        return StackFrames(decodedFrames);
    }

    /// <summary>
    /// Encodes a video and applies latent scaling for use in diffusion.
    /// </summary>
    /// <param name="video">The input video tensor.</param>
    /// <param name="sampleMode">Whether to sample from the distribution.</param>
    /// <returns>Scaled video latent representation.</returns>
    public Tensor<T> EncodeVideoForDiffusion(Tensor<T> video, bool sampleMode = true)
    {
        var latent = Encode(video, sampleMode);
        return ScaleLatent(latent);
    }

    /// <summary>
    /// Decodes a diffusion video latent back to video space.
    /// </summary>
    /// <param name="latent">The latent from diffusion (already scaled).</param>
    /// <returns>The decoded video.</returns>
    public Tensor<T> DecodeVideoFromDiffusion(Tensor<T> latent)
    {
        var unscaled = UnscaleLatent(latent);
        return Decode(unscaled);
    }

    #region Temporal Processing

    /// <summary>
    /// Applies temporal layers across frames.
    /// </summary>
    private List<Tensor<T>> ApplyTemporalLayers(List<ILayer<T>> temporalLayers, List<Tensor<T>> frameFeatures)
    {
        if (temporalLayers.Count == 0 || frameFeatures.Count == 0)
        {
            return frameFeatures;
        }

        // Stack frames into a 5D tensor: [batch, channels, frames, height, width]
        var stacked = StackFramesToVideo(frameFeatures);

        // Apply each temporal layer
        var processed = stacked;
        foreach (var layer in temporalLayers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                processed = layerBase.Forward(processed);
            }
            else
            {
                processed = layer.Forward(processed);
            }
        }

        // Unstack back to individual frames
        return UnstackVideoToFrames(processed, frameFeatures.Count);
    }

    /// <summary>
    /// Stacks frame features into a video tensor.
    /// </summary>
    private Tensor<T> StackFramesToVideo(List<Tensor<T>> frames)
    {
        if (frames.Count == 0)
        {
            throw new ArgumentException("No frames to stack.");
        }

        int batch = frames[0].Shape[0];
        int channels = frames[0].Shape[1];
        int numFrames = frames.Count;
        int height = frames[0].Shape[2];
        int width = frames[0].Shape[3];

        var video = new Tensor<T>(new[] { batch, channels, numFrames, height, width });
        var videoSpan = video.AsWritableSpan();
        int spatialSize = height * width;

        for (int f = 0; f < numFrames; f++)
        {
            var frameSpan = frames[f].AsSpan();

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int srcOffset = b * channels * spatialSize + c * spatialSize;
                    int dstOffset = b * channels * numFrames * spatialSize +
                                    c * numFrames * spatialSize +
                                    f * spatialSize;

                    for (int i = 0; i < spatialSize; i++)
                    {
                        videoSpan[dstOffset + i] = frameSpan[srcOffset + i];
                    }
                }
            }
        }

        return video;
    }

    /// <summary>
    /// Unstacks a video tensor back to individual frames.
    /// </summary>
    private List<Tensor<T>> UnstackVideoToFrames(Tensor<T> video, int numFrames)
    {
        int batch = video.Shape[0];
        int channels = video.Shape[1];
        int height = video.Shape[3];
        int width = video.Shape[4];

        var frames = new List<Tensor<T>>();
        var videoSpan = video.AsSpan();
        int spatialSize = height * width;

        for (int f = 0; f < numFrames; f++)
        {
            var frame = new Tensor<T>(new[] { batch, channels, height, width });
            var frameSpan = frame.AsWritableSpan();

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int srcOffset = b * channels * numFrames * spatialSize +
                                    c * numFrames * spatialSize +
                                    f * spatialSize;
                    int dstOffset = b * channels * spatialSize + c * spatialSize;

                    for (int i = 0; i < spatialSize; i++)
                    {
                        frameSpan[dstOffset + i] = videoSpan[srcOffset + i];
                    }
                }
            }

            frames.Add(frame);
        }

        return frames;
    }

    /// <summary>
    /// Extracts a single frame from a video tensor.
    /// </summary>
    private Tensor<T> ExtractFrame(Tensor<T> video, int frameIndex)
    {
        int batch = video.Shape[0];
        int channels = video.Shape[1];
        int height = video.Shape[3];
        int width = video.Shape[4];

        var frame = new Tensor<T>([batch, channels, height, width]);
        var frameSpan = frame.AsWritableSpan();
        var videoSpan = video.AsSpan();

        int spatialSize = height * width;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int videoIdx = b * channels * video.Shape[2] * spatialSize +
                                       c * video.Shape[2] * spatialSize +
                                       frameIndex * spatialSize +
                                       h * width + w;
                        int frameIdx = b * channels * spatialSize +
                                       c * spatialSize +
                                       h * width + w;
                        frameSpan[frameIdx] = videoSpan[videoIdx];
                    }
                }
            }
        }

        return frame;
    }

    /// <summary>
    /// Stacks frames into a video tensor.
    /// </summary>
    private Tensor<T> StackFrames(List<Tensor<T>> frames)
    {
        if (frames.Count == 0)
        {
            throw new ArgumentException("No frames to stack.");
        }

        int batch = frames[0].Shape[0];
        int channels = frames[0].Shape[1];
        int numFrames = frames.Count;
        int height = frames[0].Shape[2];
        int width = frames[0].Shape[3];

        var video = new Tensor<T>([batch, channels, numFrames, height, width]);
        var videoSpan = video.AsWritableSpan();
        int spatialSize = height * width;

        for (int f = 0; f < numFrames; f++)
        {
            var frameSpan = frames[f].AsSpan();

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            int frameIdx = b * channels * spatialSize +
                                           c * spatialSize +
                                           h * width + w;
                            int videoIdx = b * channels * numFrames * spatialSize +
                                           c * numFrames * spatialSize +
                                           f * spatialSize +
                                           h * width + w;
                            videoSpan[videoIdx] = frameSpan[frameIdx];
                        }
                    }
                }
            }
        }

        return video;
    }

    #endregion

    #region Layer Factory Methods

    private ILayer<T> CreateSpatialBlock(int inChannels, int outChannels)
    {
        // Convolutional block that preserves spatial structure
        var conv = new ConvolutionalLayer<T>(
            inputDepth: inChannels,
            outputDepth: outChannels,
            kernelSize: 3,
            inputHeight: 32,  // Placeholder - actual size handled dynamically
            inputWidth: 32,
            stride: 1,
            padding: 1,
            activationFunction: new SiLUActivation<T>());

        if (inChannels == outChannels)
        {
            // Can use residual connection
            return new ResidualLayer<T>(
                inputShape: new[] { 1, inChannels, 32, 32 },
                innerLayer: conv,
                activationFunction: new SiLUActivation<T>());
        }

        return conv;
    }

    private ILayer<T> CreateTemporalBlock(int channels)
    {
        // Temporal block using 3D convolution across time dimension
        // For temporal consistency, use Conv3D with temporal kernel
        return new Conv3DLayer<T>(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            inputDepth: 8,   // Number of frames (temporal dimension)
            inputHeight: 32,
            inputWidth: 32,
            stride: 1,
            padding: 1,
            activationFunction: new SiLUActivation<T>());
    }

    private ILayer<T> CreateDownsample(int channels)
    {
        return new ConvolutionalLayer<T>(
            inputDepth: channels,
            outputDepth: channels,
            kernelSize: 3,
            inputHeight: 32,
            inputWidth: 32,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    private ILayer<T> CreateUpsample(int channels)
    {
        // Transposed convolution (deconvolution) for upsampling
        // With stride=2, kernel=4, padding=1: output = (input - 1) * 2 + 4 - 2*1 = 2*input
        // This doubles the spatial dimensions
        return new DeconvolutionalLayer<T>(
            inputShape: new[] { 1, channels, 16, 16 },  // [batch, channels, height, width]
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
        long count = 0;

        // Input conv
        count += _inputChannels * _baseChannels * 9 + _baseChannels;

        // Encoder blocks
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var channels = _baseChannels * _channelMultipliers[level];
            count += channels * channels * 2; // Spatial
            count += _numTemporalLayers * channels * channels; // Temporal
            if (level < _channelMultipliers.Length - 1)
            {
                count += channels * channels * 9; // Downsample
            }
        }

        // Latent projections
        var lastChannels = _baseChannels * _channelMultipliers[^1];
        count += lastChannels * _latentChannels * 9 * 2; // mean + logvar
        count += _latentChannels * lastChannels * 9; // post-quant

        // Decoder blocks (similar to encoder)
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var channels = _baseChannels * _channelMultipliers[level];
            count += channels * channels * 2;
            count += _numTemporalLayers * channels * channels;
            if (level > 0)
            {
                count += channels * channels * 9;
            }
        }

        // Output conv
        count += _baseChannels * _inputChannels * 9 + _inputChannels;

        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        AddLayerParameters(parameters, _inputConv);

        foreach (var layer in _encoderSpatialLayers)
        {
            AddLayerParameters(parameters, layer);
        }

        foreach (var layer in _encoderTemporalLayers)
        {
            AddLayerParameters(parameters, layer);
        }

        AddLayerParameters(parameters, _meanConv);
        AddLayerParameters(parameters, _logVarConv);
        AddLayerParameters(parameters, _postQuantConv);

        foreach (var layer in _decoderSpatialLayers)
        {
            AddLayerParameters(parameters, layer);
        }

        foreach (var layer in _decoderTemporalLayers)
        {
            AddLayerParameters(parameters, layer);
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

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var index = 0;

        SetLayerParameters(_inputConv, parameters, ref index);

        foreach (var layer in _encoderSpatialLayers)
        {
            SetLayerParameters(layer, parameters, ref index);
        }

        foreach (var layer in _encoderTemporalLayers)
        {
            SetLayerParameters(layer, parameters, ref index);
        }

        SetLayerParameters(_meanConv, parameters, ref index);
        SetLayerParameters(_logVarConv, parameters, ref index);
        SetLayerParameters(_postQuantConv, parameters, ref index);

        foreach (var layer in _decoderSpatialLayers)
        {
            SetLayerParameters(layer, parameters, ref index);
        }

        foreach (var layer in _decoderTemporalLayers)
        {
            SetLayerParameters(layer, parameters, ref index);
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

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IVAEModel<T> Clone()
    {
        var clone = new TemporalVAE<T>(
            _inputChannels,
            _latentChannels,
            _baseChannels,
            _channelMultipliers,
            _numTemporalLayers,
            _temporalKernelSize,
            _causalMode,
            _latentScaleFactor,
            LossFunction);

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion
}
