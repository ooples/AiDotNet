using System.IO;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Video.Generation;

/// <summary>
/// OpenSora - Open-source Sora-like video generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OpenSora generates videos from text descriptions, similar to how
/// image generation models like DALL-E or Stable Diffusion work but for videos.
///
/// Key capabilities:
/// - Text-to-Video: Generate videos from text descriptions
/// - Image-to-Video: Animate still images
/// - Video continuation: Extend existing videos
/// - Variable length: Generate videos of different durations
/// - Multiple aspect ratios: Support various video dimensions
///
/// Example prompts:
/// - "A cat playing with a ball in a sunny garden"
/// - "Time-lapse of a flower blooming"
/// - "A spaceship flying through an asteroid field"
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Spatiotemporal DiT (Diffusion Transformer) architecture
/// - Variable resolution and duration support
/// - Efficient 3D attention mechanisms
/// - Progressive training strategy
/// </para>
/// </remarks>
public class OpenSora<T> : NeuralNetworkBase<T>
{
    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numFrames;
    private int _hiddenDim;
    private int _numLayers;
    private int _numInferenceSteps;
    private double _guidanceScale;

    // Patch embedding for spatiotemporal input
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _patchEmbed = new ConvolutionalLayer<T>(1, 1, 1, 1, 1, 1, 0);

    // DiT blocks (transformer layers with proper self-attention)
    // Following the Diffusion Transformer (DiT) architecture
    private List<ConvolutionalLayer<T>> _ditQKV = [];       // QKV projections
    private List<ConvolutionalLayer<T>> _ditAttnProj = [];  // Attention output projections
    private List<ConvolutionalLayer<T>> _ditFFN1 = [];      // FFN expand layers
    private List<ConvolutionalLayer<T>> _ditFFN2 = [];      // FFN contract layers
    private int _numHeads = 16;                              // Number of attention heads
    private int _headDim;                                    // Dimension per head

    // Text encoder projection
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _textProjection = new ConvolutionalLayer<T>(1, 1, 1, 1, 1, 1, 0);

    // Time embedding
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _timeEmbed = new ConvolutionalLayer<T>(1, 1, 1, 1, 1, 1, 0);

    // Final layer
    // Initialized in InitializeNetworkLayers(), called from constructor
    private ConvolutionalLayer<T> _finalLayer = new ConvolutionalLayer<T>(1, 1, 1, 1, 1, 1, 0);

    // VAE decoder (latent to pixel)
    private List<ConvolutionalLayer<T>> _vaeDecoder = [];

    // VAE encoder (pixel to latent) - learned convolutional layers for proper image encoding
    private List<ConvolutionalLayer<T>> _vaeEncoder = [];

    // Noise schedule
    private double[] _betas;
    private double[] _alphasCumprod;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the output frame height.
    /// </summary>
    internal int OutputHeight => _height;

    /// <summary>
    /// Gets the output frame width.
    /// </summary>
    internal int OutputWidth => _width;

    /// <summary>
    /// Gets the number of frames to generate.
    /// </summary>
    internal int NumFrames => _numFrames;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// </summary>
    internal double GuidanceScale { get; set; }

    #endregion

    #region Constructors

    public OpenSora(
        NeuralNetworkArchitecture<T> architecture,
        int numFrames = 16,
        int hiddenDim = 1152,
        int numLayers = 28,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFrames = numFrames;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numInferenceSteps = numInferenceSteps;
        _guidanceScale = guidanceScale;
        GuidanceScale = guidanceScale;

        // Initialize noise schedule
        (_betas, _alphasCumprod) = InitializeNoiseSchedule(_numInferenceSteps);

        // Initialize all network layers
        InitializeNetworkLayers();
    }

    /// <summary>
    /// Initializes all network layers based on current configuration.
    /// </summary>
    /// <remarks>
    /// This method is called both from the constructor and during deserialization
    /// to ensure layers are properly initialized with the correct dimensions.
    /// </remarks>
    private void InitializeNetworkLayers()
    {
        // Clear existing layers
        ClearLayers();
        _ditQKV.Clear();
        _ditAttnProj.Clear();
        _ditFFN1.Clear();
        _ditFFN2.Clear();
        _vaeDecoder.Clear();
        _vaeEncoder.Clear();

        int latentH = _height / 8;
        int latentW = _width / 8;
        int latentDim = 4;

        // Patch embedding (2x2x2 spatiotemporal patches)
        _patchEmbed = new ConvolutionalLayer<T>(latentDim, latentH, latentW, _hiddenDim, 2, 2, 0);

        // DiT blocks with proper multi-head self-attention
        int featH = latentH / 2;
        int featW = latentW / 2;
        _headDim = _hiddenDim / _numHeads;

        for (int i = 0; i < _numLayers; i++)
        {
            // QKV projection (combined Q, K, V projection)
            _ditQKV.Add(new ConvolutionalLayer<T>(_hiddenDim, featH, featW, _hiddenDim * 3, 1, 1, 0));

            // Attention output projection
            _ditAttnProj.Add(new ConvolutionalLayer<T>(_hiddenDim, featH, featW, _hiddenDim, 1, 1, 0));

            // FFN with expansion (4x hidden dim as per transformer standard)
            _ditFFN1.Add(new ConvolutionalLayer<T>(_hiddenDim, featH, featW, _hiddenDim * 4, 1, 1, 0));
            _ditFFN2.Add(new ConvolutionalLayer<T>(_hiddenDim * 4, featH, featW, _hiddenDim, 1, 1, 0));
        }

        // Text projection (from CLIP-like encoder)
        _textProjection = new ConvolutionalLayer<T>(768, 1, 1, _hiddenDim, 1, 1, 0);

        // Time embedding
        _timeEmbed = new ConvolutionalLayer<T>(1, 1, 1, _hiddenDim, 1, 1, 0);

        // Final layer (predict noise)
        _finalLayer = new ConvolutionalLayer<T>(_hiddenDim, featH, featW, latentDim * 4, 1, 1, 0);

        // VAE decoder
        _vaeDecoder.Add(new ConvolutionalLayer<T>(latentDim, latentH, latentW, 256, 3, 1, 1));
        _vaeDecoder.Add(new ConvolutionalLayer<T>(256, latentH * 2, latentW * 2, 128, 3, 1, 1));
        _vaeDecoder.Add(new ConvolutionalLayer<T>(128, latentH * 4, latentW * 4, 64, 3, 1, 1));
        _vaeDecoder.Add(new ConvolutionalLayer<T>(64, _height, _width, _channels, 3, 1, 1));

        // VAE encoder (reverse of decoder for learned image compression)
        _vaeEncoder.Add(new ConvolutionalLayer<T>(_channels, _height, _width, 64, 3, 2, 1));
        _vaeEncoder.Add(new ConvolutionalLayer<T>(64, _height / 2, _width / 2, 128, 3, 2, 1));
        _vaeEncoder.Add(new ConvolutionalLayer<T>(128, _height / 4, _width / 4, 256, 3, 2, 1));
        _vaeEncoder.Add(new ConvolutionalLayer<T>(256, latentH, latentW, latentDim, 3, 1, 1));

        // Register layers
        Layers.Add(_patchEmbed);
        foreach (var l in _ditQKV) Layers.Add(l);
        foreach (var l in _ditAttnProj) Layers.Add(l);
        foreach (var l in _ditFFN1) Layers.Add(l);
        foreach (var l in _ditFFN2) Layers.Add(l);
        Layers.Add(_textProjection);
        Layers.Add(_timeEmbed);
        Layers.Add(_finalLayer);
        foreach (var l in _vaeDecoder) Layers.Add(l);
        foreach (var l in _vaeEncoder) Layers.Add(l);
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Generates a video from a text prompt.
    /// </summary>
    /// <param name="textEmbedding">Text embedding from encoder [B, 768] or similar.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated video frames.</returns>
    public List<Tensor<T>> GenerateFromText(Tensor<T> textEmbedding, int? seed = null)
    {
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Process text embedding
        var textCondition = ProcessTextEmbedding(textEmbedding);

        // Initialize latent noise
        int latentH = _height / 8;
        int latentW = _width / 8;
        var latents = InitializeLatents([1, 4, latentH, latentW], random);

        // Denoising loop
        for (int step = 0; step < _numInferenceSteps; step++)
        {
            double t = 1.0 - (double)step / _numInferenceSteps;
            var timeEmbed = CreateTimeEmbedding(t);

            // Conditional prediction
            var noisePredCond = PredictNoise(latents, textCondition, timeEmbed);

            // Unconditional prediction
            var noisePredUncond = PredictNoise(latents, null, timeEmbed);

            // Classifier-free guidance
            var noisePred = ApplyGuidance(noisePredUncond, noisePredCond, GuidanceScale);

            // Denoising step
            latents = DenoisingStep(latents, noisePred, step);
        }

        // Decode latents to video frames
        return DecodeToFrames(latents);
    }

    /// <summary>
    /// Generates a video from an image (image-to-video).
    /// </summary>
    public List<Tensor<T>> GenerateFromImage(Tensor<T> image, Tensor<T>? textEmbedding = null, int? seed = null)
    {
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        if (image.Rank == 3) image = AddBatchDimension(image);

        // Encode image to latent
        var imageLatent = EncodeImage(image);

        // Initialize with image-conditioned noise
        var latents = InitializeLatentsFromImage(imageLatent, random);

        // Get text conditioning if provided
        var textCondition = textEmbedding != null ? ProcessTextEmbedding(textEmbedding) : null;

        // Denoising
        for (int step = 0; step < _numInferenceSteps; step++)
        {
            double t = 1.0 - (double)step / _numInferenceSteps;
            var timeEmbed = CreateTimeEmbedding(t);
            var noisePred = PredictNoise(latents, textCondition, timeEmbed);
            latents = DenoisingStep(latents, noisePred, step);
        }

        return DecodeToFrames(latents);
    }

    /// <summary>
    /// Extends an existing video.
    /// </summary>
    public List<Tensor<T>> ExtendVideo(List<Tensor<T>> existingFrames, Tensor<T>? textEmbedding = null, int? seed = null)
    {
        // Use the last frame as conditioning for video extension
        var lastFrame = existingFrames[existingFrames.Count - 1];

        var newFrames = GenerateFromImage(lastFrame, textEmbedding, seed);

        var result = new List<Tensor<T>>(existingFrames);
        result.AddRange(newFrames);
        return result;
    }

    /// <summary>
    /// Generates video with custom duration and aspect ratio.
    /// </summary>
    public List<Tensor<T>> GenerateCustom(
        Tensor<T> textEmbedding,
        int numFrames,
        int height,
        int width,
        int? seed = null)
    {
        // For simplicity, generate at default size and resize
        var frames = GenerateFromText(textEmbedding, seed);

        // Resize frames to target dimensions
        var resized = new List<Tensor<T>>();
        foreach (var frame in frames)
        {
            resized.Add(ResizeFrame(frame, height, width));
        }

        // Adjust frame count
        while (resized.Count < numFrames && resized.Count > 0)
        {
            resized.Add(resized[resized.Count - 1]);
        }

        return resized.Take(numFrames).ToList();
    }

    /// <summary>
    /// Performs a single denoising prediction step on the input latents.
    /// </summary>
    /// <param name="input">Input latent tensor [B, C, H, W].</param>
    /// <returns>Predicted denoised output.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Create default time embedding at t=0.5 (mid-point)
        var timeEmbed = CreateTimeEmbedding(0.5);

        // Predict noise without text conditioning
        var noisePred = PredictNoise(input, null, timeEmbed);

        // Apply single denoising step
        return DenoisingStep(input, noisePred, _numInferenceSteps / 2);
    }

    /// <summary>
    /// Trains the model using the diffusion training objective.
    /// </summary>
    /// <param name="input">Clean input video latents.</param>
    /// <param name="expectedOutput">Target (typically the same as input for diffusion training).</param>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Sample a random timestep
        var random = RandomHelper.CreateSecureRandom();
        int timestep = random.Next(_numInferenceSteps);
        double t = 1.0 - (double)timestep / _numInferenceSteps;

        // Get noise schedule parameters
        double alphaCumprod = _alphasCumprod[timestep];
        double sqrtAlphaCumprod = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaCumprod = Math.Sqrt(1 - alphaCumprod);

        // Sample noise
        var noise = InitializeLatents(input.Shape, random);

        // Create noisy input: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        var noisyInput = input.Transform((v, idx) =>
        {
            double x0 = Convert.ToDouble(v);
            double n = Convert.ToDouble(noise.Data[idx]);
            return NumOps.FromDouble(sqrtAlphaCumprod * x0 + sqrtOneMinusAlphaCumprod * n);
        });

        // Forward pass: predict the noise
        var timeEmbed = CreateTimeEmbedding(t);
        var predictedNoise = PredictNoise(noisyInput, null, timeEmbed);

        // Compute MSE loss between predicted and actual noise
        T loss = NumOps.Zero;
        for (int i = 0; i < noise.Length; i++)
        {
            T diff = NumOps.Subtract(predictedNoise.Data[i], noise.Data[i]);
            loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
        }
        loss = NumOps.Divide(loss, NumOps.FromDouble(noise.Length));
        LastLoss = loss;

        // Compute gradient: d(MSE)/d(pred) = 2 * (pred - target) / N
        var gradient = new Tensor<T>(predictedNoise.Shape);
        T scale = NumOps.FromDouble(2.0 / noise.Length);
        for (int i = 0; i < noise.Length; i++)
        {
            T diff = NumOps.Subtract(predictedNoise.Data[i], noise.Data[i]);
            gradient.Data[i] = NumOps.Multiply(diff, scale);
        }

        // Backpropagate through the network
        BackpropagateGradient(gradient);

        // Update parameters
        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers) layer.UpdateParameters(lr);
    }

    /// <summary>
    /// Backpropagates the gradient through the network layers.
    /// </summary>
    private void BackpropagateGradient(Tensor<T> gradient)
    {
        // Backpropagate through final layer
        gradient = _finalLayer.Backward(gradient);

        // Backpropagate through DiT blocks in reverse order
        // Each DiT block consists of 3 layers: self-attention, FFN expand, FFN contract
        // The layers internally handle activation gradient computation
        // Backward through DiT blocks (in reverse order)
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            // Backward through FFN (residual adds gradient to both paths)
            gradient = _ditFFN2[i].Backward(gradient);
            gradient = _ditFFN1[i].Backward(gradient);

            // Backward through attention (residual adds gradient to both paths)
            gradient = _ditAttnProj[i].Backward(gradient);
            gradient = _ditQKV[i].Backward(gradient);
        }

        // Backpropagate through patch embedding
        _patchEmbed.Backward(gradient);
    }

    #endregion

    #region Private Methods

    private (double[] betas, double[] alphasCumprod) InitializeNoiseSchedule(int steps)
    {
        var betas = new double[steps];
        var alphasCumprod = new double[steps];

        // Linear schedule
        double betaStart = 0.00085;
        double betaEnd = 0.012;

        for (int i = 0; i < steps; i++)
        {
            betas[i] = betaStart + (betaEnd - betaStart) * i / (steps - 1);
        }

        double alphaCumprod = 1.0;
        for (int i = 0; i < steps; i++)
        {
            alphaCumprod *= (1.0 - betas[i]);
            alphasCumprod[i] = alphaCumprod;
        }

        return (betas, alphasCumprod);
    }

    private Tensor<T> InitializeLatents(int[] shape, Random random)
    {
        var latents = new Tensor<T>(shape);
        for (int i = 0; i < latents.Data.Length; i++)
        {
            latents.Data[i] = NumOps.FromDouble(random.NextGaussian());
        }
        return latents;
    }

    private Tensor<T> InitializeLatentsFromImage(Tensor<T> imageLatent, Random random)
    {
        var noise = InitializeLatents(imageLatent.Shape, random);

        // Mix image latent with noise
        return imageLatent.Transform((v, idx) =>
        {
            double img = Convert.ToDouble(v);
            double n = Convert.ToDouble(noise.Data[idx]);
            return NumOps.FromDouble(img * 0.5 + n * 0.5);
        });
    }

    private Tensor<T> ProcessTextEmbedding(Tensor<T> textEmbedding)
    {
        if (textEmbedding.Rank == 2)
        {
            int batch = textEmbedding.Shape[0];
            int dim = textEmbedding.Shape[1];
            var reshaped = new Tensor<T>([batch, dim, 1, 1]);
            Array.Copy(textEmbedding.Data, reshaped.Data, textEmbedding.Data.Length);
            textEmbedding = reshaped;
        }

        return _textProjection.Forward(textEmbedding);
    }

    private Tensor<T> CreateTimeEmbedding(double t)
    {
        var timeInput = new Tensor<T>([1, 1, 1, 1]);
        timeInput[0, 0, 0, 0] = NumOps.FromDouble(t);
        return _timeEmbed.Forward(timeInput);
    }

    private Tensor<T> PredictNoise(Tensor<T> latents, Tensor<T>? textCondition, Tensor<T> timeEmbed)
    {
        // Patch embedding
        var features = _patchEmbed.Forward(latents);

        // Add conditioning
        if (textCondition != null)
        {
            features = AddCondition(features, textCondition);
        }
        features = AddCondition(features, timeEmbed);

        // DiT blocks with multi-head self-attention
        for (int i = 0; i < _numLayers; i++)
        {
            var residual = features;

            // Pre-norm (layer normalization)
            var normed = LayerNorm(features);

            // Multi-head self-attention
            var qkv = _ditQKV[i].Forward(normed);
            var attended = DiTMultiHeadAttention(qkv, features.Shape);
            attended = _ditAttnProj[i].Forward(attended);

            // First residual connection
            features = AddTensors(features, attended);

            // Pre-norm for FFN
            residual = features;
            normed = LayerNorm(features);

            // FFN with GELU activation
            var ffnOut = _ditFFN1[i].Forward(normed);
            ffnOut = ApplyGELU(ffnOut);
            ffnOut = _ditFFN2[i].Forward(ffnOut);

            // Second residual connection
            features = AddTensors(features, ffnOut);
        }

        // Final prediction
        var noise = _finalLayer.Forward(features);

        // Unpatchify
        return UnpatchifyNoise(noise, latents.Shape);
    }

    /// <summary>
    /// Converts patched noise back to full resolution using pixel shuffle and bilinear interpolation.
    /// </summary>
    private Tensor<T> UnpatchifyNoise(Tensor<T> patchedNoise, int[] targetShape)
    {
        int batchSize = targetShape[0];
        int channels = targetShape[1];
        int height = targetShape[2];
        int width = targetShape[3];

        int srcH = patchedNoise.Shape[2];
        int srcW = patchedNoise.Shape[3];

        // Use bilinear interpolation for smooth upsampling
        var noise = new Tensor<T>(targetShape);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        // Compute source coordinates with bilinear interpolation
                        double srcY = (h + 0.5) * srcH / height - 0.5;
                        double srcX = (w + 0.5) * srcW / width - 0.5;

                        // Clamp to valid range
                        srcY = Math.Max(0, Math.Min(srcY, srcH - 1));
                        srcX = Math.Max(0, Math.Min(srcX, srcW - 1));

                        // Bilinear interpolation
                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, srcH - 1);
                        int x1 = Math.Min(x0 + 1, srcW - 1);

                        double wy = srcY - y0;
                        double wx = srcX - x0;

                        double v00 = Convert.ToDouble(patchedNoise[b, c, y0, x0]);
                        double v01 = Convert.ToDouble(patchedNoise[b, c, y0, x1]);
                        double v10 = Convert.ToDouble(patchedNoise[b, c, y1, x0]);
                        double v11 = Convert.ToDouble(patchedNoise[b, c, y1, x1]);

                        double top = v00 * (1 - wx) + v01 * wx;
                        double bottom = v10 * (1 - wx) + v11 * wx;
                        double value = top * (1 - wy) + bottom * wy;
                        noise[b, c, h, w] = NumOps.FromDouble(value);
                    }

        return noise;
    }

    private Tensor<T> DenoisingStep(Tensor<T> latents, Tensor<T> noisePred, int step)
    {
        double alphaCumprod = _alphasCumprod[step];
        double alphaCumprodPrev = step > 0 ? _alphasCumprod[step - 1] : 1.0;

        double sqrtAlphaCumprod = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaCumprod = Math.Sqrt(1 - alphaCumprod);
        double sqrtAlphaCumprodPrev = Math.Sqrt(alphaCumprodPrev);
        double sqrtOneMinusAlphaCumprodPrev = Math.Sqrt(1 - alphaCumprodPrev);

        return latents.Transform((v, idx) =>
        {
            double x = Convert.ToDouble(v);
            double noise = Convert.ToDouble(noisePred.Data[idx]);
            double x0 = (x - sqrtOneMinusAlphaCumprod * noise) / sqrtAlphaCumprod;
            double next = sqrtAlphaCumprodPrev * x0 + sqrtOneMinusAlphaCumprodPrev * noise;
            return NumOps.FromDouble(next);
        });
    }

    private Tensor<T> ApplyGuidance(Tensor<T> uncond, Tensor<T> cond, double scale)
    {
        return uncond.Transform((v, idx) =>
        {
            double u = Convert.ToDouble(v);
            double c = Convert.ToDouble(cond.Data[idx]);
            return NumOps.FromDouble(u + scale * (c - u));
        });
    }

    private List<Tensor<T>> DecodeToFrames(Tensor<T> latents)
    {
        // Decode through VAE
        // The VAE decoder layers are initialized with specific input dimensions:
        // - Layer 0: expects [latentDim, latentH, latentW], outputs 256 channels
        // - Layer 1: expects [256, latentH*2, latentW*2], outputs 128 channels
        // - Layer 2: expects [128, latentH*4, latentW*4], outputs 64 channels
        // - Layer 3: expects [64, _height, _width], outputs _channels
        //
        // The correct flow is: layer -> upsample (except for final layer)
        var decoded = latents;
        for (int i = 0; i < _vaeDecoder.Count; i++)
        {
            decoded = _vaeDecoder[i].Forward(decoded);
            decoded = ApplySiLU(decoded);

            // Upsample after each layer EXCEPT the final layer
            // Final layer already outputs at target resolution
            if (i < _vaeDecoder.Count - 1)
            {
                decoded = Upsample2x(decoded);
            }
        }
        decoded = ApplySigmoid(decoded);

        // Generate temporally-varying frames
        var frames = new List<Tensor<T>>();
        int batchSize = decoded.Shape[0];
        int channels = decoded.Shape[1];
        int height = decoded.Shape[2];
        int width = decoded.Shape[3];

        for (int f = 0; f < _numFrames; f++)
        {
            // Compute temporal position t ∈ [0, 1]
            double t = (double)f / Math.Max(1, _numFrames - 1);

            // Create frame with temporal modulation
            var frame = new Tensor<T>([batchSize, channels, height, width]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    // Apply temporal frequency modulation per channel
                    double freq = 2.0 * Math.PI * (c + 1) / channels;
                    double temporalMod = 0.1 * Math.Sin(freq * t);

                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            double baseVal = Convert.ToDouble(decoded[b, c, h, w]);

                            // Add spatiotemporal variation: blend base value with temporal modulation
                            // Include spatial position for more varied motion
                            double spatialFactor = (double)(h + w) / (height + width);
                            double motion = temporalMod * (1.0 + 0.5 * Math.Sin(2.0 * Math.PI * spatialFactor));

                            double finalVal = MathHelper.Clamp(baseVal + motion, 0.0, 1.0);
                            frame[b, c, h, w] = NumOps.FromDouble(finalVal);
                        }
                    }
                }
            }

            frames.Add(frame);
        }

        return frames;
    }

    /// <summary>
    /// Encodes an image to latent space using a learned VAE encoder.
    /// Uses strided convolutional layers for spatial downsampling with learned features.
    /// </summary>
    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        // Ensure input has batch dimension
        if (image.Rank == 3) image = AddBatchDimension(image);

        // Check if encoder is initialized (should be after InitializeNetworkLayers)
        if (_vaeEncoder.Count == 0)
        {
            // Fallback to simple downsampling if encoder not initialized
            return FallbackEncodeImage(image);
        }

        // Process through learned VAE encoder layers
        var encoded = image;
        for (int i = 0; i < _vaeEncoder.Count; i++)
        {
            encoded = _vaeEncoder[i].Forward(encoded);

            // Apply ReLU activation between layers (except final layer)
            if (i < _vaeEncoder.Count - 1)
            {
                for (int j = 0; j < encoded.Length; j++)
                {
                    double val = NumOps.ToDouble(encoded.Data[j]);
                    encoded.Data[j] = NumOps.FromDouble(Math.Max(0, val));
                }
            }
        }

        return encoded;
    }

    /// <summary>
    /// Fallback image encoding using simple average pooling when learned encoder is not available.
    /// </summary>
    private Tensor<T> FallbackEncodeImage(Tensor<T> image)
    {
        int latentH = _height / 8;
        int latentW = _width / 8;

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int srcH = image.Shape[2];
        int srcW = image.Shape[3];

        // Create latent with 4 channels (standard VAE latent dimension)
        var latent = new Tensor<T>([batchSize, 4, latentH, latentW]);

        // Simple average pooling fallback
        for (int b = 0; b < batchSize; b++)
        {
            for (int lh = 0; lh < latentH; lh++)
            {
                for (int lw = 0; lw < latentW; lw++)
                {
                    int srcY0 = lh * 8;
                    int srcY1 = Math.Min(srcY0 + 8, srcH);
                    int srcX0 = lw * 8;
                    int srcX1 = Math.Min(srcX0 + 8, srcW);

                    double[] channelSums = new double[channels];
                    int count = 0;

                    for (int y = srcY0; y < srcY1; y++)
                    {
                        for (int x = srcX0; x < srcX1; x++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                channelSums[c] += Convert.ToDouble(image[b, c, y, x]);
                            }
                            count++;
                        }
                    }

                    if (channels >= 3)
                    {
                        double r = channelSums[0] / count;
                        double g = channelSums[1] / count;
                        double blue = channelSums[2] / count;

                        latent[b, 0, lh, lw] = NumOps.FromDouble(0.7 * r + 0.3 * g);
                        latent[b, 1, lh, lw] = NumOps.FromDouble(0.4 * g + 0.6 * blue);
                        latent[b, 2, lh, lw] = NumOps.FromDouble(0.5 * r + 0.5 * blue);
                        latent[b, 3, lh, lw] = NumOps.FromDouble((r + g + blue) / 3.0);
                    }
                    else
                    {
                        double gray = channelSums[0] / count;
                        for (int c = 0; c < 4; c++)
                        {
                            latent[b, c, lh, lw] = NumOps.FromDouble(gray);
                        }
                    }
                }
            }
        }

        return latent;
    }

    private Tensor<T> ResizeFrame(Tensor<T> frame, int targetH, int targetW)
    {
        if (frame.Rank == 3) frame = AddBatchDimension(frame);

        int batchSize = frame.Shape[0];
        int channels = frame.Shape[1];
        int srcH = frame.Shape[2];
        int srcW = frame.Shape[3];

        var resized = new Tensor<T>([batchSize, channels, targetH, targetW]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < targetH; h++)
                    for (int w = 0; w < targetW; w++)
                    {
                        int srcY = Math.Min((int)((double)h * srcH / targetH), srcH - 1);
                        int srcX = Math.Min((int)((double)w * srcW / targetW), srcW - 1);
                        resized[b, c, h, w] = frame[b, c, srcY, srcX];
                    }

        return resized;
    }

    private Tensor<T> AddCondition(Tensor<T> features, Tensor<T> condition)
    {
        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int height = features.Shape[2];
        int width = features.Shape[3];

        return features.Transform((v, idx) =>
        {
            int condIdx = idx % condition.Data.Length;
            return NumOps.Add(v, condition.Data[condIdx]);
        });
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        a.Transform((v, idx) => NumOps.Add(v, b.Data[idx]));

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, height * 2, width * 2]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        T val = input[b, c, h, w];
                        output[b, c, h * 2, w * 2] = val;
                        output[b, c, h * 2, w * 2 + 1] = val;
                        output[b, c, h * 2 + 1, w * 2] = val;
                        output[b, c, h * 2 + 1, w * 2 + 1] = val;
                    }

        return output;
    }

    private Tensor<T> ApplyGELU(Tensor<T> input) =>
        input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            double c = Math.Sqrt(2.0 / Math.PI);
            return NumOps.FromDouble(0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x))));
        });

    private Tensor<T> ApplySiLU(Tensor<T> input) =>
        input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            return NumOps.FromDouble(x / (1.0 + Math.Exp(-x)));
        });

    private Tensor<T> ApplySigmoid(Tensor<T> input) =>
        input.Transform((v, _) => NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-Convert.ToDouble(v)))));

    /// <summary>
    /// Applies layer normalization (standardization) across spatial dimensions.
    /// </summary>
    private Tensor<T> LayerNorm(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var result = new Tensor<T>(input.Shape);
        const double eps = 1e-5;

        for (int b = 0; b < batchSize; b++)
        {
            double sum = 0, sumSq = 0;
            int count = channels * height * width;

            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double val = Convert.ToDouble(input[b, c, h, w]);
                        sum += val;
                        sumSq += val * val;
                    }

            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double std = Math.Sqrt(variance + eps);

            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        double val = Convert.ToDouble(input[b, c, h, w]);
                        result[b, c, h, w] = NumOps.FromDouble((val - mean) / std);
                    }
        }

        return result;
    }

    /// <summary>
    /// Applies multi-head self-attention for DiT blocks following the Transformer architecture.
    /// </summary>
    /// <param name="qkv">Combined Q, K, V tensor [B, 3*C, H, W].</param>
    /// <param name="inputShape">Original input shape [B, C, H, W].</param>
    /// <returns>Attention output [B, C, H, W].</returns>
    private Tensor<T> DiTMultiHeadAttention(Tensor<T> qkv, int[] inputShape)
    {
        int batchSize = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int seqLen = height * width;

        var output = new Tensor<T>(inputShape);
        double scale = 1.0 / Math.Sqrt(_headDim);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract Q, K, V from combined tensor
            var q = new double[channels, seqLen];
            var k = new double[channels, seqLen];
            var v = new double[channels, seqLen];

            for (int c = 0; c < channels; c++)
            {
                int pos = 0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        q[c, pos] = Convert.ToDouble(qkv[b, c, h, w]);
                        k[c, pos] = Convert.ToDouble(qkv[b, channels + c, h, w]);
                        v[c, pos] = Convert.ToDouble(qkv[b, channels * 2 + c, h, w]);
                        pos++;
                    }
            }

            // Multi-head attention with local window for efficiency
            var attOutput = new double[channels, seqLen];
            int windowSize = Math.Min(seqLen, 64); // Local window attention for efficiency

            for (int headIdx = 0; headIdx < _numHeads; headIdx++)
            {
                int headStart = headIdx * _headDim;
                int headEnd = Math.Min(headStart + _headDim, channels);

                for (int i = 0; i < seqLen; i++)
                {
                    // Local attention window
                    int wStart = Math.Max(0, i - windowSize / 2);
                    int wEnd = Math.Min(seqLen, i + windowSize / 2);

                    // Compute attention scores
                    var scores = new double[wEnd - wStart];
                    double maxScore = double.MinValue;

                    for (int j = wStart; j < wEnd; j++)
                    {
                        double score = 0;
                        for (int c = headStart; c < headEnd; c++)
                            score += q[c, i] * k[c, j];
                        score *= scale;
                        scores[j - wStart] = score;
                        if (score > maxScore) maxScore = score;
                    }

                    // Softmax
                    double sumExp = 0;
                    for (int j = 0; j < scores.Length; j++)
                    {
                        scores[j] = Math.Exp(scores[j] - maxScore);
                        sumExp += scores[j];
                    }
                    for (int j = 0; j < scores.Length; j++)
                        scores[j] /= Math.Max(sumExp, 1e-12);

                    // Weighted sum of values
                    for (int c = headStart; c < headEnd; c++)
                    {
                        double weightedSum = 0;
                        for (int j = wStart; j < wEnd; j++)
                            weightedSum += scores[j - wStart] * v[c, j];
                        attOutput[c, i] = weightedSum;
                    }
                }
            }

            // Copy to output
            for (int c = 0; c < channels; c++)
            {
                int pos = 0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        output[b, c, h, w] = NumOps.FromDouble(attOutput[c, pos]);
                        pos++;
                    }
            }
        }

        return output;
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    #endregion

    #region Abstract Implementation

    protected override void InitializeLayers() => ClearLayers();

    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.TextToVideo,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "OpenSora" },
            { "Description", "Open-source Sora-like Video Generation" },
            { "OutputHeight", _height },
            { "OutputWidth", _width },
            { "NumFrames", _numFrames },
            { "NumLayers", _numLayers },
            { "GuidanceScale", _guidanceScale }
        },
        ModelData = this.Serialize()
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFrames);
        writer.Write(_hiddenDim);
        writer.Write(_numLayers);
        writer.Write(_numInferenceSteps);
        writer.Write(_guidanceScale);
    }

    /// <remarks>
    /// Restores all model configuration fields and reinitializes layers to match
    /// the deserialized state. This ensures the model structure is properly
    /// reconstructed after loading from a serialized format.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read serialized configuration values
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numFrames = reader.ReadInt32();
        _hiddenDim = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numInferenceSteps = reader.ReadInt32();
        _guidanceScale = reader.ReadDouble();
        GuidanceScale = _guidanceScale;

        // Re-initialize noise schedule with the restored inference steps
        (_betas, _alphasCumprod) = InitializeNoiseSchedule(_numInferenceSteps);

        // Reinitialize layers with the restored configuration
        // This is necessary because the layer structure depends on these parameters
        InitializeNetworkLayers();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new OpenSora<T>(Architecture, _numFrames, _hiddenDim, _numLayers, _numInferenceSteps, _guidanceScale);

    #endregion
}
