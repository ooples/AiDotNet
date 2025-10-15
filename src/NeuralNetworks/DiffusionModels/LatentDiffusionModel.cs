using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks.DiffusionModels
{
    /// <summary>
    /// Latent Diffusion Model (LDM) - Operates in compressed latent space for efficient high-resolution generation
    /// </summary>
    /// <remarks>
    /// <para>
    /// Latent Diffusion Models perform the diffusion process in a compressed latent space rather than 
    /// directly on high-resolution images. This approach, popularized by Stable Diffusion, dramatically 
    /// reduces computational requirements while maintaining high-quality generation capabilities.
    /// </para>
    /// <para><b>For Beginners:</b> Think of it like working with compressed files instead of originals.
    /// 
    /// Instead of working with huge image files directly:
    /// 1. Compress images to a smaller "latent" representation (like creating thumbnails)
    /// 2. Learn to generate in this compressed space (much faster and efficient)
    /// 3. Decompress back to full images when done
    /// 
    /// Benefits:
    /// - Much faster training and generation
    /// - Can work with higher resolution images
    /// - Requires less memory and computation
    /// - Powers models like Stable Diffusion
    /// 
    /// This allows generating 512x512 or larger images on consumer hardware!
    /// </para>
    /// </remarks>
    public class LatentDiffusionModel : DiffusionModel, IDisposable
    {
        private readonly IAutoencoder _encoder;
        private readonly IAutoencoder _decoder;
        private readonly int _latentChannels;
        private readonly double _scaleFactor;
        private readonly ITextEncoder? _textEncoder;
        private readonly bool _useConditioning;
        private readonly double _defaultGuidanceScale;
        private readonly object _lockObject = new object();
        private bool _disposed;
        
        // Validation parameters
        private readonly double _maxStrength = 1.0;
        private readonly double _minStrength = 0.0;
        private readonly int _maxBatchSize = 16;
        
        /// <summary>
        /// Gets the encoder used for compressing images to latent space
        /// </summary>
        public IAutoencoder Encoder => _encoder;
        
        /// <summary>
        /// Gets the decoder used for reconstructing images from latent space
        /// </summary>
        public IAutoencoder Decoder => _decoder;
        
        /// <summary>
        /// Gets the text encoder used for conditioning (if available)
        /// </summary>
        public ITextEncoder? TextEncoder => _textEncoder;
        
        /// <summary>
        /// Gets whether text conditioning is enabled
        /// </summary>
        public bool UseConditioning => _useConditioning;
        
        /// <summary>
        /// Initializes a new instance of the LatentDiffusionModel class
        /// </summary>
        /// <param name="encoder">The encoder for compressing images to latent space</param>
        /// <param name="decoder">The decoder for reconstructing images from latent space</param>
        /// <param name="textEncoder">Optional text encoder for text conditioning</param>
        /// <param name="latentChannels">Number of channels in latent space</param>
        /// <param name="scaleFactor">Scaling factor for latent values</param>
        /// <param name="timesteps">Number of diffusion timesteps</param>
        /// <param name="guidanceScale">Default classifier-free guidance scale</param>
        /// <param name="modelName">Name of the model</param>
        /// <exception cref="ArgumentNullException">Thrown when encoder or decoder is null</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        public LatentDiffusionModel(
            NeuralNetworkArchitecture<double> architecture,
            IAutoencoder encoder,
            IAutoencoder decoder,
            ITextEncoder? textEncoder = null,
            int latentChannels = 4,
            double scaleFactor = 0.18215,
            int timesteps = 1000,
            double guidanceScale = 7.5,
            string modelName = "LatentDiffusionModel")
            : base(timesteps, 0.00085, 0.012, modelName)
        {
            _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
            _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
            
            if (latentChannels <= 0)
                throw new ArgumentException("Latent channels must be positive", nameof(latentChannels));
            if (scaleFactor <= 0)
                throw new ArgumentException("Scale factor must be positive", nameof(scaleFactor));
            if (guidanceScale < 1.0)
                throw new ArgumentException("Guidance scale must be at least 1.0", nameof(guidanceScale));
            
            _textEncoder = textEncoder;
            _latentChannels = latentChannels;
            _scaleFactor = scaleFactor;
            _useConditioning = textEncoder != null;
            _defaultGuidanceScale = guidanceScale;
        }
        
        /// <summary>
        /// Generates an image from a text prompt
        /// </summary>
        /// <param name="prompt">The text prompt describing the desired image</param>
        /// <param name="imageShape">The shape of the output image [batch, channels, height, width]</param>
        /// <param name="guidanceScale">Classifier-free guidance scale (higher = more prompt adherence)</param>
        /// <param name="seed">Optional random seed for reproducibility</param>
        /// <param name="negativePrompt">Optional negative prompt to avoid certain features</param>
        /// <returns>Generated image tensor</returns>
        /// <exception cref="InvalidOperationException">Thrown when text encoder is not configured</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        /// <exception cref="ObjectDisposedException">Thrown when the model has been disposed</exception>
        public Tensor<double> GenerateFromText(
            string prompt, 
            int[] imageShape, 
            double? guidanceScale = null,
            int? seed = null,
            string? negativePrompt = null)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LatentDiffusionModel));
            
            if (!_useConditioning || _textEncoder == null)
                throw new InvalidOperationException("Text encoder not configured for text-to-image generation");
            
            if (string.IsNullOrWhiteSpace(prompt))
                throw new ArgumentException("Prompt cannot be null or empty", nameof(prompt));
            
            ValidateImageShape(imageShape);
            
            guidanceScale ??= _defaultGuidanceScale;
            if (guidanceScale < 1.0)
                throw new ArgumentException("Guidance scale must be at least 1.0", nameof(guidanceScale));
            
            lock (_lockObject)
            {
                try
                {
                    // Encode text prompts
                    var textEmbedding = _textEncoder.Encode(prompt);
                    Tensor<double>? negativeEmbedding = null;
                    
                    if (!string.IsNullOrWhiteSpace(negativePrompt))
                    {
                        negativeEmbedding = _textEncoder.Encode(negativePrompt);
                    }
                    
                    // Calculate latent shape
                    var latentShape = CalculateLatentShape(imageShape);
                    
                    // Generate in latent space
                    var latentSample = GenerateConditional(latentShape, textEmbedding, guidanceScale.Value, seed, negativeEmbedding);
                    
                    // Decode to image space
                    return DecodeLatent(latentSample);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to generate image from text: {ex.Message}", ex);
                }
            }
        }
        
        /// <summary>
        /// Performs image-to-image generation with optional text guidance
        /// </summary>
        /// <param name="inputImage">The input image to transform</param>
        /// <param name="prompt">Optional text prompt for guidance</param>
        /// <param name="strength">Denoising strength (0 = no change, 1 = complete regeneration)</param>
        /// <param name="guidanceScale">Classifier-free guidance scale</param>
        /// <param name="seed">Optional random seed for reproducibility</param>
        /// <returns>Transformed image tensor</returns>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        /// <exception cref="ObjectDisposedException">Thrown when the model has been disposed</exception>
        public Tensor<double> GenerateFromImage(
            Tensor<double> inputImage, 
            string? prompt = null, 
            double strength = 0.75,
            double? guidanceScale = null,
            int? seed = null)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LatentDiffusionModel));
            
            if (inputImage == null)
                throw new ArgumentNullException(nameof(inputImage));
            
            if (strength < _minStrength || strength > _maxStrength)
                throw new ArgumentException($"Strength must be between {_minStrength} and {_maxStrength}", nameof(strength));
            
            guidanceScale ??= _defaultGuidanceScale;
            
            lock (_lockObject)
            {
                try
                {
                    // Encode input image to latent space
                    var latent = EncodeImage(inputImage);
                    
                    // Add noise based on strength
                    var noiseLevel = (int)(strength * Timesteps);
                    var random = seed.HasValue ? new Random(seed.Value) : new Random();
                    
                    var (noisyLatent, _) = ForwardDiffusion(latent, noiseLevel, random);
                    
                    // Encode text if provided
                    Tensor<double>? conditioning = null;
                    if (!string.IsNullOrWhiteSpace(prompt) && _textEncoder != null)
                    {
                        conditioning = _textEncoder.Encode(prompt);
                    }
                    
                    // Denoise from partially noised latent
                    for (int t = noiseLevel; t >= 0; t--)
                    {
                        noisyLatent = ReverseStepConditional(noisyLatent, t, conditioning, guidanceScale.Value, random);
                    }
                    
                    // Decode to image
                    return DecodeLatent(noisyLatent);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to generate image from image: {ex.Message}", ex);
                }
            }
        }
        
        /// <summary>
        /// Performs inpainting to fill masked regions of an image
        /// </summary>
        /// <param name="image">The original image</param>
        /// <param name="mask">Binary mask indicating regions to inpaint (1 = inpaint, 0 = keep)</param>
        /// <param name="prompt">Text prompt describing what to generate in masked regions</param>
        /// <param name="guidanceScale">Classifier-free guidance scale</param>
        /// <param name="seed">Optional random seed for reproducibility</param>
        /// <returns>Inpainted image tensor</returns>
        /// <exception cref="ArgumentNullException">Thrown when required parameters are null</exception>
        /// <exception cref="ArgumentException">Thrown when parameters are invalid</exception>
        /// <exception cref="ObjectDisposedException">Thrown when the model has been disposed</exception>
        public Tensor<double> Inpaint(
            Tensor<double> image, 
            Tensor<double> mask, 
            string prompt,
            double? guidanceScale = null,
            int? seed = null)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LatentDiffusionModel));
            
            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (mask == null)
                throw new ArgumentNullException(nameof(mask));
            if (string.IsNullOrWhiteSpace(prompt))
                throw new ArgumentException("Prompt cannot be null or empty for inpainting", nameof(prompt));
            
            if (!ShapesMatch(image.Shape, mask.Shape))
                throw new ArgumentException("Image and mask shapes must match", nameof(mask));
            
            guidanceScale ??= _defaultGuidanceScale;
            
            lock (_lockObject)
            {
                try
                {
                    // Encode image and mask to latent space
                    var latentImage = EncodeImage(image);
                    var latentMask = DownsampleMask(mask, latentImage.Shape);
                    
                    // Generate new content for masked area
                    var conditioning = _textEncoder?.Encode(prompt);
                    var generated = GenerateConditional(latentImage.Shape, conditioning, guidanceScale.Value, seed);
                    
                    // Blend based on mask
                    var blended = BlendWithMask(latentImage, generated, latentMask);
                    
                    // Decode to image
                    return DecodeLatent(blended);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to inpaint image: {ex.Message}", ex);
                }
            }
        }
        
        /// <summary>
        /// Generates multiple images in parallel
        /// </summary>
        /// <param name="prompts">Array of text prompts</param>
        /// <param name="imageShape">Shape of each output image</param>
        /// <param name="guidanceScale">Classifier-free guidance scale</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Array of generated images</returns>
        public async Task<Tensor<double>[]> GenerateMultipleAsync(
            string[] prompts,
            int[] imageShape,
            double? guidanceScale = null,
            CancellationToken cancellationToken = default)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LatentDiffusionModel));
            
            if (prompts == null || prompts.Length == 0)
                throw new ArgumentException("Prompts array cannot be null or empty", nameof(prompts));
            
            if (prompts.Length > _maxBatchSize)
                throw new ArgumentException($"Batch size cannot exceed {_maxBatchSize}", nameof(prompts));
            
            var tasks = new Task<Tensor<double>>[prompts.Length];
            
            for (int i = 0; i < prompts.Length; i++)
            {
                int index = i;
                tasks[i] = Task.Run(() => GenerateFromText(prompts[index], imageShape, guidanceScale, seed: index), cancellationToken);
            }
            
            return await Task.WhenAll(tasks);
        }
        
        /// <summary>
        /// Trains the latent diffusion model
        /// </summary>
        /// <param name="data">Batch of training images</param>
        /// <param name="optimizer">Optimizer for updating model parameters</param>
        /// <returns>Average loss for the batch</returns>
        public override double TrainStep(Tensor<double> data, IOptimizer<double, Tensor<double>, Tensor<double>> optimizer)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LatentDiffusionModel));
            
            lock (_lockObject)
            {
                // First encode images to latent space
                var latents = EncodeImage(data);
                
                // Scale latents
                latents = latents.Multiply(_scaleFactor);
                
                // Train diffusion in latent space
                return base.TrainStep(latents, optimizer);
            }
        }
        
        /// <summary>
        /// Gets model metadata including latent diffusion specific information
        /// </summary>
        public override ModelMetadata<double> GetModelMetadata()
        {
            var baseMetadata = base.GetModelMetadata();
            
            // Add latent diffusion specific information
            baseMetadata.AdditionalInfo["LatentChannels"] = _latentChannels;
            baseMetadata.AdditionalInfo["ScaleFactor"] = _scaleFactor;
            baseMetadata.AdditionalInfo["UseConditioning"] = _useConditioning;
            baseMetadata.AdditionalInfo["DefaultGuidanceScale"] = _defaultGuidanceScale;
            baseMetadata.AdditionalInfo["HasTextEncoder"] = _textEncoder != null;
            baseMetadata.AdditionalInfo["ModelType"] = "LatentDiffusion";
            
            baseMetadata.Description = $"Latent Diffusion Model with {Timesteps} timesteps, " +
                                     $"{_latentChannels} latent channels, " +
                                     $"conditioning: {_useConditioning}";
            
            return baseMetadata;
        }
        
        private Tensor<double> GenerateConditional(
            int[] shape, 
            Tensor<double>? conditioning, 
            double guidanceScale,
            int? seed,
            Tensor<double>? negativeConditioning = null)
        {
            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Start from noise
            var sample = GenerateNoise(shape, random);
            
            // Reverse diffusion with conditioning
            for (int t = Timesteps - 1; t >= 0; t--)
            {
                sample = ClassifierFreeGuidanceStep(sample, t, conditioning, guidanceScale, random, negativeConditioning);
            }
            
            return sample;
        }
        
        private Tensor<double> ClassifierFreeGuidanceStep(
            Tensor<double> x, 
            int t, 
            Tensor<double>? conditioning, 
            double guidanceScale, 
            Random random,
            Tensor<double>? negativeConditioning = null)
        {
            // Use negative conditioning if provided, otherwise use null (unconditional)
            var unconditionalNoise = PredictNoiseConditional(x, t, negativeConditioning);
            
            Tensor<double> guidedNoise;
            if (conditioning != null && guidanceScale > 1.0)
            {
                // Predict noise with conditioning
                var conditionalNoise = PredictNoiseConditional(x, t, conditioning);
                
                // Apply classifier-free guidance
                guidedNoise = unconditionalNoise.Add(
                    conditionalNoise.Subtract(unconditionalNoise).Multiply(guidanceScale)
                );
            }
            else
            {
                guidedNoise = unconditionalNoise;
            }
            
            // Standard reverse step with guided noise
            return ReverseStepWithNoise(x, t, guidedNoise, random);
        }
        
        private Tensor<double> PredictNoiseConditional(Tensor<double> x, int t, Tensor<double>? conditioning)
        {
            var timestepTensor = CreateTimestepTensor(t, x.Shape[0]);
            
            if (conditioning != null && NoisePredictor is IConditionalModel conditionalModel)
            {
                return conditionalModel.PredictConditional(x, timestepTensor, conditioning);
            }
            
            return PredictNoise(x, timestepTensor);
        }
        
        private Tensor<double> ReverseStepWithNoise(Tensor<double> x, int t, Tensor<double> predictedNoise, Random random)
        {
            // Use the base class ReverseDiffusion method to get the mean
            var mean = ReverseDiffusion(x, t);
            
            if (t > 0)
            {
                // Add noise for all timesteps except the last
                var noise = GenerateNoise(x.Shape, random);
                // Use a small fixed variance for stability
                var variance = 0.0001;
                return mean.Add(noise.Multiply(Math.Sqrt(variance)));
            }
            
            return mean;
        }
        
        private Tensor<double> EncodeImage(Tensor<double> image)
        {
            return _encoder.Encode(image).Multiply(_scaleFactor);
        }
        
        private Tensor<double> DecodeLatent(Tensor<double> latent)
        {
            return _decoder.Decode(latent.Divide(_scaleFactor));
        }
        
        private int[] CalculateLatentShape(int[] imageShape)
        {
            ValidateImageShape(imageShape);
            
            // Assuming 8x downsampling factor (common for VAE encoders)
            const int downsampleFactor = 8;
            
            return new int[]
            {
                imageShape[0], // batch size
                _latentChannels,
                imageShape[2] / downsampleFactor,
                imageShape[3] / downsampleFactor
            };
        }
        
        private void ValidateImageShape(int[] shape)
        {
            if (shape == null || shape.Length != 4)
                throw new ArgumentException("Image shape must have 4 dimensions [batch, channels, height, width]");
            
            if (shape[0] <= 0 || shape[0] > _maxBatchSize)
                throw new ArgumentException($"Batch size must be between 1 and {_maxBatchSize}");
            
            if (shape[2] % 8 != 0 || shape[3] % 8 != 0)
                throw new ArgumentException("Image dimensions must be divisible by 8 for VAE encoding");
        }
        
        private bool ShapesMatch(int[] shape1, int[] shape2)
        {
            if (shape1.Length != shape2.Length)
                return false;
            
            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                    return false;
            }
            
            return true;
        }
        
        private Tensor<double> DownsampleMask(Tensor<double> mask, int[] targetShape)
        {
            // Simple nearest neighbor downsampling
            // In production, use proper interpolation (e.g., bilinear)
            var result = new Tensor<double>(targetShape);
            
            var scaleH = mask.Shape[2] / targetShape[2];
            var scaleW = mask.Shape[3] / targetShape[3];
            
            for (int b = 0; b < targetShape[0]; b++)
            {
                for (int c = 0; c < targetShape[1]; c++)
                {
                    for (int h = 0; h < targetShape[2]; h++)
                    {
                        for (int w = 0; w < targetShape[3]; w++)
                        {
                            var srcH = h * scaleH;
                            var srcW = w * scaleW;
                            var idx = new[] { b, c, h, w };
                            var srcIdx = new[] { b, Math.Min(c, mask.Shape[1] - 1), srcH, srcW };
                            result[idx] = mask[srcIdx];
                        }
                    }
                }
            }
            
            return result;
        }
        
        private Tensor<double> BlendWithMask(Tensor<double> original, Tensor<double> generated, Tensor<double> mask)
        {
            // blend = mask * generated + (1 - mask) * original
            var invertedMask = mask.Multiply(-1).Add(1); // 1 - mask
            return generated.Multiply(mask).Add(original.Multiply(invertedMask));
        }
        
        private Tensor<double> ReverseStepConditional(
            Tensor<double> x, 
            int t, 
            Tensor<double>? conditioning, 
            double guidanceScale,
            Random random)
        {
            var noise = conditioning != null
                ? PredictNoiseConditional(x, t, conditioning)
                : PredictNoise(x, CreateTimestepTensor(t, x.Shape[0]));
                
            if (conditioning != null && guidanceScale > 1.0)
            {
                var unconditionalNoise = PredictNoise(x, CreateTimestepTensor(t, x.Shape[0]));
                noise = unconditionalNoise.Add(
                    noise.Subtract(unconditionalNoise).Multiply(guidanceScale)
                );
            }
                
            return ReverseStepWithNoise(x, t, noise, random);
        }
        
        private Tensor<double> CreateTimestepTensor(int t, int batchSize)
        {
            var timestepData = new double[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                timestepData[i] = t;
            }
            return new Tensor<double>(new[] { batchSize }, new Vector<double>(timestepData));
        }
        
        private Tensor<double> PredictNoise(Tensor<double> noisyData, Tensor<double> timestep)
        {
            if (NoisePredictor == null)
                throw new InvalidOperationException("Noise predictor not set");
                
            // In practice, concatenate or condition the data with timestep
            return NoisePredictor.Predict(noisyData);
        }
        
        private Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            var totalElements = shape.Aggregate(1, (acc, dim) => acc * dim);
            var noiseData = new double[totalElements];
            
            // Generate samples from standard normal distribution using Box-Muller transform
            for (int i = 0; i < totalElements; i += 2)
            {
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                
                var radius = Math.Sqrt(-2.0 * Math.Log(u1));
                var theta = 2.0 * Math.PI * u2;
                
                noiseData[i] = radius * Math.Cos(theta);
                if (i + 1 < totalElements)
                {
                    noiseData[i + 1] = radius * Math.Sin(theta);
                }
            }
            
            return new Tensor<double>(shape, new Vector<double>(noiseData));
        }
        
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            base.SerializeNetworkSpecificData(writer);
            
            // Write latent diffusion specific data
            writer.Write(_latentChannels);
            writer.Write(_scaleFactor);
            writer.Write(_useConditioning);
            writer.Write(_defaultGuidanceScale);
        }
        
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            base.DeserializeNetworkSpecificData(reader);
            
            // Read latent diffusion specific data
            var latentChannels = reader.ReadInt32();
            var scaleFactor = reader.ReadDouble();
            var useConditioning = reader.ReadBoolean();
            var guidanceScale = reader.ReadDouble();
            
            // Validate against current instance
            if (latentChannels != _latentChannels || 
                Math.Abs(scaleFactor - _scaleFactor) > 1e-10 ||
                useConditioning != _useConditioning)
            {
                throw new InvalidOperationException("Serialized model has different configuration");
            }
        }
        
        /// <summary>
        /// Disposes of the latent diffusion model and its resources
        /// </summary>
        public new void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        
        /// <summary>
        /// Releases resources used by the latent diffusion model
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    lock (_lockObject)
                    {
                        // Dispose encoders/decoders if they implement IDisposable
                        if (_encoder is IDisposable disposableEncoder)
                        {
                            disposableEncoder.Dispose();
                        }
                        
                        if (_decoder is IDisposable disposableDecoder)
                        {
                            disposableDecoder.Dispose();
                        }
                        
                        if (_textEncoder is IDisposable disposableTextEncoder)
                        {
                            disposableTextEncoder.Dispose();
                        }
                    }
                }
                
                _disposed = true;
                base.Dispose(disposing);
            }
        }
    }
}