using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Base class for latent diffusion models that operate in a compressed latent space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all latent diffusion models,
/// including encoding/decoding, text-to-image generation, image-to-image transformation, and inpainting.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation for latent diffusion models like Stable Diffusion.
/// It combines a VAE (for compression), a noise predictor (for denoising), and optional conditioning
/// (for guided generation from text or images).
/// </para>
/// </remarks>
public abstract class LatentDiffusionModelBase<T> : DiffusionModelBase<T>, ILatentDiffusionModel<T>
{
    /// <summary>
    /// The default guidance scale for classifier-free guidance.
    /// </summary>
    private double _guidanceScale = 7.5;

    /// <inheritdoc />
    public abstract IVAEModel<T> VAE { get; }

    /// <inheritdoc />
    public abstract INoisePredictor<T> NoisePredictor { get; }

    /// <inheritdoc />
    public abstract IConditioningModule<T>? Conditioner { get; }

    /// <inheritdoc />
    public abstract int LatentChannels { get; }

    /// <inheritdoc />
    public virtual double GuidanceScale => _guidanceScale;

    /// <inheritdoc />
    public virtual bool SupportsNegativePrompt => Conditioner != null && NoisePredictor.SupportsCFG;

    /// <inheritdoc />
    public virtual bool SupportsInpainting => true;

    /// <summary>
    /// Initializes a new instance of the LatentDiffusionModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    protected LatentDiffusionModelBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(options, scheduler, architecture)
    {
    }

    #region ILatentDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> EncodeToLatent(Tensor<T> image, bool sampleMode = true)
    {
        var latent = VAE.Encode(image, sampleMode);
        return VAE.ScaleLatent(latent);
    }

    /// <inheritdoc />
    public virtual Tensor<T> DecodeFromLatent(Tensor<T> latent)
    {
        var unscaled = VAE.UnscaleLatent(latent);
        return VAE.Decode(unscaled);
    }

    /// <inheritdoc />
    public virtual Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        if (Conditioner == null)
            throw new InvalidOperationException("Text-to-image generation requires a conditioning module.");

        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var useCFG = effectiveGuidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / VAE.DownsampleFactor;
        var latentWidth = width / VAE.DownsampleFactor;
        var latentShape = new[] { 1, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                // Classifier-free guidance: predict with and without conditioning
                var condPred = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = NoisePredictor.PredictNoise(latents, timestep, negativeEmbedding);

                // Guided prediction = uncond + guidance_scale * (cond - uncond)
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to image
        return DecodeFromLatent(latents);
    }

    /// <inheritdoc />
    public virtual Tensor<T> ImageToImage(
        Tensor<T> inputImage,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.8,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        if (Conditioner == null)
            throw new InvalidOperationException("Image-to-image generation requires a conditioning module.");

        strength = MathPolyfill.Clamp(strength, 0.0, 1.0);

        // If strength is 0, no modification is needed - return the input image
        if (strength <= 0.0)
        {
            return inputImage;
        }

        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var useCFG = effectiveGuidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode input image to latent
        var latents = EncodeToLatent(inputImage, sampleMode: false);
        var latentShape = latents.Shape;

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate starting timestep based on strength
        Scheduler.SetTimesteps(numInferenceSteps);
        var startStep = (int)(numInferenceSteps * (1.0 - strength));

        // If startStep equals numInferenceSteps, there's nothing to denoise
        if (startStep >= numInferenceSteps)
        {
            return DecodeFromLatent(latents);
        }

        var startTimestep = Scheduler.Timesteps.Skip(startStep).First();

        // Add noise to latents at starting timestep
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var noise = SampleNoiseTensor(latentShape, rng);
        var noisyLatents = Scheduler.AddNoise(latents.ToVector(), noise.ToVector(), startTimestep);
        latents = new Tensor<T>(latentShape, noisyLatents);

        // Denoising loop (starting from startStep)
        foreach (var timestep in Scheduler.Timesteps.Skip(startStep))
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = NoisePredictor.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        return DecodeFromLatent(latents);
    }

    /// <inheritdoc />
    public virtual Tensor<T> Inpaint(
        Tensor<T> inputImage,
        Tensor<T> mask,
        string prompt,
        string? negativePrompt = null,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        if (Conditioner == null)
            throw new InvalidOperationException("Inpainting requires a conditioning module.");

        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var useCFG = effectiveGuidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode input image to latent
        var originalLatents = EncodeToLatent(inputImage, sampleMode: false);
        var latentShape = originalLatents.Shape;

        // Resize mask to latent size
        var latentMask = ResizeMaskToLatent(mask, latentShape);

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = NoisePredictor.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);

            // Blend with original latents in unmasked regions
            latents = BlendLatentsWithMask(latents, originalLatents, latentMask, timestep);
        }

        return DecodeFromLatent(latents);
    }

    /// <inheritdoc />
    public virtual void SetGuidanceScale(double scale)
    {
        if (scale < 0)
            throw new ArgumentOutOfRangeException(nameof(scale), "Guidance scale must be non-negative.");
        _guidanceScale = scale;
    }

    #endregion

    #region IDiffusionModel<T> Override

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        // Use noise predictor without conditioning
        return NoisePredictor.PredictNoise(noisySample, timestep, null);
    }

    /// <inheritdoc />
    public override Tensor<T> Generate(int[] shape, int numInferenceSteps = 50, int? seed = null)
    {
        // For latent diffusion, shape should be image dimensions
        // If latent shape is provided, use it directly; otherwise assume image dimensions
        int[] latentShape;

        if (shape.Length >= 4 && shape[1] == LatentChannels)
        {
            // Already latent shape
            latentShape = shape;
        }
        else if (shape.Length >= 4)
        {
            // Image shape, convert to latent
            latentShape = new[]
            {
                shape[0],
                LatentChannels,
                shape[2] / VAE.DownsampleFactor,
                shape[3] / VAE.DownsampleFactor
            };
        }
        else
        {
            // Default shape
            latentShape = new[] { 1, LatentChannels, 64, 64 };
        }

        // Generate in latent space
        var latentSample = base.Generate(latentShape, numInferenceSteps, seed);

        // Decode to image
        return DecodeFromLatent(latentSample);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Applies classifier-free guidance to combine conditional and unconditional predictions.
    /// </summary>
    /// <param name="unconditional">The unconditional noise prediction.</param>
    /// <param name="conditional">The conditional noise prediction.</param>
    /// <param name="scale">The guidance scale.</param>
    /// <returns>The guided noise prediction.</returns>
    protected virtual Tensor<T> ApplyGuidance(Tensor<T> unconditional, Tensor<T> conditional, double scale)
    {
        var result = new Tensor<T>(unconditional.Shape);
        var uncondSpan = unconditional.AsSpan();
        var condSpan = conditional.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            // guided = uncond + scale * (cond - uncond)
            var diff = NumOps.Subtract(condSpan[i], uncondSpan[i]);
            resultSpan[i] = NumOps.Add(uncondSpan[i], NumOps.Multiply(scaleT, diff));
        }

        return result;
    }

    /// <summary>
    /// Samples a noise tensor from standard normal distribution.
    /// </summary>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>A tensor filled with Gaussian noise.</returns>
    protected virtual Tensor<T> SampleNoiseTensor(int[] shape, Random rng)
    {
        var noise = new Tensor<T>(shape);
        var noiseSpan = noise.AsWritableSpan();

        // Box-Muller transform
        for (int i = 0; i < noiseSpan.Length; i += 2)
        {
            var u1 = rng.NextDouble();
            var u2 = rng.NextDouble();

            while (u1 <= double.Epsilon)
                u1 = rng.NextDouble();

            var mag = Math.Sqrt(-2.0 * Math.Log(u1));
            var z0 = mag * Math.Cos(2.0 * Math.PI * u2);
            var z1 = mag * Math.Sin(2.0 * Math.PI * u2);

            noiseSpan[i] = NumOps.FromDouble(z0);
            if (i + 1 < noiseSpan.Length)
                noiseSpan[i + 1] = NumOps.FromDouble(z1);
        }

        return noise;
    }

    /// <summary>
    /// Resizes a mask tensor to match latent dimensions.
    /// </summary>
    /// <param name="mask">The original mask [batch, 1, height, width].</param>
    /// <param name="latentShape">The target latent shape.</param>
    /// <returns>The resized mask matching latent dimensions.</returns>
    protected virtual Tensor<T> ResizeMaskToLatent(Tensor<T> mask, int[] latentShape)
    {
        // Simple nearest-neighbor downsampling
        var latentHeight = latentShape[2];
        var latentWidth = latentShape[3];
        var result = new Tensor<T>(new[] { latentShape[0], 1, latentHeight, latentWidth });

        var maskShape = mask.Shape;
        var inputHeight = maskShape[2];
        var inputWidth = maskShape[3];

        var scaleH = (double)inputHeight / latentHeight;
        var scaleW = (double)inputWidth / latentWidth;

        for (int b = 0; b < latentShape[0]; b++)
        {
            for (int h = 0; h < latentHeight; h++)
            {
                for (int w = 0; w < latentWidth; w++)
                {
                    var srcH = (int)(h * scaleH);
                    var srcW = (int)(w * scaleW);
                    result[b, 0, h, w] = mask[b, 0, Math.Min(srcH, inputHeight - 1), Math.Min(srcW, inputWidth - 1)];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Blends generated latents with original latents based on mask for inpainting.
    /// </summary>
    /// <param name="generated">The generated latents.</param>
    /// <param name="original">The original latents.</param>
    /// <param name="mask">The mask (1 = inpaint, 0 = keep original).</param>
    /// <param name="timestep">Current timestep for noise addition to original.</param>
    /// <returns>Blended latents.</returns>
    protected virtual Tensor<T> BlendLatentsWithMask(Tensor<T> generated, Tensor<T> original, Tensor<T> mask, int timestep)
    {
        // Add noise to original latents at current timestep
        // Use a seeded RNG based on timestep for consistency across calls
        // This ensures the same noise is used for the same timestep during denoising
        var seededRng = RandomHelper.CreateSeededRandom(timestep);
        var noise = SampleNoiseTensor(original.Shape, seededRng);
        var noisyOriginal = Scheduler.AddNoise(original.ToVector(), noise.ToVector(), timestep);

        var result = new Tensor<T>(generated.Shape);
        var genSpan = generated.AsSpan();
        var resultSpan = result.AsWritableSpan();

        // Expand mask to all latent channels
        var maskShape = mask.Shape;
        var genShape = generated.Shape;

        for (int b = 0; b < genShape[0]; b++)
        {
            for (int c = 0; c < genShape[1]; c++)
            {
                for (int h = 0; h < genShape[2]; h++)
                {
                    for (int w = 0; w < genShape[3]; w++)
                    {
                        var idx = b * genShape[1] * genShape[2] * genShape[3] +
                                  c * genShape[2] * genShape[3] +
                                  h * genShape[3] + w;
                        var maskIdx = b * maskShape[2] * maskShape[3] +
                                      h * maskShape[3] + w;

                        var maskVal = NumOps.ToDouble(mask.AsSpan()[maskIdx]);
                        var origVal = noisyOriginal[idx];
                        var genVal = genSpan[idx];

                        // Blend: mask * generated + (1 - mask) * noisy_original
                        var maskT = NumOps.FromDouble(maskVal);
                        var invMask = NumOps.FromDouble(1.0 - maskVal);
                        resultSpan[idx] = NumOps.Add(
                            NumOps.Multiply(maskT, genVal),
                            NumOps.Multiply(invMask, origVal));
                    }
                }
            }
        }

        return result;
    }

    #endregion
}
