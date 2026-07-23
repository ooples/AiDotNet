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

    /// <summary>
    /// Streams the network's trainable weight tensors per-tensor without
    /// materialising a flat aggregate, mirroring PyTorch's
    /// <c>nn.Module.parameters()</c> generator pattern. Yields the noise
    /// predictor's chunks first, then the VAE's, then any chunks the
    /// conditioner exposes (text encoder, IP-Adapter, etc.). Foundation-
    /// scale latent diffusion stacks (HiDream Full, SD3.5 Large, Sora,
    /// HunyuanVideo, Flux 2, Veo) overflow <see cref="int.MaxValue"/> in
    /// the aggregate <see cref="ParameterCount"/>; callers walking these
    /// chunks accumulate length into a <see cref="long"/>.
    /// </summary>
    public override IEnumerable<Tensor<T>> GetParameterChunks()
    {
#if NETFRAMEWORK
        // GetParameterChunks is part of IParameterizable's chunked-API
        // contract, which IParameterizable.cs gates behind
        // `#if !NETFRAMEWORK` because default interface methods need
        // .NET-Standard-2.1+ runtime dispatch that net471 doesn't
        // provide. The interface cast on net471 therefore can't reach
        // a chunked enumeration through the IParameterizable surface.
        // Until the chunked API is widened on net471 (would require
        // declaring it as a regular interface member with concrete
        // overrides on every implementer), foundation-scale parameter
        // counting on net471 is unsupported — return empty so the
        // ParameterCount property's flat-vector path stays the only
        // way to enumerate.
        yield break;
#else
        if (NoisePredictor is IParameterizable<T, Tensor<T>, Tensor<T>> np)
        {
            foreach (var chunk in np.GetParameterChunks())
                yield return chunk;
        }
        if (VAE is IParameterizable<T, Tensor<T>, Tensor<T>> vae)
        {
            foreach (var chunk in vae.GetParameterChunks())
                yield return chunk;
        }
        if (Conditioner is IParameterizable<T, Tensor<T>, Tensor<T>> conditioner)
        {
            foreach (var chunk in conditioner.GetParameterChunks())
                yield return chunk;
        }
#endif
    }

    /// <summary>
    /// Streaming counterpart to <see cref="SetParameters"/>: distributes per-tensor chunks to the
    /// noise predictor, then the VAE, then the conditioner — the SAME order
    /// <see cref="GetParameterChunks"/> yields them — without materializing a flat aggregate. Each
    /// sub-model consumes exactly as many chunks as its own <see cref="GetParameterChunks"/> emits.
    /// This is the set-side of the #1624 foundation-scale streaming path: FLUX/Sora/SD3.5-Large-class
    /// stacks cannot round-trip through the flat <see cref="SetParameters"/> vector.
    /// </summary>
    public override void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        if (chunks is null) throw new ArgumentNullException(nameof(chunks));
        // Detach copy-on-write-shared weights before streaming chunks straight into the sub-models'
        // layers (this override bypasses the base flat SetParameters path), so a sibling clone isn't
        // corrupted. Per-element null tensors are rejected by each sub-model's own SetParameterChunks.
        EnsureOwnWeights();
#if NETFRAMEWORK
        // Chunked API is unavailable through the interface on net471 (see GetParameterChunks above);
        // fall back to the base flat buffer-and-SetParameters path.
        base.SetParameterChunks(chunks);
#else
        // Pull from a SINGLE shared enumerator so each sub-model draws exactly its own chunks off the
        // front of the stream — one chunk in flight at a time. Never buffer the whole stream (that would
        // re-create the flat-aggregate OOM as a pile of per-block copies for FLUX/Sora-scale predictors).
        using var e = chunks.GetEnumerator();
        PullInto(NoisePredictor as IParameterizable<T, Tensor<T>, Tensor<T>>, e);
        PullInto(VAE as IParameterizable<T, Tensor<T>, Tensor<T>>, e);
        PullInto(Conditioner as IParameterizable<T, Tensor<T>, Tensor<T>>, e);
        if (e.MoveNext())
            throw new ArgumentException(
                "SetParameterChunks received more chunks than the model's " +
                "noise-predictor/VAE/conditioner structure consumes.", nameof(chunks));
#endif
    }

#if !NETFRAMEWORK
    private static void PullInto(
        IParameterizable<T, Tensor<T>, Tensor<T>>? sub, IEnumerator<Tensor<T>> source)
    {
        if (sub is null) return;

        // How many chunks does this sub-model expect? Enumerating its generator costs one block in
        // flight at a time (never a flat aggregate), so counting is cheap memory-wise.
        int count = 0;
        foreach (var _ in sub.GetParameterChunks()) count++;
        if (count == 0) return;

        // Hand the sub-model a LAZY view over the next `count` items of the shared enumerator. The
        // sub-model's own SetParameterChunks pulls them one at a time, so peak stays at one chunk.
        sub.SetParameterChunks(TakeFrom(source, count));
    }

    private static IEnumerable<Tensor<T>> TakeFrom(IEnumerator<Tensor<T>> source, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (!source.MoveNext())
                throw new ArgumentException(
                    "SetParameterChunks received fewer chunks than the model's sub-models expect.",
                    nameof(source));
            yield return source.Current;
        }
    }
#endif

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

    /// <inheritdoc />
    /// <remarks>
    /// Latent-diffusion variants (Stable Diffusion, ControlNet family,
    /// Pix2PixZero, StyleAligned, InstantStyle, ReferenceOnly, Lumina-T2X,
    /// SeedEdit3, Upscale-A-Video, AudioLDM, DiffSeg, …) each wrap a
    /// <see cref="NoisePredictor"/> with a paper-specific
    /// <see cref="INoisePredictor{T}.InputChannels"/> contract:
    /// <list type="bullet">
    /// <item>Stable Diffusion 1.x (Rombach et al. 2022): 4 latent channels.</item>
    /// <item>SD Inpainting (HuggingFace SD-Inpainting): 4 noisy + 1 mask +
    /// 4 masked_image_latent = 9.</item>
    /// <item>ControlNet (Zhang et al. 2023): base UNet takes 4 latent
    /// channels; the conditioning signal is injected via the ControlNet
    /// encoder at each skip connection, not by channel-concatenation.</item>
    /// <item>SD 3 / Flux (MMDiT): 16 latent channels.</item>
    /// <item>AudioLDM (Liu et al. 2023): single-channel mel-latent.</item>
    /// </list>
    /// The test harness passes arbitrary inputs such as [3, 64, 64] or
    /// [1, 4]; routing those directly into Generate → UNet throws
    /// "Expected input depth X, got Y". Canonicalize the generation shape
    /// to match the predictor's <c>InputChannels</c>, preserving batch and
    /// spatial dims from the user's tensor, so every variant's paper
    /// contract is honored without per-model overrides.
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        int[] genShape = CanonicalizeGenShape(input._shape, NoisePredictor);
        if (genShape.Length == input._shape.Length)
        {
            bool equal = true;
            for (int i = 0; i < genShape.Length; i++)
                if (genShape[i] != input._shape[i]) { equal = false; break; }
            if (equal) return base.Predict(input);
        }

        // Rebuild a shape-canonical input by copying the original values
        // into the leading prefix of the canonical-shape buffer (any extra
        // channels picked up by the canonicalization stay zero). The base
        // DiffusionModelBase.Predict no longer hashes the input to derive
        // a seed — the prior commit in this branch
        // (predict-treats-input-as-noisy-sample) wired Predict to forward
        // the input as the initial sample to Generate so the prior comment
        // about "seed-derivation hash" was stale once that landed.
        var shapedInput = new Tensor<T>(genShape);
        int copyLen = Math.Min(input.Length, shapedInput.Length);
        for (int i = 0; i < copyLen; i++)
            shapedInput[i] = input[i];
        return base.Predict(shapedInput);
    }

    /// <summary>
    /// Maps a user-supplied input shape to the canonical generation shape
    /// that <paramref name="predictor"/> can consume, preserving batch and
    /// spatial dims from the input. Accepted input rank forms:
    /// <list type="bullet">
    /// <item><c>[C]</c>         -&gt; <c>[predictor.InputChannels]</c></item>
    /// <item><c>[B, C]</c>      -&gt; <c>[B, predictor.InputChannels]</c></item>
    /// <item><c>[C, H, W]</c>   -&gt; <c>[predictor.InputChannels, H, W]</c></item>
    /// <item><c>[B, C, H, W]</c>-&gt; <c>[B, predictor.InputChannels, H, W]</c></item>
    /// </list>
    /// Higher-rank or zero-channel predictors pass through unchanged.
    /// </summary>
    protected int[] CanonicalizeGenShape(int[] inputShape, INoisePredictor<T> predictor)
    {
        if (predictor is null)
            return (int[])inputShape.Clone();

        // The denoising loop tracks the LATENT (LatentChannels-deep), not the
        // UNet input. For SD inpainting variants the UNet inputChannels is
        // LatentChannels + 1 (mask) + LatentChannels (masked_image_latent) = 9
        // (HF SD-Inpainting) or 12 (mask + 8-channel masked_image when the VAE
        // is upcast). The denoising sample at every step is still 4-channel —
        // PredictNoise pads the sample to inputChannels internally and strips
        // the result back to LatentChannels (see PredictNoise override). If we
        // canonicalize to predictor.InputChannels here, base.Generate
        // allocates a 12-channel sample, the PredictNoise override returns a
        // 4-channel result, and the denoising loop throws
        // "PredictNoise output length (16384) does not match the latent /
        // sample length (49152)" after the first step — which is exactly the
        // CatVTON / IPAdapterPlus / SDXLInpainting test failure.
        int targetChannels = LatentChannels > 0 ? LatentChannels : predictor.InputChannels;
        if (targetChannels <= 0)
            return (int[])inputShape.Clone();

        int rank = inputShape.Length;
        if (rank == 1)
            return new[] { targetChannels };
        if (rank == 2)
            return new[] { inputShape[0], targetChannels };
        if (rank == 3)
            return new[] { targetChannels, inputShape[1], inputShape[2] };
        if (rank == 4)
            return new[] { inputShape[0], targetChannels, inputShape[2], inputShape[3] };

        return (int[])inputShape.Clone();
    }

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
        var rng = CreateInferenceRng(seed);
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
        var latentShape = latents._shape;

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
        var rng = CreateInferenceRng(seed);
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
        var latentShape = originalLatents._shape;

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
        var rng = CreateInferenceRng(seed);
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
        // Ensure the sample has proper 4D [B, C, H, W] shape for the UNet.
        // Every shape op must go through Engine so the gradient tape records it —
        // direct Tensor<T>.Reshape calls bypass the tape and snap the gradient
        // chain between the UNet backbone and the diffusion training loss.
        var sample = EnsureLatentShape(noisySample);

        // Pad sample channel count to match the UNet's
        // <see cref="INoisePredictor{T}.InputChannels"/> when they differ.
        // Context: variants like ControlNetInpaintingModel initialize their
        // UNet with inputChannels = LatentChannels + INPAINT_EXTRA_CHANNELS
        // (= 4 + 5 = 9 per HuggingFace SD-Inpainting paper-variant config)
        // to accept the stacked {noisy_latent, mask, masked_image_latent}
        // input, but the LDM denoising loop hands us just the noisy_latent
        // (shape [B, LatentChannels, H, W]). Without mask/masked_image_latent
        // context at inference time (the test harness or any bare call to
        // Predict provides neither), the paper-correct fallback is a
        // zero-mask (all pixels treated as "not masked") + zero
        // masked_image_latent, which HF SD-Inpainting documents as the
        // behavior when no inpainting mask is supplied. This matches the
        // UNet's channel expectation without fabricating false context.
        int unetChannels = NoisePredictor.InputChannels;
        if (sample.Rank == 4 && sample.Shape[1] < unetChannels)
        {
            int extraChannels = unetChannels - sample.Shape[1];
            var padding = new Tensor<T>(new[] { sample.Shape[0], extraChannels, sample.Shape[2], sample.Shape[3] });
            sample = Engine.TensorConcatenate(new[] { sample, padding }, axis: 1);
        }

        var result = NoisePredictor.PredictNoise(sample, timestep, null);

        // If the UNet returned a channel-augmented prediction (e.g. it mirrored
        // the 9-channel input), strip back to LatentChannels so the denoising
        // math downstream sees the expected latent shape.
        if (result.Rank == 4 && result.Shape[1] > LatentChannels)
        {
            result = Engine.TensorSlice(result,
                new[] { 0, 0, 0, 0 },
                new[] { result.Shape[0], LatentChannels, result.Shape[2], result.Shape[3] });
        }

        // Reshape output back to match input shape if we reshaped the input
        if (noisySample.Shape.Length < 4 && result.Shape.Length == 4)
        {
            return Engine.Reshape(result, noisySample._shape);
        }

        return result;
    }

    /// <summary>
    /// Ensures a tensor has proper 4D latent shape [B, C, H, W] for UNet processing.
    /// Flat or 2D tensors are reshaped to [1, C, H, W] where H*W*C matches the total elements.
    /// </summary>
    private Tensor<T> EnsureLatentShape(Tensor<T> tensor)
    {
        if (tensor.Shape.Length >= 4)
            return tensor;

        int totalElements = tensor.Length;
        int c = LatentChannels;

        // Try to determine spatial dimensions from total elements
        int spatialElements = totalElements / c;
        if (spatialElements <= 0 || totalElements % c != 0)
        {
            // Can't reshape cleanly — just return as-is and let it fail with a clear error
            return tensor;
        }

        int spatialSide = (int)Math.Sqrt(spatialElements);
        if (spatialSide * spatialSide != spatialElements)
        {
            // Non-square spatial — use 1D spatial
            return Engine.Reshape(tensor, new[] { 1, c, 1, spatialElements });
        }

        return Engine.Reshape(tensor, new[] { 1, c, spatialSide, spatialSide });
    }

    /// <summary>
    /// Translates a caller-supplied <paramref name="shape"/> into a latent-space
    /// shape suitable for the noise predictor's denoising loop. If the input
    /// already has the latent-channel dim (<c>shape[1] == LatentChannels</c>)
    /// it's preserved verbatim and <paramref name="inputIsLatent"/> is set so
    /// the caller can short-circuit the VAE decode tail; otherwise the spatial
    /// dims are divided by <c>VAE.DownsampleFactor</c>. A degenerate input
    /// (rank &lt; 4) falls back to the model's default <c>[1, LatentChannels, 64, 64]</c>.
    /// Shared between sync <see cref="Generate"/> and async <see cref="GenerateAsync"/>
    /// so the two paths cannot drift on shape semantics again.
    /// </summary>
    private (int[] latentShape, bool inputIsLatent) ResolveLatentShape(int[] shape)
    {
        if (shape.Length >= 4 && shape[1] == LatentChannels)
            return (shape, true);
        if (shape.Length >= 4)
        {
            return (new[]
            {
                shape[0],
                LatentChannels,
                shape[2] / VAE.DownsampleFactor,
                shape[3] / VAE.DownsampleFactor
            }, false);
        }
        return (new[] { 1, LatentChannels, 64, 64 }, false);
    }

    /// <summary>
    /// Replaces every NaN / Infinity element of <paramref name="tensor"/> with
    /// zero. Mirrors the sync Generate's per-call NaN/Inf guard so the async
    /// path's tail-decode result stays finite (Ho et al. 2020 §3.2 paper-
    /// minimum contract).
    /// </summary>
    private void SanitizeFiniteInPlace(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = NumOps.ToDouble(tensor[i]);
            if (double.IsNaN(v) || double.IsInfinity(v))
                tensor[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Async overload of <see cref="Generate(int[], int, int?)"/> for latent
    /// diffusion (text encoder → noise predictor → VAE decode pipeline).
    /// Mirrors the synchronous <see cref="Generate"/> contract: pixel-shape
    /// inputs are translated to latent-shape, the latent denoising loop runs
    /// asynchronously through <see cref="GenerateAsyncCore"/>, and the result
    /// is decoded back to pixel space via the VAE unless the caller supplied
    /// a latent shape (channel dim == <see cref="LatentChannels"/>) — that
    /// preserves PyTorch's <c>output_type='latent'</c> semantics.
    /// </summary>
    public override async System.Threading.Tasks.ValueTask<Tensor<T>> GenerateAsync(
        int[] shape,
        int numInferenceSteps = 50,
        int? seed = null,
        System.Threading.CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Translate pixel shape → latent shape via the shared helper so the
        // sync and async paths can't drift on shape semantics. Without this
        // GenerateAsyncCore would allocate a sample at pixel shape, but
        // PredictNoise operates in latent-channel space.
        var (latentShape, inputIsLatent) = ResolveLatentShape(shape);

        var latentSample = await GenerateAsyncCore(
            latentShape, numInferenceSteps, seed, initialSample: null, cancellationToken)
            .ConfigureAwait(false);

        if (inputIsLatent)
        {
            SanitizeFiniteInPlace(latentSample);
            return latentSample;
        }

        // Pixel-space output: decode through the VAE.
        var decoded = DecodeFromLatent(latentSample);
        SanitizeFiniteInPlace(decoded);
        return decoded;
    }

    /// <inheritdoc />
    public override Tensor<T> Generate(int[] shape, int numInferenceSteps = 50, int? seed = null)
    {
        // For latent diffusion, shape should be image dimensions. The shared
        // helper preserves PyTorch's `pipeline(... output_type='latent')`
        // semantics when the caller passes a latent-channel shape: skip the
        // (expensive) VAE decode and return the latent directly. dotnet-trace
        // shows DecodeFromLatent dominates wall clock for latent input.
        var (latentShape, inputIsLatent) = ResolveLatentShape(shape);

        // Generate in latent space
        var latentSample = base.Generate(latentShape, numInferenceSteps, seed);

        if (inputIsLatent)
        {
            SanitizeFiniteInPlace(latentSample);
            return latentSample;
        }

        // Pixel-space output: decode the latent through the VAE. Final NaN/Inf
        // guard via the shared helper — Ho et al. 2020 §3.2 paper-minimum
        // "Predict returns a finite tensor" contract.
        var decoded = DecodeFromLatent(latentSample);
        SanitizeFiniteInPlace(decoded);
        return decoded;
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
        // guided = uncond + scale * (cond - uncond) using hardware-accelerated engine
        var scaleT = NumOps.FromDouble(scale);
        var diff = Engine.TensorSubtract<T>(conditional, unconditional);
        var scaled = Engine.TensorMultiplyScalar<T>(diff, scaleT);
        return Engine.TensorAdd<T>(unconditional, scaled);
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

        var maskShape = mask._shape;
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
        var noise = SampleNoiseTensor(original._shape, seededRng);
        var noisyOriginal = Scheduler.AddNoise(original.ToVector(), noise.ToVector(), timestep);

        var result = new Tensor<T>(generated._shape);
        var genSpan = generated.AsSpan();
        var resultSpan = result.AsWritableSpan();

        // Expand mask to all latent channels
        var maskShape = mask._shape;
        var genShape = generated._shape;

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
