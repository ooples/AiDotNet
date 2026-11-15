using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// A camera relative to the reference view, as DreamGaussian passes it to its guidance models
/// (<c>vers, hors, radii</c>).
/// </summary>
public readonly struct GuidanceViewOffset
{
    /// <summary>Creates a view offset.</summary>
    /// <param name="elevation">Elevation offset in degrees (negative is above, the reference's convention).</param>
    /// <param name="azimuth">Azimuth offset in degrees.</param>
    /// <param name="radius">Camera distance offset; DreamGaussian always uses 0.</param>
    public GuidanceViewOffset(double elevation, double azimuth, double radius)
    {
        Elevation = elevation;
        Azimuth = azimuth;
        Radius = radius;
    }

    /// <summary>Elevation offset in degrees.</summary>
    public double Elevation { get; }

    /// <summary>Azimuth offset in degrees.</summary>
    public double Azimuth { get; }

    /// <summary>Camera distance offset.</summary>
    public double Radius { get; }
}

/// <summary>
/// A frozen 2D diffusion prior that guides a 3D representation: the gradient of Score Distillation Sampling for a
/// batch of renders, and SDEdit-style refinement of renders (DreamGaussian, Tang et al. 2024, eq. 1 and section 3.2).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <see cref="ComputeImageGradient"/> returns dL/dx for renders x in [0, 1] where, for latents z = E(x) and a fixed
/// noise eps at timestep t, <c>dL/dz = w(t) (eps_hat(z_t; y, t) - eps)</c> with <c>w(t) = 1 - alpha_bar_t</c>
/// (the reference's <c>train_step</c>: <c>loss = 0.5 * mse(z, (z - grad).detach(), reduction='sum')</c>), back-propagated
/// through the encoder. <see cref="Refine"/> encodes the renders, noises them to the step <c>int(steps * strength)</c> of
/// a <paramref name="steps"/>-step schedule, denoises to the end and decodes (the reference's <c>refine</c>).
/// </para>
/// <para><b>For Beginners:</b> this is the "judge" that tells the 3D model how to change so its pictures look more
/// like what the 2D diffusion model expects.</para>
/// </remarks>
public interface IScoreDistillationGuidance<T>
{
    /// <summary>The diffusion schedule's training timesteps.</summary>
    int TrainTimesteps { get; }

    /// <summary>The smallest timestep SDS samples (<c>int(T * 0.02)</c> in the reference).</summary>
    int MinStep { get; }

    /// <summary>The largest timestep SDS samples (<c>int(T * 0.98)</c> in the reference).</summary>
    int MaxStep { get; }

    /// <summary>The image resolution the prior works at (Zero-1-to-3: 256; Stable Diffusion: 512).</summary>
    int Resolution { get; }

    /// <summary>The SDS gradient with respect to the renders.</summary>
    /// <param name="images">Renders [B, 3, H, W] in [0, 1].</param>
    /// <param name="views">One view offset per render.</param>
    /// <param name="timestep">The diffusion timestep.</param>
    /// <param name="random">Random source for the posterior sample and the noise.</param>
    /// <returns>dL/d images, the shape of <paramref name="images"/>.</returns>
    Tensor<T> ComputeImageGradient(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views, int timestep, Random random);

    /// <summary>SDEdit refinement of the renders.</summary>
    /// <param name="images">Renders [B, 3, H, W] in [0, 1].</param>
    /// <param name="views">One view offset per render.</param>
    /// <param name="strength">Fraction of the schedule skipped before noising (0.8 to 0.95 in DreamGaussian).</param>
    /// <param name="steps">Steps of the denoising schedule. DreamGaussian: 50.</param>
    /// <param name="random">Random source.</param>
    /// <returns>Refined images [B, 3, R, R] in [0, 1], R = <see cref="Resolution"/>.</returns>
    Tensor<T> Refine(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views, double strength, int steps, Random random);
}

/// <summary>
/// The latent-space machinery shared by the guidance models: encoding with a posterior sample on a gradient tape,
/// the SDS residual and the SDEdit loop.
/// </summary>
internal static class LatentScoreDistillation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// <c>t = round((1 - ratio) T)</c> clipped to [min, max]: DreamGaussian's linear timestep annealing.
    /// </summary>
    public static int AnnealedTimestep(double ratio, int trainTimesteps, int minStep, int maxStep)
    {
        double t = Math.Round((1.0 - ratio) * trainTimesteps, MidpointRounding.ToEven);
        return (int)Math.Min(maxStep, Math.Max(minStep, t));
    }

    /// <summary>Resizes [B, C, H, W] to <paramref name="size"/> with bilinear interpolation (align_corners=False).</summary>
    public static Tensor<T> Resize(Tensor<T> images, int size)
    {
        if (images.Shape[2] == size && images.Shape[3] == size)
            return images;
        return AiDotNetEngine.Current.Interpolate(images, new[] { size, size }, InterpolateMode.Bilinear, false);
    }

    /// <summary>
    /// The reference's <c>encode_imgs</c>: <c>z = scale * (mu + sigma n)</c> of the posterior of <c>2 x - 1</c>, with
    /// the noise n given so the encoding is reproducible; diffusers clamps the log variance to [-30, 20].
    /// </summary>
    public static Tensor<T> Encode(IVAEModel<T> vae, Tensor<T> imagesInUnitRange, Tensor<T>? posteriorNoise, bool scaled)
    {
        var engine = AiDotNetEngine.Current;
        var signed = engine.TensorAddScalar(engine.TensorMultiplyScalar(imagesInUnitRange, NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var (mean, logVariance) = vae.EncodeWithDistribution(signed);
        var latent = mean;
        if (posteriorNoise is not null)
        {
            var clamped = engine.TensorClamp(logVariance, NumOps.FromDouble(-30.0), NumOps.FromDouble(20.0));
            var std = engine.TensorExp(engine.TensorMultiplyScalar(clamped, NumOps.FromDouble(0.5)));
            latent = engine.TensorAdd(mean, engine.TensorMultiply(std, posteriorNoise));
        }

        return scaled ? engine.TensorMultiplyScalar(latent, NumOps.FromDouble(vae.LatentScaleFactor)) : latent;
    }

    /// <summary>The shape of the latent of an image batch: [B, latent channels, H / f, W / f].</summary>
    public static int[] LatentShape(IVAEModel<T> vae, int batch, int resolution)
    {
        int factor = Math.Max(1, vae.DownsampleFactor);
        return new[] { batch, vae.LatentChannels, resolution / factor, resolution / factor };
    }

    /// <summary>
    /// <c>w(t) (eps_hat - eps)</c> with <c>w(t) = 1 - alpha_bar_t</c> and non-finite entries set to zero
    /// (<c>torch.nan_to_num</c>), for latents z noised with <paramref name="noise"/>.
    /// </summary>
    public static Tensor<T> LatentGradient(
        INoiseScheduler<T> scheduler, Tensor<T> latents, Tensor<T> noise, int timestep, Func<Tensor<T>, int, Tensor<T>> predictNoise)
    {
        var noisy = new Tensor<T>(latents._shape, scheduler.AddNoise(latents.ToVector(), noise.ToVector(), timestep));
        var predicted = predictNoise(noisy, timestep);
        double weight = 1.0 - NumOps.ToDouble(scheduler.GetAlphaCumulativeProduct(timestep));
        var residual = AiDotNetEngine.Current.TensorMultiplyScalar(
            AiDotNetEngine.Current.TensorSubtract(predicted, noise), NumOps.FromDouble(weight));
        var values = residual.Cast<double>().ToArray();
        for (int i = 0; i < values.Length; i++)
        {
            if (double.IsNaN(values[i])) values[i] = 0.0;
            else if (double.IsPositiveInfinity(values[i])) values[i] = double.MaxValue;
            else if (double.IsNegativeInfinity(values[i])) values[i] = double.MinValue;
        }

        return new Tensor<double>(residual._shape, new Vector<double>(values)).Cast<T>();
    }

    /// <summary>
    /// The SDS gradient with respect to <paramref name="images"/> (any resolution; resized to
    /// <paramref name="resolution"/> on the tape, as the reference's <c>F.interpolate</c> is).
    /// </summary>
    public static Tensor<T> ImageGradient(
        IVAEModel<T> vae, INoiseScheduler<T> scheduler, int resolution, Tensor<T> images, int timestep, Random random,
        Func<Tensor<T>, int, Tensor<T>> predictNoise)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (images.Rank != 4 || images.Shape[1] != 3)
            throw new ArgumentException("Renders must be [B, 3, H, W].", nameof(images));
        var engine = AiDotNetEngine.Current;
        var shape = LatentShape(vae, images.Shape[0], resolution);
        var posteriorNoise = DiffusionNoiseHelper<T>.SampleGaussian(shape, random);
        var noise = DiffusionNoiseHelper<T>.SampleGaussian(shape, random);
        var input = images.Clone();

        using var tape = new GradientTape<T>();
        var latents = Encode(vae, Resize(input, resolution), posteriorNoise, scaled: true);
        Tensor<T> gradient;
        using (new NoGradScope<T>())
        {
            gradient = LatentGradient(scheduler, latents.Clone(), noise, timestep, predictNoise);
        }

        // d/dz sum(z * g) = g, the gradient of 0.5 * sum((z - (z - g).detach())^2).
        var loss = engine.ReduceSum(engine.TensorMultiply(latents, gradient), null);
        var gradients = tape.ComputeGradients(loss, new[] { input });
        if (!gradients.TryGetValue(input, out var imageGradient))
            throw new InvalidOperationException("The encoder produced no gradient for the renders.");
        return imageGradient;
    }

    /// <summary>
    /// The reference's <c>refine</c>: encode, noise to <c>timesteps[int(steps * strength)]</c>, run the remaining
    /// scheduler steps (eta 0), decode to <c>(x / 2 + 0.5).clamp(0, 1)</c>.
    /// </summary>
    public static Tensor<T> Refine(
        IVAEModel<T> vae, INoiseScheduler<T> scheduler, int resolution, Tensor<T> images, double strength, int steps,
        Random random, Func<Tensor<T>, int, Tensor<T>> predictNoise)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps), "At least one step is needed.");
        if (strength < 0.0 || strength >= 1.0) throw new ArgumentOutOfRangeException(nameof(strength), "Strength must be in [0, 1).");
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var shape = LatentShape(vae, images.Shape[0], resolution);
        scheduler.SetTimesteps(steps);
        var timesteps = scheduler.Timesteps;
        int initStep = Math.Min(timesteps.Length - 1, (int)(steps * strength));
        Tensor<T> latents;
        if (strength == 0.0)
        {
            latents = DiffusionNoiseHelper<T>.SampleGaussian(shape, random);
        }
        else
        {
            var posteriorNoise = DiffusionNoiseHelper<T>.SampleGaussian(shape, random);
            latents = Encode(vae, Resize(images, resolution), posteriorNoise, scaled: true);
            var noise = DiffusionNoiseHelper<T>.SampleGaussian(shape, random);
            latents = new Tensor<T>(shape, scheduler.AddNoise(latents.ToVector(), noise.ToVector(), timesteps[initStep]));
        }

        for (int i = initStep; i < timesteps.Length; i++)
        {
            int t = timesteps[i];
            var predicted = predictNoise(latents, t);
            latents = new Tensor<T>(shape, scheduler.Step(predicted.ToVector(), t, latents.ToVector(), NumOps.Zero));
        }

        var decoded = vae.Decode(engine.TensorMultiplyScalar(latents, NumOps.FromDouble(1.0 / vae.LatentScaleFactor)));
        var unit = engine.TensorAddScalar(engine.TensorMultiplyScalar(decoded, NumOps.FromDouble(0.5)), NumOps.FromDouble(0.5));
        return engine.TensorClamp(unit, NumOps.Zero, NumOps.One);
    }

    /// <summary>Classifier-free guidance <c>uncond + s (cond - uncond)</c>.</summary>
    public static Tensor<T> Guide(Tensor<T> conditional, Tensor<T> unconditional, double scale)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(unconditional,
            engine.TensorMultiplyScalar(engine.TensorSubtract(conditional, unconditional), NumOps.FromDouble(scale)));
    }
}

/// <summary>
/// Text-conditioned guidance from a Stable Diffusion pipeline: DreamGaussian's text-to-3D prior
/// (<c>guidance/sd_utils.py</c>, Stable Diffusion 2.1 base at 512 x 512, guidance scale 100).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// The prompt and negative prompt are encoded once; the noise prediction is classifier-free guided between them. The
/// denoiser is expected to predict epsilon, as Stable Diffusion 2.1 base does.
/// </remarks>
public sealed class StableDiffusionScoreDistillationGuidance<T> : IScoreDistillationGuidance<T>
{
    private readonly INoisePredictor<T> _noisePredictor;
    private readonly IVAEModel<T> _vae;
    private readonly INoiseScheduler<T> _scheduler;
    private readonly Tensor<T> _promptEmbedding;
    private readonly Tensor<T> _negativeEmbedding;

    /// <summary>Creates the guidance from a denoiser, a VAE, a scheduler and the encoded prompts.</summary>
    /// <param name="noisePredictor">The epsilon-predicting denoiser.</param>
    /// <param name="vae">The first-stage autoencoder.</param>
    /// <param name="scheduler">The noise scheduler (DDIM in the reference).</param>
    /// <param name="promptEmbedding">The encoded prompt.</param>
    /// <param name="negativeEmbedding">The encoded negative (or empty) prompt.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale. DreamGaussian: 100.</param>
    /// <param name="resolution">Render resolution fed to the encoder. Stable Diffusion: 512.</param>
    /// <param name="minTimestepRatio">t_range[0]. Reference: 0.02.</param>
    /// <param name="maxTimestepRatio">t_range[1]. Reference: 0.98.</param>
    public StableDiffusionScoreDistillationGuidance(
        INoisePredictor<T> noisePredictor,
        IVAEModel<T> vae,
        INoiseScheduler<T> scheduler,
        Tensor<T> promptEmbedding,
        Tensor<T> negativeEmbedding,
        double guidanceScale = 100.0,
        int resolution = 512,
        double minTimestepRatio = 0.02,
        double maxTimestepRatio = 0.98)
    {
        _noisePredictor = noisePredictor ?? throw new ArgumentNullException(nameof(noisePredictor));
        _vae = vae ?? throw new ArgumentNullException(nameof(vae));
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _promptEmbedding = promptEmbedding ?? throw new ArgumentNullException(nameof(promptEmbedding));
        _negativeEmbedding = negativeEmbedding ?? throw new ArgumentNullException(nameof(negativeEmbedding));
        if (resolution < 1) throw new ArgumentOutOfRangeException(nameof(resolution), "Resolution must be positive.");
        if (!(minTimestepRatio >= 0.0 && minTimestepRatio <= maxTimestepRatio && maxTimestepRatio <= 1.0))
            throw new ArgumentOutOfRangeException(nameof(minTimestepRatio), "Need 0 <= min <= max <= 1.");
        GuidanceScale = guidanceScale;
        Resolution = resolution;
        TrainTimesteps = scheduler.TrainTimesteps;
        MinStep = (int)(TrainTimesteps * minTimestepRatio);
        MaxStep = (int)(TrainTimesteps * maxTimestepRatio);
    }

    /// <summary>
    /// Creates the guidance from a latent diffusion pipeline, encoding the prompts with its conditioner.
    /// </summary>
    /// <param name="pipeline">A text-to-image latent diffusion model with a conditioner.</param>
    /// <param name="prompt">The prompt.</param>
    /// <param name="negativePrompt">The negative prompt; empty uses the unconditional embedding.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale. DreamGaussian: 100.</param>
    /// <param name="resolution">Render resolution. Stable Diffusion: 512.</param>
    /// <returns>The guidance.</returns>
    public static StableDiffusionScoreDistillationGuidance<T> FromPipeline(
        LatentDiffusionModelBase<T> pipeline, string prompt, string? negativePrompt = null,
        double guidanceScale = 100.0, int resolution = 512)
    {
        if (pipeline is null) throw new ArgumentNullException(nameof(pipeline));
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        var conditioner = pipeline.Conditioner
            ?? throw new InvalidOperationException("Text guidance needs a pipeline with a text conditioner.");
        var promptEmbedding = conditioner.EncodeText(conditioner.Tokenize(prompt));
        var negativeEmbedding = string.IsNullOrEmpty(negativePrompt)
            ? conditioner.GetUnconditionalEmbedding(1)
            : conditioner.EncodeText(conditioner.Tokenize(negativePrompt ?? string.Empty));
        return new StableDiffusionScoreDistillationGuidance<T>(
            pipeline.NoisePredictor, pipeline.VAE, pipeline.Scheduler, promptEmbedding, negativeEmbedding,
            guidanceScale, resolution);
    }

    /// <summary>The classifier-free guidance scale.</summary>
    public double GuidanceScale { get; }

    /// <inheritdoc />
    public int TrainTimesteps { get; }

    /// <inheritdoc />
    public int MinStep { get; }

    /// <inheritdoc />
    public int MaxStep { get; }

    /// <inheritdoc />
    public int Resolution { get; }

    /// <inheritdoc />
    /// <remarks>The reference's Stable Diffusion loss is divided by the batch size (<c>/ latents.shape[0]</c>), unlike
    /// its Zero-1-to-3 loss, so the gradient is too.</remarks>
    public Tensor<T> ComputeImageGradient(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views, int timestep, Random random)
    {
        var gradient = LatentScoreDistillation<T>.ImageGradient(_vae, _scheduler, Resolution, images, timestep, random, PredictGuided);
        return images.Shape[0] == 1
            ? gradient
            : AiDotNetEngine.Current.TensorMultiplyScalar(gradient, MathHelper.GetNumericOperations<T>().FromDouble(1.0 / images.Shape[0]));
    }

    /// <inheritdoc />
    public Tensor<T> Refine(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views, double strength, int steps, Random random)
        => LatentScoreDistillation<T>.Refine(_vae, _scheduler, Resolution, images, strength, steps, random, PredictGuided);

    /// <summary>The SDS latent residual <c>w(t) (eps_hat - eps)</c> for given latents and noise.</summary>
    internal Tensor<T> LatentGradient(Tensor<T> latents, Tensor<T> noise, int timestep)
        => LatentScoreDistillation<T>.LatentGradient(_scheduler, latents, noise, timestep, PredictGuided);

    /// <summary>The guided noise prediction for a latent batch.</summary>
    internal Tensor<T> PredictGuided(Tensor<T> noisyLatents, int timestep)
    {
        var conditional = _noisePredictor.PredictNoise(noisyLatents, timestep, _promptEmbedding);
        var unconditional = _noisePredictor.PredictNoise(noisyLatents, timestep, _negativeEmbedding);
        return LatentScoreDistillation<T>.Guide(conditional, unconditional, GuidanceScale);
    }
}

/// <summary>
/// Image- and camera-conditioned guidance from Zero-1-to-3: DreamGaussian's image-to-3D prior
/// (<c>guidance/zero123_utils.py</c>, 256 x 256, guidance scale 5).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// The reference image is encoded once into the CLIP image embedding and the VAE latent that Zero-1-to-3 concatenates
/// to its input; each render is guided with its camera offset from the reference view. The unconditional branch of
/// classifier-free guidance zeroes both the embedding and the concatenated latent, as the reference does.
/// </remarks>
public sealed class Zero123ScoreDistillationGuidance<T> : IScoreDistillationGuidance<T>
{
    private readonly Zero123Model<T> _model;
    private readonly Zero123ImageCondition<T> _condition;

    /// <summary>Creates the guidance for a reference image.</summary>
    /// <param name="model">The Zero-1-to-3 model.</param>
    /// <param name="referenceImage">The reference image [1, 3, H, W] in [0, 1] (on a white background).</param>
    /// <param name="random">Random source for the reference latent's posterior sample.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale. DreamGaussian: 5.</param>
    /// <param name="resolution">Render resolution. Zero-1-to-3: 256.</param>
    /// <param name="minTimestepRatio">t_range[0]. Reference: 0.02.</param>
    /// <param name="maxTimestepRatio">t_range[1]. Reference: 0.98.</param>
    public Zero123ScoreDistillationGuidance(
        Zero123Model<T> model,
        Tensor<T> referenceImage,
        Random random,
        double guidanceScale = 5.0,
        int resolution = 256,
        double minTimestepRatio = 0.02,
        double maxTimestepRatio = 0.98)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        if (referenceImage is null) throw new ArgumentNullException(nameof(referenceImage));
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (resolution < 1) throw new ArgumentOutOfRangeException(nameof(resolution), "Resolution must be positive.");
        if (!(minTimestepRatio >= 0.0 && minTimestepRatio <= maxTimestepRatio && maxTimestepRatio <= 1.0))
            throw new ArgumentOutOfRangeException(nameof(minTimestepRatio), "Need 0 <= min <= max <= 1.");
        GuidanceScale = guidanceScale;
        Resolution = resolution;
        TrainTimesteps = model.Scheduler.TrainTimesteps;
        MinStep = (int)(TrainTimesteps * minTimestepRatio);
        MaxStep = (int)(TrainTimesteps * maxTimestepRatio);
        _condition = model.EncodeImageCondition(referenceImage, resolution, random);
    }

    /// <summary>The classifier-free guidance scale.</summary>
    public double GuidanceScale { get; }

    /// <inheritdoc />
    public int TrainTimesteps { get; }

    /// <inheritdoc />
    public int MinStep { get; }

    /// <inheritdoc />
    public int MaxStep { get; }

    /// <inheritdoc />
    public int Resolution { get; }

    /// <inheritdoc />
    public Tensor<T> ComputeImageGradient(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views, int timestep, Random random)
    {
        RequireViews(images, views);
        return LatentScoreDistillation<T>.ImageGradient(_model.VAE, _model.Scheduler, Resolution, images, timestep, random,
            (latents, t) => _model.PredictViewConditionedNoise(latents, t, _condition, views, GuidanceScale));
    }

    /// <inheritdoc />
    public Tensor<T> Refine(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views, double strength, int steps, Random random)
    {
        RequireViews(images, views);
        return LatentScoreDistillation<T>.Refine(_model.VAE, _model.Scheduler, Resolution, images, strength, steps, random,
            (latents, t) => _model.PredictViewConditionedNoise(latents, t, _condition, views, GuidanceScale));
    }

    private static void RequireViews(Tensor<T> images, IReadOnlyList<GuidanceViewOffset> views)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (views is null) throw new ArgumentNullException(nameof(views));
        if (images.Rank != 4 || views.Count != images.Shape[0])
            throw new ArgumentException("Zero-1-to-3 guidance needs one view offset per render.", nameof(views));
    }
}
