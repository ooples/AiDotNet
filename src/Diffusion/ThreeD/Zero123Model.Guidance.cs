using AiDotNet.Helpers;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// A reference image encoded for Zero-1-to-3 conditioning: the image embedding and the first-stage latent that is
/// concatenated to the denoiser's input.
/// </summary>
internal sealed class Zero123ImageCondition<T>
{
    public Zero123ImageCondition(Tensor<T> embedding, Tensor<T> latent)
    {
        Embedding = embedding;
        Latent = latent;
    }

    /// <summary>The image embedding (the reference's <c>embeddings[0]</c>).</summary>
    public Tensor<T> Embedding { get; }

    /// <summary>The unscaled posterior sample of the image (the reference's <c>embeddings[1]</c>), [1, 4, h, w].</summary>
    public Tensor<T> Latent { get; }
}

public partial class Zero123Model<T>
{
    /// <summary>
    /// Encodes a reference image for camera-conditioned guidance, as the reference's <c>get_img_embeds</c> does: the
    /// image embedding of the encoder's input resolution, and <c>encode_imgs(x) / scaling_factor</c>, the unscaled
    /// posterior sample of the image at <paramref name="resolution"/>.
    /// </summary>
    /// <param name="image">The reference image [1, 3, H, W] in [0, 1].</param>
    /// <param name="resolution">The guidance resolution (256 for Zero-1-to-3).</param>
    /// <param name="random">Random source for the posterior sample.</param>
    internal Zero123ImageCondition<T> EncodeImageCondition(Tensor<T> image, int resolution, Random random)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (image.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException("The reference image must be [1, 3, H, W].", nameof(image));

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var embedding = _imageEncoder.Encode(LatentScoreDistillation<T>.Resize(image, ImageEncoderResolution));
        var shape = LatentScoreDistillation<T>.LatentShape(_vae, 1, resolution);
        var posteriorNoise = DiffusionNoiseHelper<T>.SampleGaussian(shape, random);
        var latent = LatentScoreDistillation<T>.Encode(
            _vae, LatentScoreDistillation<T>.Resize(image, resolution), posteriorNoise, scaled: false);
        return new Zero123ImageCondition<T>(embedding, latent);
    }

    /// <summary>
    /// The classifier-free-guided noise prediction for a batch of noisy latents, each seen from its camera offset to
    /// the reference view (the reference's <c>train_step</c> / <c>refine</c> denoiser call): the conditional branch
    /// combines the image embedding with the camera pose and concatenates the reference latent; the unconditional
    /// branch zeroes both.
    /// </summary>
    internal Tensor<T> PredictViewConditionedNoise(
        Tensor<T> noisyLatents, int timestep, Zero123ImageCondition<T> condition, IReadOnlyList<GuidanceViewOffset> views,
        double guidanceScale)
    {
        if (noisyLatents is null) throw new ArgumentNullException(nameof(noisyLatents));
        if (condition is null) throw new ArgumentNullException(nameof(condition));
        if (views is null) throw new ArgumentNullException(nameof(views));
        int batch = noisyLatents.Shape[0];
        if (views.Count != batch)
            throw new ArgumentException("One view offset per latent is needed.", nameof(views));

        var zeroLatent = new Tensor<T>(condition.Latent._shape);
        var outputs = new Tensor<T>[batch];
        for (int b = 0; b < batch; b++)
        {
            var latent = Engine.TensorNarrow(noisyLatents, 0, b, 1);
            var view = views[b];
            // The pose encoder takes the relative radius as a ratio (1 = same distance).
            var pose = _poseEncoder.Encode(view.Elevation, view.Azimuth, 1.0 + view.Radius);
            var conditioning = CombineEmbeddings(condition.Embedding, pose);
            var unconditioning = new Tensor<T>(conditioning._shape);
            var conditional = _unet.PredictNoise(ConcatenateLatents(latent, condition.Latent), timestep, conditioning);
            var unconditional = _unet.PredictNoise(ConcatenateLatents(latent, zeroLatent), timestep, unconditioning);
            outputs[b] = LatentScoreDistillation<T>.Guide(conditional, unconditional, guidanceScale);
        }

        return batch == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
    }

    /// <summary>The image encoder's input resolution (CLIP's 224).</summary>
    private const int ImageEncoderResolution = 224;
}
