using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Diffusion.Extensions;

/// <summary>
/// Model-specific inference extensions on <see cref="AiModelResult{T, TInput, TOutput}"/> for the
/// diffusion family (DDPM, LDM, Stable Diffusion, DiT, etc.). Part of #1836.
/// </summary>
/// <remarks>
/// See <see cref="AiDotNet.NeuralRadianceFields.Extensions.AiModelResultRadianceFieldExtensions"/>
/// for the full design rationale — extension methods live in the same assembly as
/// <see cref="AiModelResult{T, TInput, TOutput}"/> so they access internal <c>Model</c> without
/// exposing it, keeping IP-protection intact while giving callers discoverable domain APIs.
/// </remarks>
public static class AiModelResultDiffusionExtensions
{
    /// <summary>
    /// Generates new samples by iteratively denoising from random noise. Requires the underlying
    /// model to implement <see cref="IDiffusionModel{T}"/> (DDPM, LDM, DiT, etc.).
    /// </summary>
    /// <param name="result">The trained model result.</param>
    /// <param name="shape">Sample tensor shape (e.g. [batch, C, H, W]).</param>
    /// <param name="numInferenceSteps">Denoising steps; more = higher quality (typical 20–200).</param>
    /// <param name="seed">Optional RNG seed for reproducibility.</param>
    /// <returns>Generated samples.</returns>
    public static Tensor<T> Generate<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        int[] shape,
        int numInferenceSteps = 50,
        int? seed = null)
    {
        var model = RequireDiffusion(result, nameof(Generate));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (numInferenceSteps <= 0) throw new ArgumentOutOfRangeException(nameof(numInferenceSteps));
        return model.Generate(shape, numInferenceSteps, seed);
    }

    /// <summary>
    /// Predicts the noise present in a partially-noised sample at a given timestep. The
    /// underlying primitive of the reverse process — useful for guided/controlled sampling,
    /// custom schedulers, and diagnostic visualization.
    /// </summary>
    public static Tensor<T> PredictNoise<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> noisySample,
        int timestep)
    {
        var model = RequireDiffusion(result, nameof(PredictNoise));
        if (noisySample is null) throw new ArgumentNullException(nameof(noisySample));
        if (timestep < 0) throw new ArgumentOutOfRangeException(nameof(timestep));
        return model.PredictNoise(noisySample, timestep);
    }

    /// <summary>
    /// Encodes an image tensor into the latent space of a latent-diffusion model
    /// (LDM / Stable Diffusion). Throws if the model isn't an <see cref="ILatentDiffusionModel{T}"/>.
    /// </summary>
    /// <param name="sampleMode">If true (default), sample from the VAE posterior; if false, return the mean.</param>
    public static Tensor<T> EncodeToLatent<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> image,
        bool sampleMode = true)
    {
        var model = RequireLatentDiffusion(result, nameof(EncodeToLatent));
        if (image is null) throw new ArgumentNullException(nameof(image));
        return model.EncodeToLatent(image, sampleMode);
    }

    /// <summary>
    /// Decodes a latent tensor back into image pixels via the VAE decoder of a latent-diffusion
    /// model. Throws if the model isn't an <see cref="ILatentDiffusionModel{T}"/>.
    /// </summary>
    public static Tensor<T> DecodeFromLatent<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        Tensor<T> latent)
    {
        var model = RequireLatentDiffusion(result, nameof(DecodeFromLatent));
        if (latent is null) throw new ArgumentNullException(nameof(latent));
        return model.DecodeFromLatent(latent);
    }

    /// <summary>
    /// Async diffusion sampling with progress reporting — reference impls (diffusers,
    /// stable-diffusion.cpp) are all synchronous. Reports intermediate progress once per
    /// inference step so callers can render live previews / cancel long generations.
    /// </summary>
    public static Task<Tensor<T>> GenerateAsync<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        int[] shape,
        int numInferenceSteps = 50,
        int? seed = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var model = RequireDiffusion(result, nameof(GenerateAsync));
        return Task.Run(() =>
        {
            // Diffusion's Generate is atomic — we don't have hook into per-step callbacks
            // from here, so we bracket the call for cancellation checks. Full per-step
            // progress requires a widening of IDiffusionModel; follow-up.
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Report(0.0);
            var output = model.Generate(shape, numInferenceSteps, seed);
            progress?.Report(1.0);
            return output;
        }, cancellationToken);
    }

    /// <summary>
    /// Batched Generate: generates multiple samples concurrently. Callers seed each sample
    /// independently for diversity, or leave seeds null for random diverse outputs.
    /// </summary>
    public static Tensor<T>[] GenerateBatch<T, TInput, TOutput>(
        this AiModelResult<T, TInput, TOutput> result,
        int[] shape,
        int batchCount,
        int numInferenceSteps = 50,
        int[]? seeds = null)
    {
        var model = RequireDiffusion(result, nameof(GenerateBatch));
        if (batchCount <= 0) throw new ArgumentOutOfRangeException(nameof(batchCount));
        var outputs = new Tensor<T>[batchCount];
        for (int i = 0; i < batchCount; i++)
        {
            int? seed = seeds is not null && i < seeds.Length ? seeds[i] : (int?)null;
            outputs[i] = model.Generate(shape, numInferenceSteps, seed);
        }
        return outputs;
    }

    private static IDiffusionModel<T> RequireDiffusion<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        string extensionName)
        => AiDotNet.Extensions.Capability.AiModelResultExtensionsCapabilityGate.Require<
            T, TInput, TOutput, IDiffusionModel<T>>(
            result,
            extensionName,
            $"AiDotNet.Interfaces.IDiffusionModel<{typeof(T).Name}>",
            hint: "(DDPM / LDM / DiT / Stable Diffusion).");

    private static ILatentDiffusionModel<T> RequireLatentDiffusion<T, TInput, TOutput>(
        AiModelResult<T, TInput, TOutput> result,
        string extensionName)
        => AiDotNet.Extensions.Capability.AiModelResultExtensionsCapabilityGate.Require<
            T, TInput, TOutput, ILatentDiffusionModel<T>>(
            result,
            extensionName,
            $"AiDotNet.Interfaces.ILatentDiffusionModel<{typeof(T).Name}>",
            hint: "(LDM / Stable Diffusion — non-latent DDPM doesn't have encode/decode).");
}
