using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Data;

namespace AiDotNet.NeuralRadianceFields.Interfaces;

/// <summary>
/// Implemented by radiance-field models that support image-space (photometric) training
/// (NeRF, InstantNGP, GaussianSplatting). The facade calls
/// <see cref="TrainOnImageBatch"/> per iteration when the caller supplied an image-shaped
/// data loader — the model owns ray sampling, volume rendering, loss computation, and
/// gradient application internally. Added in #1834.
/// </summary>
/// <remarks>
/// <para>
/// Reference NeRF training pipelines require the caller to hand-roll: read images off disk →
/// sample rays → project to world → compute volume-rendered pixel → MSE against photo →
/// autograd → optimizer.step. The <see cref="IImageTrainable{T}"/> hook keeps every step of
/// that pipeline INSIDE the model, so consumers get one call:
/// <c>ConfigureModel(new NeRF&lt;float&gt;())</c> +
/// <c>ConfigureDataLoader(ImageTrainingDataLoaders.FromViews(views))</c> + <c>BuildAsync()</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
public interface IImageTrainable<T>
{
    /// <summary>
    /// Runs one image-space training iteration over the given views. The model samples rays,
    /// volume-renders each sampled pixel, computes the photometric loss against the sampled
    /// ground-truth colors, and applies the resulting gradient using its internal update rule
    /// (per-attribute Adam for GS, standard Adam for NeRF, hash-grid Adam for InstantNGP).
    /// </summary>
    /// <param name="loader">Image-shaped loader emitting (<see cref="ImageView{T}"/>, <see cref="PixelBatch{T}"/>) pairs.</param>
    /// <param name="raysPerBatch">Number of rays to sample per iteration (paper standard: 1024–4096).</param>
    /// <param name="optimizerOptions">Optimizer hyperparameters routed from <c>ConfigureOptimizer</c>.</param>
    /// <returns>The batch photometric loss (MSE against ground-truth pixels) for telemetry.</returns>
    T TrainOnImageBatch(
        IDataLoader<ImageView<T>, PixelBatch<T>> loader,
        int raysPerBatch,
        OptimizationAlgorithmOptions<T, LinearAlgebra.Tensor<T>, LinearAlgebra.Tensor<T>>? optimizerOptions,
        ImageTrainingOptions? imageTrainingOptions = null);
}
