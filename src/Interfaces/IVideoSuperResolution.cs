using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// The video super-resolution contract: upscale a low-resolution frame sequence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> video super-resolution takes a low-resolution clip and produces a
/// higher-resolution one, using neighbouring frames for detail a single image cannot supply.
/// </para>
/// <para>
/// This interface exists because the capability is NOT tied to one base class. Most models in the
/// library inherit <c>VideoSuperResolutionBase</c>, but diffusion-based ones (Stream-DiffVSR,
/// StableVideoSR, MGLD-VSR, DOVE, SeedVR, FlashVSR) genuinely need the latent/denoising machinery on
/// <c>VideoDiffusionModelBase</c> instead — and C# permits one base class. Declaring the capability as
/// an interface lets both hierarchies be recognised as video super-resolution by consumers and by the
/// test-scaffold generator, rather than the generator matching on a base-class NAME prefix, which
/// silently excludes any model whose base is named differently.
/// </para>
/// </remarks>
public interface IVideoSuperResolution<T>
{
    /// <summary>
    /// Gets the spatial upscaling factor (e.g. 4 for x4 super-resolution).
    /// </summary>
    int ScaleFactor { get; }

    /// <summary>
    /// Upscales a low-resolution frame sequence.
    /// </summary>
    /// <param name="lowResFrames">The low-resolution frames.</param>
    /// <returns>The upscaled frames, with spatial dimensions multiplied by <see cref="ScaleFactor"/>.</returns>
    Tensor<T> Upscale(Tensor<T> lowResFrames);

    /// <summary>
    /// Estimates the optical-flow field between two frames.
    /// </summary>
    /// <param name="frame1">First frame, [channels, height, width].</param>
    /// <param name="frame2">Second frame, [channels, height, width].</param>
    /// <returns>Flow field [2, height, width] holding (dx, dy) displacements.</returns>
    /// <remarks>
    /// Part of the contract because temporal alignment is what separates video super-resolution from
    /// per-frame image super-resolution: warping neighbouring frames into the reference frame is how
    /// these models recover detail. A model that returns zero flow is doing per-frame upscaling
    /// regardless of what its architecture claims.
    /// </remarks>
    Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2);
}
