namespace AiDotNet.Interfaces;

/// <summary>
/// Optional sampler capability for schedules whose latent coordinate system differs from the
/// denoiser's model-input coordinate system, such as sigma-space Euler sampling.
/// </summary>
/// <typeparam name="T">The numeric element type.</typeparam>
public interface INoiseSchedulerInputScaling<T>
{
    /// <summary>Scale applied to unit Gaussian initial noise after inference timesteps are configured.</summary>
    T InitialNoiseSigma { get; }

    /// <summary>
    /// Converts only the evolving noisy latent to the denoiser's input coordinates without mutating
    /// the sample. Separate source-image or other conditioning channels must be concatenated afterwards.
    /// </summary>
    Tensor<T> ScaleModelInput(Tensor<T> sample, int timestep);
}
