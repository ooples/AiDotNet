using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Training;

/// <summary>
/// Tape options for the training steps this library runs itself, where the step owns every forward tensor it makes.
/// </summary>
internal static class TrainingTapeOptions
{
    /// <summary>
    /// The default tape options (persistent, so the auto-training compiler can replay the backward) plus freeing each
    /// device-only forward intermediate at its last backward use. A user's own <c>GradientTape</c> leaves that off and
    /// never invalidates a tensor its caller holds; a library step opts in, as its forward intermediates are dead once
    /// backward has read them. That frees 30-40% of a GPU step's dead device memory in stream order, without the
    /// blocking collections waiting for the GC would cost. A step-owned intermediate read after backward is kept by the
    /// probe on the graph's first run (see <see cref="GradientTapeOptions.ReleaseDeviceActivationsAtLastUse"/>).
    /// </summary>
    internal static readonly GradientTapeOptions Step = new()
    {
        Persistent = true,
        ReleaseDeviceActivationsAtLastUse = true,
    };
}
