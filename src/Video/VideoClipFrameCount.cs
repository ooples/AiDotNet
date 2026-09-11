using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// The one rule for reconciling a clip model's <c>numFrames</c> constructor argument with the frame count
/// its architecture declares.
/// </summary>
/// <remarks>
/// <para>
/// The architecture is the model's declared input contract: for a temporal-video architecture
/// <c>GetInputShape()</c> is <c>[InputFrames, C, H, W]</c>, and the auto-batching rank check, generated
/// fixtures and serving all size their inputs from it. So a frame count the architecture declares
/// (<see cref="NeuralNetworkArchitecture{T}.InputFrames"/> &gt; 0) wins - the rule BSVD already applies. An
/// explicit <c>numFrames</c> that disagrees with it is rejected rather than silently overridden, since the
/// two would describe different clips.
/// </para>
/// <para>
/// Limitation: a parameter's default value cannot be told apart from the same value passed explicitly, so
/// passing exactly the default defers to the architecture instead of throwing.
/// </para>
/// </remarks>
internal static class VideoClipFrameCount
{
    /// <summary>Resolves the clip length a model should be built for.</summary>
    /// <param name="architecture">The model's architecture.</param>
    /// <param name="numFrames">The <c>numFrames</c> constructor argument.</param>
    /// <param name="defaultNumFrames">That argument's default value.</param>
    /// <returns>The architecture's declared frame count when it declares one; otherwise <paramref name="numFrames"/>.</returns>
    /// <exception cref="ArgumentException">An explicit, non-default <paramref name="numFrames"/> conflicts
    /// with the architecture's declared frame count.</exception>
    internal static int Resolve<T>(NeuralNetworkArchitecture<T> architecture, int numFrames, int defaultNumFrames)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        int declared = architecture.InputFrames;
        if (declared <= 0)
        {
            return numFrames;
        }

        if (numFrames != declared && numFrames != defaultNumFrames)
        {
            throw new ArgumentException(
                $"numFrames ({numFrames}) conflicts with the architecture's declared InputFrames ({declared}). " +
                "Pass the same value, or omit numFrames to use the architecture's frame count.",
                nameof(numFrames));
        }

        return declared;
    }
}
