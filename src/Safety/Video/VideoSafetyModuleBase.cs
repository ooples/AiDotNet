using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Video;

/// <summary>
/// Abstract base class for video safety modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for all video safety modules. Concrete modules implement
/// <see cref="EvaluateVideo(IReadOnlyList{Tensor{T}}, double)"/> and this base class handles
/// the <see cref="ISafetyModule{T}.Evaluate(Vector{T})"/> bridge.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class handles the plumbing so that each video safety
/// module only needs to implement one method: <c>EvaluateVideo(IReadOnlyList&lt;Tensor&lt;T&gt;&gt;, double)</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class VideoSafetyModuleBase<T> : SafetyModuleBase<T>, IVideoSafetyModule<T>
{
    // ModuleName, IsReady, Engine, NumOps inherited from SafetyModuleBase

    private readonly double _defaultFrameRate;

    /// <summary>
    /// Initializes a new video safety module base with the specified default frame rate.
    /// </summary>
    /// <param name="defaultFrameRate">Default frame rate when not explicitly provided.</param>
    protected VideoSafetyModuleBase(double defaultFrameRate = 30.0)
    {
        _defaultFrameRate = defaultFrameRate;
    }

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate);

    /// <summary>
    /// Gets the fewest frames this module needs to judge anything.
    /// </summary>
    /// <remarks>
    /// A module that judges motion - how one frame changes into the next - needs at least two. A single
    /// content vector carries exactly one frame, so such a module cannot be evaluated through
    /// <see cref="Evaluate(Vector{T})"/>; it has to be given the clip through
    /// <see cref="EvaluateVideo(IReadOnlyList{Tensor{T}}, double)"/>.
    /// </remarks>
    public virtual int MinimumFrames => 1;

    /// <inheritdoc />
    /// <remarks>
    /// The base implementation wraps the vector in a single 1D tensor frame and delegates to
    /// <see cref="EvaluateVideo(IReadOnlyList{Tensor{T}}, double)"/> using the default frame rate.
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// The module needs more than one frame (<see cref="MinimumFrames"/>), which a single vector cannot carry.
    /// It used to return no findings, which reads as "safe" rather than "not evaluated".
    /// </exception>
    public override IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null)
        {
            throw new ArgumentNullException(nameof(content));
        }

        if (MinimumFrames > 1)
        {
            throw new NotSupportedException(
                $"{ModuleName} judges change across at least {MinimumFrames} frames, and a single content "
                + "vector is one frame. Call EvaluateVideo with the clip's frames instead.");
        }

        var tensor = new Tensor<T>(content.ToArray(), new[] { content.Length });
        var frames = new List<Tensor<T>> { tensor };
        return EvaluateVideo(frames, _defaultFrameRate);
    }
}
