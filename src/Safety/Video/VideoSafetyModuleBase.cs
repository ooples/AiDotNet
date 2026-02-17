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
public abstract class VideoSafetyModuleBase<T> : IVideoSafetyModule<T>
{
    private readonly double _defaultFrameRate;

    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

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

    /// <inheritdoc />
    /// <remarks>
    /// The base implementation wraps the vector in a single 1D tensor frame and delegates to
    /// <see cref="EvaluateVideo(IReadOnlyList{Tensor{T}}, double)"/> using the default frame rate.
    /// </remarks>
    public virtual IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        var tensor = new Tensor<T>(content.ToArray(), new[] { content.Length });
        var frames = new List<Tensor<T>> { tensor };
        return EvaluateVideo(frames, _defaultFrameRate);
    }
}
