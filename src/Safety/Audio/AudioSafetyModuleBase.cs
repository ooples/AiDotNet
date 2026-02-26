using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Abstract base class for audio safety modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for all audio safety modules. Concrete modules implement
/// <see cref="EvaluateAudio(Vector{T}, int)"/> and this base class handles the
/// <see cref="ISafetyModule{T}.Evaluate(Vector{T})"/> bridge using the configured sample rate.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class handles the plumbing so that each audio safety
/// module only needs to implement one method: <c>EvaluateAudio(Vector&lt;T&gt;, int)</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class AudioSafetyModuleBase<T> : IAudioSafetyModule<T>
{
    private readonly int _defaultSampleRate;

    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <summary>
    /// Initializes a new audio safety module base with the specified default sample rate.
    /// </summary>
    /// <param name="defaultSampleRate">The default sample rate in Hz when not explicitly provided.</param>
    protected AudioSafetyModuleBase(int defaultSampleRate = 16000)
    {
        if (defaultSampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(defaultSampleRate), defaultSampleRate,
                "Sample rate must be a positive integer.");
        }

        _defaultSampleRate = defaultSampleRate;
    }

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> EvaluateAudio(Vector<T> audioSamples, int sampleRate);

    /// <inheritdoc />
    /// <remarks>
    /// The base implementation delegates to <see cref="EvaluateAudio(Vector{T}, int)"/>
    /// using the configured default sample rate.
    /// </remarks>
    public virtual IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        return EvaluateAudio(content, _defaultSampleRate);
    }
}
