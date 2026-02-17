using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Abstract base class for voice protection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for voice protectors including sample rate
/// configuration and signal-to-noise ratio monitoring. Concrete implementations
/// provide the actual protection technique (perturbation, watermark, masking).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all voice protectors.
/// Each protector type extends this and adds its own way of making voice recordings
/// resistant to AI voice cloning.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class VoiceProtectorBase<T> : IVoiceProtector<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The default sample rate for audio processing.
    /// </summary>
    protected readonly int DefaultSampleRate;

    /// <summary>
    /// Initializes the voice protector base.
    /// </summary>
    /// <param name="defaultSampleRate">Default sample rate in Hz. Default: 16000.</param>
    protected VoiceProtectorBase(int defaultSampleRate = 16000)
    {
        if (defaultSampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(defaultSampleRate), "Sample rate must be positive.");

        DefaultSampleRate = defaultSampleRate;
    }

    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <inheritdoc />
    public abstract Vector<T> ProtectVoice(Vector<T> audioSamples, int sampleRate);

    /// <inheritdoc />
    public virtual IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null) throw new ArgumentNullException(nameof(content));

        // Voice protectors don't produce findings — they modify audio.
        return Array.Empty<SafetyFinding>();
    }
}
