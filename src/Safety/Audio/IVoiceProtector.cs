using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Audio;

/// <summary>
/// Interface for voice protection modules that defend against voice cloning and deepfake attacks.
/// </summary>
/// <remarks>
/// <para>
/// Voice protectors apply active defense techniques to audio to prevent unauthorized
/// voice cloning. Approaches include adding imperceptible perturbations (SPEC),
/// embedding watermarks (AudioSeal), and psychoacoustic masking (VocalCrypt).
/// </para>
/// <para>
/// <b>For Beginners:</b> A voice protector adds invisible protection to voice recordings
/// so that AI voice cloning tools cannot accurately copy the voice. The protection is
/// designed to be inaudible to humans but disruptive to cloning algorithms.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeSpeech: SPEC perturbation framework against voice cloning (2025, arxiv:2504.09839)
/// - VocalCrypt: Pseudo-timbre jamming for voice protection (2025, arxiv:2502.10329)
/// - AudioSeal: Localized watermarking (Meta AI, 2024, arxiv:2401.17264)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IVoiceProtector<T> : ISafetyModule<T>
{
    /// <summary>
    /// Applies voice protection to the given audio samples.
    /// </summary>
    /// <param name="audioSamples">The original audio samples to protect.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>The protected audio samples with anti-cloning modifications applied.</returns>
    Vector<T> ProtectVoice(Vector<T> audioSamples, int sampleRate);
}
