namespace AiDotNet.Enums;

/// <summary>
/// Which end-to-end decoder a medical-conversation ASR model uses.
/// </summary>
/// <remarks>
/// <para>
/// Chiu et al., "Speech recognition for medical conversations" (Interspeech 2018,
/// arXiv:1711.07274) is a COMPARISON: "We explored both CTC and LAS systems for building speech
/// recognition models." Offering both is therefore what implementing that paper means; offering
/// only one reproduces half of it.
/// </para>
/// <para>
/// Their finding decides the default: "The LAS was more resilient to noisy data and CTC required
/// more data clean up." Doctor-patient conversation is exactly the noisy, spontaneous condition
/// where that mattered.
/// </para>
/// </remarks>
public enum MedicalAsrDecoderType
{
    /// <summary>
    /// Listen, Attend and Spell: a pyramidal encoder ("listen") feeding an attention decoder
    /// ("attend and spell"). The paper's more robust option and the default here.
    /// </summary>
    ListenAttendSpell = 0,

    /// <summary>
    /// Connectionist Temporal Classification: a frame-synchronous head over the encoder, with no
    /// decoder state. Cheaper and streamable, but the paper found it needed cleaner data.
    /// </summary>
    Ctc = 1,
}
