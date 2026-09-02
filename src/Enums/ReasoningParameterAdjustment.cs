namespace AiDotNet.Enums;

/// <summary>What a reasoning-model profile did to one request setting on its way to the provider.</summary>
/// <remarks>
/// <para>
/// A request aimed at a reasoning model is not sent exactly as written: some settings are removed because the model
/// rejects them, one is renamed because the model spells it differently, and one may be added because the caller
/// asked for a deliberation level. Each of those edits is reported as a diagnostic carrying one of these values, so
/// a run can prove afterwards what was actually sent rather than assuming the configured options survived.
/// </para>
/// <para><b>For Beginners:</b> Before your request is sent, the library sometimes has to adjust it so the model
/// will accept it. This says which kind of adjustment happened: a setting was removed, renamed, or added. It exists
/// so those changes are visible to you instead of happening behind your back.</para>
/// </remarks>
public enum ReasoningParameterAdjustment
{
    /// <summary>The setting was removed because the model rejects it.</summary>
    Dropped = 0,

    /// <summary>The setting was sent under a different wire name the model does accept.</summary>
    Substituted = 1,

    /// <summary>The setting was added to the request because the configuration asked for it.</summary>
    Added = 2
}
