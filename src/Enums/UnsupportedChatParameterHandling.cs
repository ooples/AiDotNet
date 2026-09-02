namespace AiDotNet.Enums;

/// <summary>What to do when a request carries a setting the target model does not support.</summary>
/// <remarks>
/// <para>
/// <see cref="Drop"/> mirrors the reference OpenEvolve behaviour: the offending field is simply left out of the
/// request body so the call succeeds. It differs in one respect that matters — the removal is always reported as a
/// diagnostic, so a run never loses a setting without saying so, whereas upstream builds a different parameter
/// dictionary with no record that anything was omitted.
/// </para>
/// <para>
/// <see cref="Throw"/> is the strict choice for a pipeline that must not quietly change meaning. A search whose
/// exploration depends on a configured temperature is not the same search once the temperature is gone, and some
/// callers would rather fail at the first request than compare two runs that were never comparable.
/// </para>
/// <para><b>For Beginners:</b> Suppose you asked for a "creativity" setting but the model you picked does not
/// accept one. <see cref="Drop"/> sends the request anyway without that setting and tells you it did;
/// <see cref="Throw"/> stops with an error so you can pick a different model or remove the setting yourself. Start
/// with <see cref="Drop"/>; switch to <see cref="Throw"/> when the setting is important enough that silently losing
/// it would invalidate your results.</para>
/// </remarks>
public enum UnsupportedChatParameterHandling
{
    /// <summary>Remove the setting from the request and report the removal as a diagnostic.</summary>
    Drop = 0,

    /// <summary>Throw an <see cref="InvalidOperationException"/> instead of changing the request.</summary>
    Throw = 1
}
