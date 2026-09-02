using AiDotNet.Agentic.Models;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>Layers chat settings so a specific setting wins over a general one, field by field.</summary>
/// <remarks>
/// <para>
/// Chat settings arrive from several places at once: the per-call options a caller passes, the defaults attached
/// to one ensemble member, the defaults attached to the whole client, and finally whatever the connector does
/// when a value is left unset. Because every property of <see cref="ChatOptions"/> is nullable, "unset" and
/// "explicitly set" are distinguishable, and layering is a simple field-wise choice: take the value from the most
/// specific layer that supplied one.
/// </para>
/// <para>
/// Merging happens at call time, which matters. The reference OpenEvolve implementation copies shared settings
/// into each model's configuration once, during construction, from a hard-coded list of field names — so a model
/// added to the ensemble afterwards silently keeps null settings, and a field added to the configuration type
/// but not to that list is never propagated at all. Layering at the point of use has neither failure mode.
/// </para>
/// <para><b>For Beginners:</b> Suppose you set a default temperature for all your models, a different one for one
/// particular model, and a third for a single call. Which wins? The most specific one that you actually set: the
/// call beats the model, the model beats the global default, and anything you left blank falls through to the
/// next level down. This helper does that combining for you.</para>
/// </remarks>
public static class ChatOptionsMerge
{
    /// <summary>Layers two sets of settings, preferring the more specific one field by field.</summary>
    /// <param name="primary">The more specific settings, or <c>null</c>.</param>
    /// <param name="fallback">The less specific settings, or <c>null</c>.</param>
    /// <returns>
    /// A new instance carrying each field from <paramref name="primary"/> when it is set and from
    /// <paramref name="fallback"/> otherwise, or <c>null</c> when both are <c>null</c>.
    /// </returns>
    public static ChatOptions? Merge(ChatOptions? primary, ChatOptions? fallback)
    {
        if (primary is null) return fallback is null ? null : Copy(fallback);
        if (fallback is null) return Copy(primary);

        return new ChatOptions
        {
            Temperature = primary.Temperature ?? fallback.Temperature,
            MaxOutputTokens = primary.MaxOutputTokens ?? fallback.MaxOutputTokens,
            TopP = primary.TopP ?? fallback.TopP,
            TopK = primary.TopK ?? fallback.TopK,
            StopSequences = primary.StopSequences ?? fallback.StopSequences,
            Seed = primary.Seed ?? fallback.Seed,
            Tools = primary.Tools ?? fallback.Tools,
            ToolChoice = primary.ToolChoice ?? fallback.ToolChoice,
            RequiredToolName = primary.RequiredToolName ?? fallback.RequiredToolName,
            ResponseFormat = primary.ResponseFormat ?? fallback.ResponseFormat,
            ResponseJsonSchema = primary.ResponseJsonSchema ?? fallback.ResponseJsonSchema
        };
    }

    /// <summary>Layers three sets of settings, most specific first.</summary>
    /// <param name="primary">The most specific settings, such as the per-call options.</param>
    /// <param name="secondary">The middle layer, such as one ensemble member's defaults.</param>
    /// <param name="tertiary">The least specific layer, such as a client-wide default.</param>
    /// <returns>A new instance carrying the first value supplied for each field, or <c>null</c> when all are <c>null</c>.</returns>
    public static ChatOptions? Merge(ChatOptions? primary, ChatOptions? secondary, ChatOptions? tertiary) =>
        Merge(Merge(primary, secondary), tertiary);

    /// <summary>Creates an independent copy of a set of settings.</summary>
    /// <param name="options">The settings to copy.</param>
    /// <returns>A new instance carrying the same values, or <c>null</c> when <paramref name="options"/> is <c>null</c>.</returns>
    public static ChatOptions? Copy(ChatOptions? options)
    {
        if (options is null) return null;
        return new ChatOptions
        {
            Temperature = options.Temperature,
            MaxOutputTokens = options.MaxOutputTokens,
            TopP = options.TopP,
            TopK = options.TopK,
            StopSequences = options.StopSequences,
            Seed = options.Seed,
            Tools = options.Tools,
            ToolChoice = options.ToolChoice,
            RequiredToolName = options.RequiredToolName,
            ResponseFormat = options.ResponseFormat,
            ResponseJsonSchema = options.ResponseJsonSchema
        };
    }
}
