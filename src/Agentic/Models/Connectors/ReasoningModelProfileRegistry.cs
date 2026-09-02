using System.Collections.ObjectModel;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>An ordered set of reasoning-model profiles, resolved against a model id by prefix.</summary>
/// <remarks>
/// <para>
/// <see cref="Default"/> reproduces the reference implementation's model list exactly: the OpenAI o-series
/// (<c>o1</c>, <c>o3</c>, <c>o4-</c>), the GPT-5 family, and the two GPT-OSS releases. Upstream keeps them in one
/// flat tuple of prefixes and calls <c>str.startswith</c> on the lower-cased model name; the split into three named
/// profiles here changes nothing about which ids match, and buys a name to put in a diagnostic and a place to hang
/// a family-specific difference should the providers ever diverge.
/// </para>
/// <para>
/// Two of upstream's prefixes are redundant and are kept anyway so the match set is provably identical: a name
/// starting with <c>o1</c> already covers <c>o1-mini</c>, so the separate <c>o1-</c> entry can never match anything
/// new. Note also what the list implies — an id such as <c>o1-preview-2024-09-12</c> matches, but so would an
/// unrelated model that happens to start with those two characters, and a reasoning model published under a name
/// outside these prefixes will not match at all. That is the reference behaviour; supply your own registry when
/// you need a different rule.
/// </para>
/// <para>
/// Resolution scans in order and returns the first match, so a caller-supplied registry can put a narrow profile
/// ahead of a broad one. Instances are immutable and safe to share.
/// </para>
/// <para><b>For Beginners:</b> This is the phone book the library uses to decide whether the model you named is a
/// "reasoning" model that needs special request handling. You get a sensible built-in list for free through
/// <see cref="Default"/>. If you use a provider that names its reasoning models differently, build your own
/// registry with the prefixes it uses and pass it in the options.</para>
/// </remarks>
public sealed class ReasoningModelProfileRegistry
{
    private static readonly ReasoningModelProfileRegistry DefaultRegistry = CreateDefault();

    private readonly ReasoningModelProfile[] _profiles;

    /// <summary>Initializes a registry from an ordered list of profiles.</summary>
    /// <param name="profiles">The profiles, most specific first. Must contain at least one entry.</param>
    /// <exception cref="ArgumentNullException"><paramref name="profiles"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="profiles"/> is empty or contains a <c>null</c> entry.</exception>
    public ReasoningModelProfileRegistry(IReadOnlyList<ReasoningModelProfile> profiles)
    {
        Guard.NotNull(profiles);
        if (profiles.Count == 0)
        {
            throw new ArgumentException("A reasoning-model registry needs at least one profile.", nameof(profiles));
        }

        var copy = new ReasoningModelProfile[profiles.Count];
        for (int index = 0; index < profiles.Count; index++)
        {
            ReasoningModelProfile profile = profiles[index];
            if (profile is null)
            {
                throw new ArgumentException("A reasoning-model registry cannot contain a null profile.", nameof(profiles));
            }

            copy[index] = profile;
        }

        _profiles = copy;
    }

    /// <summary>Gets the shared registry covering the model families the reference implementation knows about.</summary>
    public static ReasoningModelProfileRegistry Default => DefaultRegistry;

    /// <summary>Gets the profiles in resolution order.</summary>
    public IReadOnlyList<ReasoningModelProfile> Profiles => new ReadOnlyCollection<ReasoningModelProfile>(_profiles);

    /// <summary>Builds a new registry containing the built-in profiles.</summary>
    /// <returns>A registry with the OpenAI o-series, GPT-5, and GPT-OSS families.</returns>
    /// <remarks>
    /// Use this as the starting point for a customized registry: take <see cref="Profiles"/>, append your own, and
    /// construct a new registry from the combined list.
    /// </remarks>
    public static ReasoningModelProfileRegistry CreateDefault() => new(new[]
    {
        // Upstream's OPENAI_REASONING_MODEL_PREFIXES, split by family. "o1-" and "o3-" are subsumed by "o1" and
        // "o3" but are listed so the match set is byte-identical to the reference tuple.
        new ReasoningModelProfile("openai-o-series", new[] { "o1", "o1-", "o3", "o3-", "o4-" }),
        new ReasoningModelProfile("openai-gpt-5", new[] { "gpt-5", "gpt-5-" }),
        new ReasoningModelProfile("openai-gpt-oss", new[] { "gpt-oss-120b", "gpt-oss-20b" })
    });

    /// <summary>Finds the first profile that claims a model id.</summary>
    /// <param name="modelId">The model id to resolve; <c>null</c>, empty, or unmatched returns <c>null</c>.</param>
    /// <returns>The matching profile, or <c>null</c> when the model is an ordinary chat model.</returns>
    public ReasoningModelProfile? Find(string? modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId)) return null;
        foreach (ReasoningModelProfile profile in _profiles)
        {
            if (profile.Matches(modelId)) return profile;
        }

        return null;
    }

    /// <summary>Reports whether any profile claims a model id.</summary>
    /// <param name="modelId">The model id to test.</param>
    /// <returns><c>true</c> when the id belongs to a known reasoning-model family.</returns>
    public bool IsReasoningModel(string? modelId) => Find(modelId) is not null;
}
