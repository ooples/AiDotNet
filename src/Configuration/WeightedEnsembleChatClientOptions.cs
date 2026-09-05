using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;

namespace AiDotNet.Configuration;

/// <summary>Controls how a weighted ensemble picks a model, what it does when one fails, and what it reports as.</summary>
/// <remarks>
/// <para>
/// The setting that matters most here is <see cref="Seed"/>. Model selection is randomness inside an experiment,
/// and randomness inside an experiment has to be controllable or the experiment is not repeatable. Seeding the
/// selection stream means two runs of the same search choose the same models in the same order, so a difference
/// in outcome is attributable to the change under test rather than to which model happened to be picked on
/// iteration 40. The reference OpenEvolve ensemble seeds Python's process-global generator from the first model's
/// configuration only, which the rest of the process shares and can advance at any time.
/// </para>
/// <para>
/// <see cref="FallbackOnError"/> decides what a member's failure costs. With it on, a failing member's turn is
/// handed to the next-heaviest member instead of losing the iteration entirely — which matters most on the
/// rate-limited endpoints where a mixed ensemble is worth having in the first place.
/// </para>
/// <para><b>For Beginners:</b> These settings apply to a group of models used together. The seed makes the
/// choice of model repeatable, so running your search twice picks the same models both times. The fallback
/// setting decides whether a failure moves on to another model or gives up. The model id is just the name the
/// group reports for itself.</para>
/// </remarks>
public sealed class WeightedEnsembleChatClientOptions
{
    /// <summary>Gets or sets the seed for the member-selection stream.</summary>
    public ulong Seed { get; set; }

    /// <summary>Gets or sets the stream selector, so two ensembles sharing a seed still draw independently.</summary>
    public ulong Stream { get; set; }

    /// <summary>Gets or sets whether a failing member's turn passes to the remaining members in weight order.</summary>
    public bool FallbackOnError { get; set; } = true;

    /// <summary>Gets or sets settings applied to every member's calls beneath that member's own settings.</summary>
    public ChatOptions? DefaultChatOptions { get; set; }

    /// <summary>Gets or sets the model id the ensemble reports, or <c>null</c> for <c>"ensemble"</c>.</summary>
    public string? ModelId { get; set; }

    /// <summary>Gets or sets the maximum number of members called at once by a call-every-member request.</summary>
    public int MaxParallelism { get; set; } = 4;

    /// <summary>Gets or sets whether a response with no model id is stamped with the answering member's name.</summary>
    public bool StampAnsweringMember { get; set; } = true;

    /// <summary>Creates an independent copy so a running client is unaffected by later mutation.</summary>
    /// <returns>A new options instance carrying the same values.</returns>
    public WeightedEnsembleChatClientOptions Clone() => new()
    {
        Seed = Seed,
        Stream = Stream,
        FallbackOnError = FallbackOnError,
        DefaultChatOptions = ChatOptionsMerge.Copy(DefaultChatOptions),
        ModelId = ModelId,
        MaxParallelism = MaxParallelism,
        StampAnsweringMember = StampAnsweringMember
    };

    /// <summary>Validates the parallelism bound and the reported model id.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="MaxParallelism"/> is outside 1 to 64.</exception>
    /// <exception cref="ArgumentException"><see cref="ModelId"/> is present but empty or white space.</exception>
    public void Validate()
    {
        if (MaxParallelism < 1 || MaxParallelism > 64)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxParallelism), MaxParallelism,
                "Value must be between 1 and 64.");
        }

        if (ModelId is { } id && id.Trim().Length == 0)
        {
            throw new ArgumentException("ModelId cannot be empty or white space; use null for the default.", nameof(ModelId));
        }
    }
}
