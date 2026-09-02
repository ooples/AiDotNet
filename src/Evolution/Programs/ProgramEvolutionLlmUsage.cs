using System.Globalization;

namespace AiDotNet.Evolution.Programs;

/// <summary>Immutable totals describing how much language-model work a program-evolution run consumed.</summary>
/// <remarks>
/// <para>
/// The counts separate two things that are easy to confuse. <see cref="Proposals"/> is how many times the search
/// asked for a new candidate; <see cref="ChatCalls"/> is how many requests were actually sent, which is larger
/// whenever an answer had to be retried with feedback. The gap between them is the honest cost of unusable answers,
/// and <see cref="AbandonedProposals"/> counts the proposals that never produced a child at all. The reference
/// OpenEvolve worker reports none of this: an unusable answer is logged and the iteration is silently lost.
/// </para>
/// <para>
/// Token counts are reported only when the provider supplies them, so a client that returns no usage information
/// leaves them at zero rather than guessing. Because they are provider-reported rather than derived, they are safe
/// to compare across runs on the same provider and unsafe to compare across different ones.
/// </para>
/// <para><b>For Beginners:</b> This is the bill for the AI part of a run. It tells you how many times a new program
/// was requested, how many messages that actually took, how often the model's answer had to be thrown away, and how
/// many tokens were spent if the provider told you. If <see cref="ChatCalls"/> is much larger than
/// <see cref="Proposals"/>, the model is frequently answering in a form the search cannot use, which is worth fixing
/// before scaling the run up.</para>
/// </remarks>
public sealed class ProgramEvolutionLlmUsage
{
    /// <summary>Initializes a usage total.</summary>
    /// <param name="proposals">How many candidate proposals were requested.</param>
    /// <param name="chatCalls">How many requests were sent to the chat client.</param>
    /// <param name="retries">How many of those requests were retries after an unusable answer.</param>
    /// <param name="abandonedProposals">How many proposals produced no child after every retry.</param>
    /// <param name="providerErrors">How many requests failed inside the chat client.</param>
    /// <param name="inputTokens">Prompt tokens reported by the provider.</param>
    /// <param name="outputTokens">Answer tokens reported by the provider.</param>
    /// <exception cref="ArgumentOutOfRangeException">Any count is negative.</exception>
    public ProgramEvolutionLlmUsage(
        long proposals = 0,
        long chatCalls = 0,
        long retries = 0,
        long abandonedProposals = 0,
        long providerErrors = 0,
        long inputTokens = 0,
        long outputTokens = 0)
    {
        RequireNonNegative(proposals, nameof(proposals));
        RequireNonNegative(chatCalls, nameof(chatCalls));
        RequireNonNegative(retries, nameof(retries));
        RequireNonNegative(abandonedProposals, nameof(abandonedProposals));
        RequireNonNegative(providerErrors, nameof(providerErrors));
        RequireNonNegative(inputTokens, nameof(inputTokens));
        RequireNonNegative(outputTokens, nameof(outputTokens));

        Proposals = proposals;
        ChatCalls = chatCalls;
        Retries = retries;
        AbandonedProposals = abandonedProposals;
        ProviderErrors = providerErrors;
        InputTokens = inputTokens;
        OutputTokens = outputTokens;
    }

    /// <summary>Gets a total with every count at zero, for runs that used no language model.</summary>
    public static ProgramEvolutionLlmUsage Empty { get; } = new();

    /// <summary>Gets how many candidate proposals were requested.</summary>
    public long Proposals { get; }

    /// <summary>Gets how many requests were sent to the chat client, including retries.</summary>
    public long ChatCalls { get; }

    /// <summary>Gets how many requests were retries sent after an unusable answer.</summary>
    public long Retries { get; }

    /// <summary>Gets how many proposals produced no child program after every permitted retry.</summary>
    public long AbandonedProposals { get; }

    /// <summary>Gets how many requests failed inside the chat client.</summary>
    public long ProviderErrors { get; }

    /// <summary>Gets the prompt tokens the provider reported; zero when it reported none.</summary>
    public long InputTokens { get; }

    /// <summary>Gets the answer tokens the provider reported; zero when it reported none.</summary>
    public long OutputTokens { get; }

    /// <summary>Gets the sum of the reported prompt and answer tokens.</summary>
    public long TotalTokens => InputTokens + OutputTokens;

    /// <summary>Gets the share of proposals that produced no child, or zero when nothing was proposed.</summary>
    public double AbandonRate => Proposals == 0 ? 0 : (double)AbandonedProposals / Proposals;

    /// <summary>Gets the average number of requests spent per proposal, or zero when nothing was proposed.</summary>
    public double CallsPerProposal => Proposals == 0 ? 0 : (double)ChatCalls / Proposals;

    /// <summary>Adds two totals, for a run driven by more than one operator.</summary>
    /// <param name="other">The totals to add to these.</param>
    /// <returns>A new total holding the element-wise sums.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is <c>null</c>.</exception>
    public ProgramEvolutionLlmUsage Add(ProgramEvolutionLlmUsage other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        return new ProgramEvolutionLlmUsage(
            Proposals + other.Proposals,
            ChatCalls + other.ChatCalls,
            Retries + other.Retries,
            AbandonedProposals + other.AbandonedProposals,
            ProviderErrors + other.ProviderErrors,
            InputTokens + other.InputTokens,
            OutputTokens + other.OutputTokens);
    }

    /// <summary>Returns the counts in a single line.</summary>
    /// <returns>A compact description of the totals.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "proposals={0}, calls={1}, retries={2}, abandoned={3}, errors={4}, tokens={5}",
        Proposals, ChatCalls, Retries, AbandonedProposals, ProviderErrors, TotalTokens);

    private static void RequireNonNegative(long value, string parameterName)
    {
        if (value < 0) throw new ArgumentOutOfRangeException(parameterName, value, "Value cannot be negative.");
    }
}
