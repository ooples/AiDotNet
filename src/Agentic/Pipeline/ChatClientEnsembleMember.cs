using AiDotNet.Agentic.Models;
using AiDotNet.Validation;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>One model in a weighted ensemble: the client, how often it is picked, and its own default settings.</summary>
/// <typeparam name="T">The numeric type the chat abstraction is parameterized on.</typeparam>
/// <remarks>
/// <para>
/// Mixing models during a search is a real strategy rather than a convenience: a strong, expensive model produces
/// the occasional large improvement while a cheap one produces most of the volume, and the weight is the dial
/// between them. Attaching settings to the member rather than to the call is what lets one member run hot and
/// another run cold within a single run.
/// </para>
/// <para>
/// <see cref="Name"/> is what appears in selection statistics and in the response's model id, so two members that
/// wrap the same underlying model with different settings remain distinguishable in the record — which they are
/// not upstream, where a member is identified only by its model string.
/// </para>
/// <para><b>For Beginners:</b> This describes one model in a group of models. The weight decides how often it
/// gets picked: a member with weight 3 is chosen about three times as often as one with weight 1. You can also
/// give it its own temperature or token limit, which is used whenever that member answers.</para>
/// </remarks>
public sealed class ChatClientEnsembleMember<T>
{
    /// <summary>Initializes an ensemble member.</summary>
    /// <param name="client">The chat client this member calls.</param>
    /// <param name="weight">The relative selection weight; must be positive and finite.</param>
    /// <param name="chatOptions">Settings applied to this member's calls, or <c>null</c> for none.</param>
    /// <param name="name">A stable name for statistics, or <c>null</c> to use the client's model id.</param>
    /// <exception cref="ArgumentNullException"><paramref name="client"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="weight"/> is not positive and finite.</exception>
    public ChatClientEnsembleMember(
        IChatClient<T> client,
        double weight = 1.0,
        ChatOptions? chatOptions = null,
        string? name = null)
    {
        Guard.NotNull(client);
        if (double.IsNaN(weight) || double.IsInfinity(weight) || weight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(weight), weight, "A member weight must be positive and finite.");
        }

        Client = client;
        Weight = weight;
        ChatOptions = ChatOptionsMerge.Copy(chatOptions);
        Name = name is { } supplied && supplied.Trim().Length > 0 ? supplied.Trim() : client.ModelId;
    }

    /// <summary>Gets the chat client this member calls.</summary>
    public IChatClient<T> Client { get; }

    /// <summary>Gets the relative selection weight.</summary>
    public double Weight { get; }

    /// <summary>Gets this member's default settings, or <c>null</c> when it has none.</summary>
    public ChatOptions? ChatOptions { get; }

    /// <summary>Gets the stable name used in statistics and in the reported response model id.</summary>
    public string Name { get; }

    /// <summary>Returns the member's name and weight.</summary>
    /// <returns>A short description.</returns>
    public override string ToString() =>
        $"ChatClientEnsembleMember({Name}, weight={Weight.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture)})";
}
