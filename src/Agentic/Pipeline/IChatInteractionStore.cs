using AiDotNet.Agentic.Models;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// Stores recorded chat interactions (request key → response) so model calls can be deterministically
/// replayed later without invoking any model. This is the backing store for record/replay — the foundation
/// for reproducible agent runs, cheap re-runs, time-travel debugging, and offline tests.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A cache of "for this exact request, the model said this". Record once against a
/// real model, then replay from the store forever — same inputs, same outputs, no API calls.
/// </para>
/// </remarks>
public interface IChatInteractionStore
{
    /// <summary>Saves the response for a request key (overwriting any existing entry).</summary>
    /// <param name="key">The canonical request key.</param>
    /// <param name="response">The response to record.</param>
    void Save(string key, ChatResponse response);

    /// <summary>Tries to get a recorded response for a request key.</summary>
    /// <param name="key">The canonical request key.</param>
    /// <param name="response">The recorded response, when found.</param>
    /// <returns><c>true</c> when a response was recorded for the key.</returns>
    bool TryGet(string key, out ChatResponse response);

    /// <summary>Gets the number of recorded interactions.</summary>
    int Count { get; }
}
