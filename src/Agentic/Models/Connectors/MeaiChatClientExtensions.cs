using AiDotNet.Agentic.Models;
using Meai = Microsoft.Extensions.AI;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Fluent bridges between AiDotNet's <see cref="IChatClient{T}"/> and Microsoft.Extensions.AI's
/// <see cref="Microsoft.Extensions.AI.IChatClient"/>, in both directions, with full tool-calling support.
/// </summary>
public static class MeaiChatClientExtensions
{
    /// <summary>
    /// Exposes an AiDotNet chat client as a Microsoft.Extensions.AI client, so MEAI-aware code (Semantic
    /// Kernel, the MEAI middleware pipeline, etc.) can consume an AiDotNet model — including its tool calls.
    /// </summary>
    /// <typeparam name="T">The AiDotNet client's numeric type.</typeparam>
    /// <param name="client">The AiDotNet client to wrap.</param>
    /// <returns>A <see cref="Microsoft.Extensions.AI.IChatClient"/> backed by <paramref name="client"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> is <c>null</c>.</exception>
    public static Meai.IChatClient AsMeaiChatClient<T>(this IChatClient<T> client) =>
        new AiDotNetMeaiChatClient<T>(client);

    /// <summary>
    /// Wraps a Microsoft.Extensions.AI chat client as an AiDotNet <see cref="IChatClient{T}"/>, inheriting the
    /// .NET ecosystem's connectors (OpenAI, Azure, Ollama, …) — including tool calling.
    /// </summary>
    /// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
    /// <param name="client">The Microsoft.Extensions.AI client to wrap.</param>
    /// <param name="modelId">Optional model id reported by the adapter.</param>
    /// <returns>An <see cref="IChatClient{T}"/> backed by <paramref name="client"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> is <c>null</c>.</exception>
    public static IChatClient<T> AsAgenticChatClient<T>(this Meai.IChatClient client, string? modelId = null) =>
        new MeaiChatClient<T>(client, modelId);
}
