using System.Collections.Generic;
using Newtonsoft.Json;

namespace AiDotNet.Serving.Engine.Http;

/// <summary>One message in a chat conversation (OpenAI Chat Completions format).</summary>
public sealed class ChatMessage
{
    /// <summary>Role of the author: "system", "user", or "assistant".</summary>
    [JsonProperty("role")] public string Role { get; set; } = "user";

    /// <summary>The message text.</summary>
    [JsonProperty("content")] public string Content { get; set; } = string.Empty;
}

/// <summary>OpenAI-compatible request body for <c>POST /v1/chat/completions</c>.</summary>
public sealed class OpenAiChatRequest
{
    /// <summary>The model id.</summary>
    [JsonProperty("model")] public string? Model { get; set; }

    /// <summary>The conversation so far.</summary>
    [JsonProperty("messages")] public IReadOnlyList<ChatMessage> Messages { get; set; } = new List<ChatMessage>();

    /// <summary>Maximum tokens to generate.</summary>
    [JsonProperty("max_tokens")] public int MaxTokens { get; set; } = 128;

    /// <summary>Sampling temperature (0 = greedy).</summary>
    [JsonProperty("temperature")] public double Temperature { get; set; } = 1.0;

    /// <summary>Nucleus (top-p) sampling.</summary>
    [JsonProperty("top_p")] public double TopP { get; set; } = 1.0;

    /// <summary>Top-k sampling (0 disables).</summary>
    [JsonProperty("top_k")] public int TopK { get; set; }

    /// <summary>Presence penalty.</summary>
    [JsonProperty("presence_penalty")] public double PresencePenalty { get; set; }

    /// <summary>Frequency penalty.</summary>
    [JsonProperty("frequency_penalty")] public double FrequencyPenalty { get; set; }

    /// <summary>Optional RNG seed.</summary>
    [JsonProperty("seed")] public int? Seed { get; set; }

    /// <summary>Whether to stream the response as server-sent events.</summary>
    [JsonProperty("stream")] public bool Stream { get; set; }
}

/// <summary>The assistant message in a chat completion choice.</summary>
public sealed class ChatChoiceMessage
{
    /// <summary>Author role, always "assistant".</summary>
    [JsonProperty("role")] public string Role { get; set; } = "assistant";

    /// <summary>The generated content.</summary>
    [JsonProperty("content")] public string Content { get; set; } = string.Empty;
}

/// <summary>A streamed delta fragment of the assistant message.</summary>
public sealed class ChatChoiceDelta
{
    /// <summary>Author role (present on the first chunk).</summary>
    [JsonProperty("role", NullValueHandling = NullValueHandling.Ignore)] public string? Role { get; set; }

    /// <summary>Incremental content text.</summary>
    [JsonProperty("content", NullValueHandling = NullValueHandling.Ignore)] public string? Content { get; set; }
}

/// <summary>One choice in a chat completion (non-streaming or streaming).</summary>
public sealed class OpenAiChatChoice
{
    /// <summary>Choice index.</summary>
    [JsonProperty("index")] public int Index { get; set; }

    /// <summary>The full assistant message (non-streaming responses).</summary>
    [JsonProperty("message", NullValueHandling = NullValueHandling.Ignore)] public ChatChoiceMessage? Message { get; set; }

    /// <summary>The incremental delta (streaming chunks).</summary>
    [JsonProperty("delta", NullValueHandling = NullValueHandling.Ignore)] public ChatChoiceDelta? Delta { get; set; }

    /// <summary>Reason generation stopped, or null while streaming.</summary>
    [JsonProperty("finish_reason")] public string? FinishReason { get; set; }
}

/// <summary>OpenAI-compatible chat completion response (also used for streamed chunks).</summary>
public sealed class OpenAiChatResponse
{
    /// <summary>Unique response id.</summary>
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;

    /// <summary>Object type: "chat.completion" or "chat.completion.chunk".</summary>
    [JsonProperty("object")] public string Object { get; set; } = "chat.completion";

    /// <summary>Unix creation timestamp.</summary>
    [JsonProperty("created")] public long Created { get; set; }

    /// <summary>The model id.</summary>
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;

    /// <summary>The choices.</summary>
    [JsonProperty("choices")] public IReadOnlyList<OpenAiChatChoice> Choices { get; set; } = new List<OpenAiChatChoice>();

    /// <summary>Token usage (present on the final, non-streaming response).</summary>
    [JsonProperty("usage", NullValueHandling = NullValueHandling.Ignore)] public OpenAiUsage? Usage { get; set; }
}
