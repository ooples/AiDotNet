using System.Collections.Generic;
using Newtonsoft.Json;

namespace AiDotNet.Serving.Engine.Http;

/// <summary>
/// OpenAI-compatible request body for <c>POST /v1/completions</c>. Fields mirror the OpenAI Completions API so
/// existing clients and SDKs work against an AiDotNet server unchanged.
/// </summary>
public sealed class OpenAiCompletionRequest
{
    /// <summary>The model id to generate with.</summary>
    [JsonProperty("model")] public string? Model { get; set; }

    /// <summary>The text prompt to continue.</summary>
    [JsonProperty("prompt")] public string Prompt { get; set; } = string.Empty;

    /// <summary>Maximum number of tokens to generate.</summary>
    [JsonProperty("max_tokens")] public int MaxTokens { get; set; } = 16;

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

    /// <summary>Optional RNG seed for reproducible sampling.</summary>
    [JsonProperty("seed")] public int? Seed { get; set; }

    /// <summary>Whether to stream the response as server-sent events.</summary>
    [JsonProperty("stream")] public bool Stream { get; set; }
}

/// <summary>One choice in a completion response.</summary>
public sealed class OpenAiCompletionChoice
{
    /// <summary>The generated text (a delta in streaming chunks, the full text otherwise).</summary>
    [JsonProperty("text")] public string Text { get; set; } = string.Empty;

    /// <summary>Choice index.</summary>
    [JsonProperty("index")] public int Index { get; set; }

    /// <summary>Reason generation stopped ("stop", "length"), or null while streaming.</summary>
    [JsonProperty("finish_reason")] public string? FinishReason { get; set; }
}

/// <summary>Token usage accounting.</summary>
public sealed class OpenAiUsage
{
    /// <summary>Prompt token count.</summary>
    [JsonProperty("prompt_tokens")] public int PromptTokens { get; set; }

    /// <summary>Generated token count.</summary>
    [JsonProperty("completion_tokens")] public int CompletionTokens { get; set; }

    /// <summary>Sum of prompt and completion tokens.</summary>
    [JsonProperty("total_tokens")] public int TotalTokens { get; set; }
}

/// <summary>OpenAI-compatible completion response (also used for each streamed chunk).</summary>
public sealed class OpenAiCompletionResponse
{
    /// <summary>Unique response id.</summary>
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;

    /// <summary>Object type: "text_completion".</summary>
    [JsonProperty("object")] public string Object { get; set; } = "text_completion";

    /// <summary>Unix creation timestamp.</summary>
    [JsonProperty("created")] public long Created { get; set; }

    /// <summary>The model id.</summary>
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;

    /// <summary>The generated choices.</summary>
    [JsonProperty("choices")] public IReadOnlyList<OpenAiCompletionChoice> Choices { get; set; }
        = new List<OpenAiCompletionChoice>();

    /// <summary>Token usage (present on the final, non-streaming response).</summary>
    [JsonProperty("usage", NullValueHandling = NullValueHandling.Ignore)] public OpenAiUsage? Usage { get; set; }
}
