using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for Google's Gemini models via the <c>generateContent</c> API. Unlike the
/// OpenAI-compatible providers, Gemini has a bespoke wire format (contents/parts, systemInstruction,
/// generationConfig, functionDeclarations), which this connector maps to and from the unified agentic model
/// types — including native function calling and usage.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Streaming is provided as a single-shot fallback (it returns the complete answer as one update rather than
/// incremental tokens; tool-call deltas are not streamed) — use <see cref="ChatClientBase{T}.GetResponseAsync"/>
/// for native tool calling. Tool calling round-trips in Gemini's native format: assistant tool-call turns are
/// sent back as <c>functionCall</c> parts and tool results as <c>functionResponse</c> parts.
/// </para>
/// <para><b>For Beginners:</b> Same agent code, pointed at Google Gemini. This class quietly translates
/// between AiDotNet's message format and Gemini's so everything else just works.
/// </para>
/// </remarks>
public sealed class GeminiChatClient<T> : ChatClientBase<T>
{
    private static readonly JsonSerializerSettings JsonSettings = new() { NullValueHandling = NullValueHandling.Ignore };

    private readonly string _apiKey;
    private readonly string _baseUrl;

    /// <summary>
    /// Initializes a new Gemini client.
    /// </summary>
    /// <param name="apiKey">The Google AI API key.</param>
    /// <param name="modelName">The model id (default <c>gemini-1.5-flash</c>).</param>
    /// <param name="baseUrl">The models base URL. <c>null</c> uses the public Generative Language API.</param>
    /// <param name="httpClient">Optional HTTP client (for testing or custom handlers).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is empty/whitespace.</exception>
    public GeminiChatClient(string apiKey, string modelName = "gemini-1.5-flash", string? baseUrl = null, HttpClient? httpClient = null)
        : base(httpClient)
    {
        ValidateApiKey(apiKey);
        Guard.NotNullOrWhiteSpace(modelName);
        _apiKey = apiKey;
        _baseUrl = baseUrl ?? "https://generativelanguage.googleapis.com/v1beta/models";
        ModelId = modelName;
    }

    /// <inheritdoc/>
    protected override async Task<ChatResponse> GetResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        CancellationToken cancellationToken)
    {
        var payload = BuildRequest(messages, options);
        var url = $"{_baseUrl}/{ModelId}:generateContent?key={_apiKey}";
        using var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = new StringContent(payload.ToString(Formatting.None), Encoding.UTF8, "application/json"),
        };

        using var response = await HttpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var root = JObject.Parse(body);

        var candidate = (root["candidates"] as JArray)?.OfType<JObject>().FirstOrDefault();
        var contents = new List<AiContent>();
        if (candidate?["content"]?["parts"] is JArray parts)
        {
            var callIndex = 0;
            foreach (var part in parts)
            {
                var text = (string?)part["text"];
                if (text is not null && text.Length > 0)
                {
                    contents.Add(new TextContent(text));
                }
                else if (part["functionCall"] is JObject functionCall)
                {
                    var name = (string?)functionCall["name"];
                    if (name is not null && name.Trim().Length > 0)
                    {
                        var args = functionCall["args"] as JObject ?? new JObject();
                        contents.Add(new ToolCallContent($"gemini-call-{callIndex}", name, args.ToString(Formatting.None)));
                        callIndex++;
                    }
                }
            }
        }

        if (contents.Count == 0)
        {
            contents.Add(new TextContent(string.Empty));
        }

        var finishReason = MapFinishReason((string?)candidate?["finishReason"], contents);
        var usage = ParseUsage(root["usageMetadata"] as JObject);
        return new ChatResponse(ChatMessage.Assistant(contents), finishReason, usage, ModelId);
    }

    /// <inheritdoc/>
    protected override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // Single-shot fallback: emit the complete response as one update (non-incremental).
        var response = await GetResponseCoreAsync(messages, options, cancellationToken).ConfigureAwait(false);
        yield return new ChatResponseUpdate(role: ChatRole.Assistant);
        if (response.Message.Text.Length > 0)
        {
            yield return ChatResponseUpdate.ForText(response.Message.Text);
        }

        yield return ChatResponseUpdate.ForFinish(response.FinishReason, response.Usage);
    }

    private JObject BuildRequest(IReadOnlyList<ChatMessage> messages, ChatOptions options)
    {
        var contents = new JArray();
        var systemParts = new JArray();

        // Gemini's functionResponse payload is keyed by function NAME, while
        // the unified model keys tool results by call id — walk the assistant
        // tool-call turns first-to-last so each result can be mapped back.
        var callIdToName = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var message in messages)
        {
            switch (message.Role)
            {
                case ChatRole.System:
                    systemParts.Add(new JObject { ["text"] = message.Text });
                    break;
                case ChatRole.Assistant:
                    // Preserve structured tool calls as native functionCall
                    // parts — flattening them to text would break the second
                    // leg of tool calling (the model could not correlate the
                    // functionResponse that follows).
                    contents.Add(new JObject { ["role"] = "model", ["parts"] = BuildModelParts(message, callIdToName) });
                    break;
                case ChatRole.Tool:
                    contents.Add(new JObject { ["role"] = "user", ["parts"] = BuildFunctionResponseParts(message, callIdToName) });
                    break;
                default:
                    contents.Add(new JObject { ["role"] = "user", ["parts"] = TextParts(message.Text) });
                    break;
            }
        }

        var payload = new JObject { ["contents"] = contents };

        if (systemParts.Count > 0)
        {
            payload["systemInstruction"] = new JObject { ["parts"] = systemParts };
        }

        var generationConfig = BuildGenerationConfig(options);
        if (generationConfig.Count > 0)
        {
            payload["generationConfig"] = generationConfig;
        }

        if (options.Tools is { Count: > 0 } tools && options.ToolChoice != ToolChoiceMode.None)
        {
            payload["tools"] = new JArray { new JObject { ["functionDeclarations"] = BuildFunctionDeclarations(tools) } };
        }

        return payload;
    }

    private static JArray TextParts(string text) => new() { new JObject { ["text"] = text } };

    // Assistant turn → model parts, preserving functionCall structure for tool-call turns.
    private static JArray BuildModelParts(ChatMessage message, Dictionary<string, string> callIdToName)
    {
        var parts = new JArray();
        foreach (var content in message.Contents)
        {
            switch (content)
            {
                case TextContent text when text.Text.Length > 0:
                    parts.Add(new JObject { ["text"] = text.Text });
                    break;
                case ToolCallContent call:
                    callIdToName[call.CallId] = call.ToolName;
                    parts.Add(new JObject
                    {
                        ["functionCall"] = new JObject
                        {
                            ["name"] = call.ToolName,
                            ["args"] = ParseArgumentsObject(call.ArgumentsJson),
                        },
                    });
                    break;
            }
        }

        if (parts.Count == 0)
        {
            parts.Add(new JObject { ["text"] = message.Text });
        }

        return parts;
    }

    // Tool-result turn → user parts carrying native functionResponse payloads.
    private static JArray BuildFunctionResponseParts(ChatMessage message, Dictionary<string, string> callIdToName)
    {
        var parts = new JArray();
        foreach (var content in message.Contents)
        {
            if (content is not ToolResultContent result)
            {
                continue;
            }

            var functionName = callIdToName.TryGetValue(result.CallId, out var name) ? name : result.CallId;
            var responsePayload = ParseResultObject(result.Result);
            if (result.IsError)
            {
                responsePayload["error"] = true;
            }

            parts.Add(new JObject
            {
                ["functionResponse"] = new JObject
                {
                    ["name"] = functionName,
                    ["response"] = responsePayload,
                },
            });
        }

        if (parts.Count == 0)
        {
            // No structured result content — fall back to plain text so the
            // turn is never silently dropped.
            parts.Add(new JObject { ["text"] = message.Text });
        }

        return parts;
    }

    private static JObject ParseArgumentsObject(string argumentsJson)
    {
        if (argumentsJson.Trim().Length == 0)
        {
            return new JObject();
        }

        try
        {
            return JObject.Parse(argumentsJson);
        }
        catch (JsonException)
        {
            // The unified model carries arguments as the raw provider string;
            // a non-object payload is preserved verbatim under a single key.
            return new JObject { ["value"] = argumentsJson };
        }
    }

    private static JObject ParseResultObject(string result)
    {
        try
        {
            if (JToken.Parse(result) is JObject resultObject)
            {
                return new JObject(resultObject);
            }
        }
        catch (JsonException)
        {
            // Not JSON — wrap below.
        }

        // Gemini requires functionResponse.response to be an OBJECT; wrap
        // scalar/text tool output the way Google's examples do.
        return new JObject { ["content"] = result };
    }

    private static JObject BuildGenerationConfig(ChatOptions options)
    {
        var config = new JObject();
        if (options.Temperature is { } temperature)
        {
            config["temperature"] = temperature;
        }

        if (options.MaxOutputTokens is { } maxTokens)
        {
            config["maxOutputTokens"] = maxTokens;
        }

        if (options.TopP is { } topP)
        {
            config["topP"] = topP;
        }

        if (options.TopK is { } topK)
        {
            config["topK"] = topK;
        }

        return config;
    }

    private static JArray BuildFunctionDeclarations(IReadOnlyList<AiToolDefinition> tools)
    {
        var declarations = new JArray();
        foreach (var tool in tools)
        {
            declarations.Add(new JObject
            {
                ["name"] = tool.Name,
                ["description"] = tool.Description,
                ["parameters"] = tool.ParametersSchema,
            });
        }

        return declarations;
    }

    private static ChatFinishReason MapFinishReason(string? reason, List<AiContent> contents)
    {
        if (contents.OfType<ToolCallContent>().Any())
        {
            return ChatFinishReason.ToolCalls;
        }

        return reason switch
        {
            "STOP" => ChatFinishReason.Stop,
            "MAX_TOKENS" => ChatFinishReason.Length,
            "SAFETY" => ChatFinishReason.ContentFilter,
            "RECITATION" => ChatFinishReason.ContentFilter,
            null => ChatFinishReason.Stop,
            _ => ChatFinishReason.Unknown,
        };
    }

    private static ChatUsage? ParseUsage(JObject? usageMetadata)
    {
        if (usageMetadata is null)
        {
            return null;
        }

        var input = (int?)usageMetadata["promptTokenCount"] ?? 0;
        var output = (int?)usageMetadata["candidatesTokenCount"] ?? 0;
        return new ChatUsage(input, output);
    }
}
