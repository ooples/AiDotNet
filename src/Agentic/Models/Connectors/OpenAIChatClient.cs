using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for OpenAI's Chat Completions API with native tool calling, streaming,
/// structured output, and multimodal (image) input.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// Translates the provider-neutral <see cref="ChatMessage"/>/<see cref="ChatOptions"/> model into OpenAI's
/// wire format and back, mapping provider strings (roles, <c>finish_reason</c>) onto the library's enums.
/// The same wire format is used by Azure OpenAI, so that connector derives from this one.
/// </para>
/// <para><b>For Beginners:</b> This is the adapter that lets the rest of the library talk to OpenAI models
/// (GPT-4o and friends) without knowing any OpenAI-specific details.
/// </para>
/// </remarks>
public class OpenAIChatClient<T> : ChatClientBase<T>
{
    private static readonly JsonSerializerSettings JsonSettings = new() { NullValueHandling = NullValueHandling.Ignore };

    private readonly string _apiKey;
    private readonly string _endpoint;

    /// <summary>
    /// Initializes a new OpenAI chat client.
    /// </summary>
    /// <param name="apiKey">The OpenAI API key.</param>
    /// <param name="modelName">The model id (default <c>gpt-4o</c>).</param>
    /// <param name="endpoint">Optional custom endpoint (defaults to the public OpenAI Chat Completions URL).</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is empty/whitespace.</exception>
    public OpenAIChatClient(
        string apiKey,
        string modelName = "gpt-4o",
        string? endpoint = null,
        HttpClient? httpClient = null)
        : base(httpClient)
    {
        ValidateApiKey(apiKey);
        Guard.NotNullOrWhiteSpace(modelName);
        _apiKey = apiKey;
        _endpoint = endpoint ?? "https://api.openai.com/v1/chat/completions";
        ModelId = modelName;
    }

    /// <inheritdoc/>
    protected override async Task<ChatResponse> GetResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        CancellationToken cancellationToken)
    {
        var payload = BuildRequest(messages, options, stream: false);
        using var request = CreateHttpRequest(payload);
        using var response = await HttpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        await EnsureSuccessAsync(response).ConfigureAwait(false);

        var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var root = JObject.Parse(body);
        var choice = root["choices"]?.FirstOrDefault();
        var messageJson = choice?["message"] as JObject;

        var contents = new List<AiContent>();
        var text = (string?)messageJson?["content"] ?? string.Empty;
        if (text.Length > 0)
        {
            contents.Add(new TextContent(text));
        }

        if (messageJson?["tool_calls"] is JArray toolCalls)
        {
            foreach (var toolCall in toolCalls)
            {
                var id = (string?)toolCall["id"] ?? string.Empty;
                var function = toolCall["function"] as JObject;
                var name = (string?)function?["name"] ?? string.Empty;
                var arguments = (string?)function?["arguments"];
                if (!string.IsNullOrWhiteSpace(id) && !string.IsNullOrWhiteSpace(name))
                {
                    contents.Add(new ToolCallContent(id, name, arguments));
                }
            }
        }

        if (contents.Count == 0)
        {
            contents.Add(new TextContent(string.Empty));
        }

        var finishReason = ParseFinishReason((string?)choice?["finish_reason"]);
        var usage = ParseUsage(root["usage"] as JObject);
        var modelId = (string?)root["model"] ?? ModelId;

        return new ChatResponse(ChatMessage.Assistant(contents), finishReason, usage, modelId);
    }

    /// <inheritdoc/>
    protected override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var payload = BuildRequest(messages, options, stream: true);
        payload["stream_options"] = new JObject { ["include_usage"] = true };

        using var request = CreateHttpRequest(payload);
        using var response = await HttpClient
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);
        await EnsureSuccessAsync(response).ConfigureAwait(false);

#if NET5_0_OR_GREATER
        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
#else
        using var stream = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
#endif
        using var reader = new StreamReader(stream);

        ChatFinishReason? finishReason = null;
        ChatUsage? usage = null;
        bool roleEmitted = false;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync().ConfigureAwait(false);
            if (line is null)
            {
                break;
            }

            if (line.Length == 0 || !line.StartsWith("data:", StringComparison.Ordinal))
            {
                continue;
            }

            var data = line.Substring("data:".Length).Trim();
            if (data == "[DONE]")
            {
                break;
            }

            var chunk = JObject.Parse(data);

            if (chunk["usage"] is JObject usageJson)
            {
                usage = ParseUsage(usageJson) ?? usage;
            }

            var choice = chunk["choices"]?.FirstOrDefault();
            if (choice is null)
            {
                continue;
            }

            var delta = choice["delta"] as JObject;

            if (!roleEmitted && delta?["role"] is not null)
            {
                roleEmitted = true;
                yield return new ChatResponseUpdate(role: ChatRole.Assistant);
            }

            var contentDelta = (string?)delta?["content"] ?? string.Empty;
            if (contentDelta.Length > 0)
            {
                yield return ChatResponseUpdate.ForText(contentDelta);
            }

            if (delta?["tool_calls"] is JArray toolCallDeltas)
            {
                foreach (var toolCallDelta in toolCallDeltas)
                {
                    var index = (int?)toolCallDelta["index"] ?? 0;
                    var id = (string?)toolCallDelta["id"];
                    var function = toolCallDelta["function"] as JObject;
                    var name = (string?)function?["name"];
                    var argsFragment = (string?)function?["arguments"];
                    yield return ChatResponseUpdate.ForToolCall(
                        new StreamingToolCallUpdate(index, id, name, argsFragment));
                }
            }

            var reason = (string?)choice["finish_reason"];
            if (!string.IsNullOrEmpty(reason))
            {
                finishReason = ParseFinishReason(reason);
            }
        }

        yield return ChatResponseUpdate.ForFinish(finishReason ?? ChatFinishReason.Stop, usage);
    }

    /// <summary>
    /// Creates the HTTP request and applies authentication. Overridable so derived connectors (Azure) can
    /// change the endpoint and auth scheme.
    /// </summary>
    /// <param name="payload">The request body.</param>
    /// <returns>The configured HTTP request message.</returns>
    protected virtual HttpRequestMessage CreateHttpRequest(JObject payload)
    {
        var json = JsonConvert.SerializeObject(payload, JsonSettings);
        var request = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
        };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        return request;
    }

    /// <summary>
    /// Builds the OpenAI Chat Completions request body from the conversation and options.
    /// </summary>
    /// <param name="messages">The conversation.</param>
    /// <param name="options">The effective options.</param>
    /// <param name="stream">Whether to request a streamed response.</param>
    /// <returns>The request body as a <see cref="JObject"/>.</returns>
    protected virtual JObject BuildRequest(IReadOnlyList<ChatMessage> messages, ChatOptions options, bool stream)
    {
        var payload = new JObject
        {
            ["model"] = ModelId,
            ["messages"] = BuildMessages(messages)
        };

        if (stream) payload["stream"] = true;
        if (options.Temperature is { } temperature) payload["temperature"] = temperature;
        if (options.MaxOutputTokens is { } maxTokens) payload["max_tokens"] = maxTokens;
        if (options.TopP is { } topP) payload["top_p"] = topP;
        if (options.Seed is { } seed) payload["seed"] = seed;

        if (options.StopSequences is { Count: > 0 } stops)
        {
            payload["stop"] = new JArray(stops.Cast<object>().ToArray());
        }

        if (options.Tools is { Count: > 0 } tools)
        {
            payload["tools"] = BuildTools(tools);
            payload["tool_choice"] = BuildToolChoice(options);
        }

        var responseFormat = BuildResponseFormat(options);
        if (responseFormat is not null)
        {
            payload["response_format"] = responseFormat;
        }

        return payload;
    }

    private static JArray BuildMessages(IReadOnlyList<ChatMessage> messages)
    {
        var array = new JArray();
        foreach (var message in messages)
        {
            switch (message.Role)
            {
                case ChatRole.Tool:
                    foreach (var part in message.Contents.OfType<ToolResultContent>())
                    {
                        array.Add(new JObject
                        {
                            ["role"] = "tool",
                            ["tool_call_id"] = part.CallId,
                            ["content"] = part.Result
                        });
                    }
                    break;

                case ChatRole.Assistant:
                    array.Add(BuildAssistantMessage(message));
                    break;

                default:
                    array.Add(new JObject
                    {
                        ["role"] = message.Role == ChatRole.System ? "system" : "user",
                        ["content"] = BuildContent(message)
                    });
                    break;
            }
        }

        return array;
    }

    private static JObject BuildAssistantMessage(ChatMessage message)
    {
        var json = new JObject { ["role"] = "assistant" };

        var text = message.Text;
        json["content"] = string.IsNullOrEmpty(text) ? JValue.CreateNull() : text;

        var toolCalls = message.Contents.OfType<ToolCallContent>().ToList();
        if (toolCalls.Count > 0)
        {
            var calls = new JArray();
            foreach (var call in toolCalls)
            {
                calls.Add(new JObject
                {
                    ["id"] = call.CallId,
                    ["type"] = "function",
                    ["function"] = new JObject
                    {
                        ["name"] = call.ToolName,
                        ["arguments"] = call.ArgumentsJson
                    }
                });
            }

            json["tool_calls"] = calls;
        }

        return json;
    }

    private static JToken BuildContent(ChatMessage message)
    {
        var images = message.Contents.OfType<ImageContent>().ToList();
        if (images.Count == 0)
        {
            return message.Text;
        }

        var parts = new JArray();
        foreach (var textPart in message.Contents.OfType<TextContent>())
        {
            parts.Add(new JObject { ["type"] = "text", ["text"] = textPart.Text });
        }

        foreach (var image in images)
        {
            string url;
            var data = image.Data;
            if (data is not null)
            {
                var mime = (image.MediaType ?? ImageMediaType.Png).ToMimeType();
                url = $"data:{mime};base64,{Convert.ToBase64String(data)}";
            }
            else
            {
                url = image.Uri ?? string.Empty;
            }

            parts.Add(new JObject
            {
                ["type"] = "image_url",
                ["image_url"] = new JObject { ["url"] = url }
            });
        }

        return parts;
    }

    private static JArray BuildTools(IReadOnlyList<AiToolDefinition> tools)
    {
        var array = new JArray();
        foreach (var tool in tools)
        {
            array.Add(new JObject
            {
                ["type"] = "function",
                ["function"] = new JObject
                {
                    ["name"] = tool.Name,
                    ["description"] = tool.Description,
                    ["parameters"] = tool.ParametersSchema
                }
            });
        }

        return array;
    }

    private static JToken BuildToolChoice(ChatOptions options)
    {
        switch (options.ToolChoice ?? ToolChoiceMode.Auto)
        {
            case ToolChoiceMode.None:
                return "none";
            case ToolChoiceMode.Required when !string.IsNullOrWhiteSpace(options.RequiredToolName):
                return new JObject
                {
                    ["type"] = "function",
                    ["function"] = new JObject { ["name"] = options.RequiredToolName }
                };
            case ToolChoiceMode.Required:
                return "required";
            default:
                return "auto";
        }
    }

    private static JObject? BuildResponseFormat(ChatOptions options)
    {
        switch (options.ResponseFormat ?? ChatResponseFormatKind.Text)
        {
            case ChatResponseFormatKind.Json:
                return new JObject { ["type"] = "json_object" };
            case ChatResponseFormatKind.JsonSchema when options.ResponseJsonSchema is not null:
                return new JObject
                {
                    ["type"] = "json_schema",
                    ["json_schema"] = new JObject
                    {
                        ["name"] = "response",
                        ["strict"] = true,
                        ["schema"] = options.ResponseJsonSchema
                    }
                };
            default:
                return null;
        }
    }

    private static ChatFinishReason ParseFinishReason(string? reason) => reason switch
    {
        "stop" => ChatFinishReason.Stop,
        "length" => ChatFinishReason.Length,
        "tool_calls" => ChatFinishReason.ToolCalls,
        "function_call" => ChatFinishReason.ToolCalls,
        "content_filter" => ChatFinishReason.ContentFilter,
        _ => ChatFinishReason.Unknown
    };

    private static ChatUsage? ParseUsage(JObject? usage)
    {
        if (usage is null)
        {
            return null;
        }

        var input = (int?)usage["prompt_tokens"] ?? 0;
        var output = (int?)usage["completion_tokens"] ?? 0;
        return new ChatUsage(input, output);
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var error = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
#if NET5_0_OR_GREATER
        throw new HttpRequestException(
            $"OpenAI request failed with status {(int)response.StatusCode}: {error}", null, response.StatusCode);
#else
        throw new HttpRequestException(
            $"OpenAI request failed with status {(int)response.StatusCode}: {error}");
#endif
    }
}
