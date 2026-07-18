using AiDotNet.Serving.Models.OpenAi;
using AiDotNet.Serving.StructuredOutput;
using AiDotNet.Tokenization.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Builds a guided-decoding constraint that forces the model to emit a valid OpenAI tool call, reusing the
/// structured-output engine: each tool's JSON-schema <c>parameters</c> is compiled to a regex and the model
/// is constrained to produce <c>{"name":"&lt;fn&gt;","arguments":&lt;args-matching-that-tool's-schema&gt;}</c>.
/// Also parses the generated JSON back into <see cref="ToolCall"/>s for the response.
/// </summary>
internal static class ToolConstraintFactory
{
    /// <summary>
    /// Builds a tool-call constraint from the request's <c>tools</c> and <c>tool_choice</c>.
    /// </summary>
    /// <returns>The constraint plus a flag indicating the output must be parsed as a tool call. Returns
    /// <c>(null, false)</c> when tools are absent or <c>tool_choice</c> is <c>"none"</c>.</returns>
    /// <exception cref="ArgumentException">Thrown when a specific tool_choice names an unknown function.</exception>
    public static (ITokenConstraint? Constraint, bool ToolMode) Build(
        List<ToolDefinition>? tools, JToken? toolChoice, ITokenizer tokenizer, int eosTokenId)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tools is null || tools.Count == 0)
        {
            return (null, false);
        }

        // tool_choice may be a string ("none" | "auto" | "required") or an object naming a function.
        string? choiceStr = toolChoice is { Type: JTokenType.String } ? toolChoice.ToString() : null;
        if (string.Equals(choiceStr, "none", StringComparison.Ordinal))
        {
            return (null, false);
        }

        string? forcedName = null;
        if (toolChoice is JObject tc && tc["function"]?["name"]?.ToString() is { Length: > 0 } fn)
        {
            forcedName = fn;
        }

        var eligible = tools
            .Where(t => t.Function is not null && !string.IsNullOrEmpty(t.Function.Name)
                        && (forcedName is null || t.Function.Name == forcedName))
            .ToList();
        if (eligible.Count == 0)
        {
            if (forcedName is not null)
            {
                throw new ArgumentException($"tool_choice names an unknown function '{forcedName}'.");
            }
            return (null, false);
        }

        // Build an alternation of one compact JSON object per eligible tool.
        var alternatives = new List<string>(eligible.Count);
        foreach (var t in eligible)
        {
            var fnDef = t.Function!;
            var wrapper = new JObject
            {
                ["type"] = "object",
                ["properties"] = new JObject
                {
                    ["name"] = new JObject { ["enum"] = new JArray(fnDef.Name) },
                    ["arguments"] = fnDef.Parameters ?? new JObject { ["type"] = "object" }
                }
            };
            alternatives.Add("(?:" + JsonSchemaConstraint.CompileToRegex(wrapper) + ")");
        }

        string regex = "(?:" + string.Join("|", alternatives) + ")";
        var vocab = StructuredOutputFactory.GetVocabStrings(tokenizer);
        return (new RegexTokenConstraint(regex, vocab, eosTokenId), true);
    }

    /// <summary>
    /// Parses a completed tool-call JSON string (<c>{"name":...,"arguments":...}</c>) into a single
    /// <see cref="ToolCall"/>. Returns null if the text is not the expected shape.
    /// </summary>
    public static List<ToolCall>? Parse(string json, string callId)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return null;
        }
        JObject obj;
        try
        {
            obj = JObject.Parse(json);
        }
        catch (JsonException)
        {
            return null;
        }

        string? name = obj["name"]?.ToString();
        if (string.IsNullOrEmpty(name))
        {
            return null;
        }
        // OpenAI encodes arguments as a JSON string; serialize the arguments object compactly.
        string arguments = obj["arguments"] is { } args ? args.ToString(Formatting.None) : "{}";

        return new List<ToolCall>
        {
            new ToolCall
            {
                Id = callId,
                Type = "function",
                Function = new FunctionCallOut { Name = name!, Arguments = arguments }
            }
        };
    }
}
