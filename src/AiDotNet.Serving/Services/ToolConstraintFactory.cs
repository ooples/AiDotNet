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

        // tool_choice controls whether / which tool is forced:
        //   "none"            -> no tool call (free text)
        //   "auto" (default)  -> the model MAY call a tool; we do NOT force one. A hard grammar constraint
        //                        cannot express "either a valid tool call or free text", so auto preserves
        //                        normal text generation rather than forcing every request into a tool call.
        //   "required"        -> force a call to ANY provided tool
        //   {type:function,function:{name}} -> force a call to that named tool
        // Unknown strings and malformed objects are rejected (400) instead of silently forcing a call.
        string? forcedName = null;
        if (toolChoice is { Type: JTokenType.String })
        {
            switch (toolChoice.ToString())
            {
                case "none":
                case "auto":
                    return (null, false);
                case "required":
                    break; // force a call to any tool, handled below
                default:
                    throw new ArgumentException(
                        $"Invalid 'tool_choice' value '{toolChoice}'. Expected 'none', 'auto', 'required', or a function-selection object.");
            }
        }
        else if (toolChoice is null || toolChoice.Type == JTokenType.Null)
        {
            // tools supplied without an explicit tool_choice defaults to "auto".
            return (null, false);
        }
        else if (toolChoice is JObject tc)
        {
            forcedName = tc["function"]?["name"]?.ToString();
            if (string.IsNullOrEmpty(forcedName))
            {
                throw new ArgumentException(
                    "A 'tool_choice' object must be of the form {\"type\":\"function\",\"function\":{\"name\":\"...\"}}.");
            }
        }
        else
        {
            throw new ArgumentException("'tool_choice' must be a string ('none' | 'auto' | 'required') or a function-selection object.");
        }

        var eligible = tools
            .Where(t => t.Function is not null && !string.IsNullOrEmpty(t.Function.Name)
                        && (forcedName is null || t.Function.Name == forcedName))
            .ToList();
        if (eligible.Count == 0)
        {
            throw new ArgumentException(forcedName is not null
                ? $"tool_choice names an unknown function '{forcedName}'."
                : "tool_choice 'required' was set but no tool with a valid function name was provided.");
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
                },
                // Both members are mandatory: the schema compiler enforces "required", so the constraint
                // cannot accept {} or an object missing "arguments" (an incomplete tool-call envelope).
                ["required"] = new JArray("name", "arguments")
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
        // The envelope must carry an arguments object; a tool call missing it is not valid, so do not
        // silently substitute "{}" (which would report a call the model never actually specified).
        if (obj["arguments"] is not { } args)
        {
            return null;
        }
        string arguments = args.ToString(Formatting.None);

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
