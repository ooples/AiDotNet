using AiDotNet.Agentic.Models;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>Pulls the JSON object out of a model reply that may wrap it in prose or a code fence.</summary>
/// <remarks>
/// <para>
/// A model asked for JSON often answers with JSON surrounded by something else: a fenced block, a sentence of
/// preamble, a closing remark. This finds the object regardless. The first choice is always to make the reply
/// unambiguous at the source — set <see cref="ChatOptions.ResponseFormat"/> to
/// <see cref="ChatResponseFormatKind.JsonSchema"/> and supply
/// <see cref="ChatOptions.ResponseJsonSchema"/>, which providers with constrained decoding honour exactly — and
/// to treat this as the fallback for models and endpoints that cannot.
/// </para>
/// <para>
/// Extraction is brace-aware rather than regular-expression based, which is the difference that matters: a
/// pattern that matches from the first <c>{</c> to the last <c>}</c> breaks the moment a string value inside the
/// object contains a brace, and one that stops at the first <c>}</c> breaks on any nested object. This scanner
/// tracks string literals and escapes, so a value such as <c>"reason": "use {n} buckets"</c> parses correctly.
/// </para>
/// <para><b>For Beginners:</b> You asked the AI for a JSON answer and it replied with JSON plus a sentence of
/// explanation, or wrapped it in a code block. This digs out the JSON and hands it back. If nothing in the reply
/// is valid JSON, it tells you so rather than throwing, which lets a caller retry or fall back instead of
/// crashing.</para>
/// </remarks>
public static class LlmJsonExtractor
{
    /// <summary>Tries to find and parse a JSON object in a model reply.</summary>
    /// <param name="response">The reply text.</param>
    /// <param name="json">The parsed object, when one was found.</param>
    /// <returns><c>true</c> when a JSON object was found and parsed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="response"/> is <c>null</c>.</exception>
    public static bool TryExtract(string response, out JObject json)
    {
        Guard.NotNull(response);

        foreach (string candidate in Candidates(response))
        {
            if (!TryParse(candidate, out JObject? parsed)) continue;
            json = parsed;
            return true;
        }

        json = new JObject();
        return false;
    }

    /// <summary>Finds and parses a JSON object in a model reply, or returns <c>null</c>.</summary>
    /// <param name="response">The reply text.</param>
    /// <returns>The parsed object, or <c>null</c> when the reply holds no valid JSON object.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="response"/> is <c>null</c>.</exception>
    public static JObject? Extract(string response) => TryExtract(response, out JObject json) ? json : null;

    /// <summary>Reads a numeric field, accepting a number or a numeric string.</summary>
    /// <param name="json">The parsed object.</param>
    /// <param name="name">The field name.</param>
    /// <param name="value">The value, when the field is present and numeric and finite.</param>
    /// <returns><c>true</c> when a finite number was read.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="json"/> or <paramref name="name"/> is <c>null</c>.</exception>
    public static bool TryReadNumber(JObject json, string name, out double value)
    {
        Guard.NotNull(json);
        Guard.NotNull(name);

        value = 0;
        JToken? token = json[name];
        if (token is null) return false;

        switch (token.Type)
        {
            case JTokenType.Integer:
            case JTokenType.Float:
                value = token.Value<double>();
                break;
            case JTokenType.String:
                // Models frequently quote their numbers; rejecting that would fail
                // an answer that is otherwise exactly what was asked for.
                if (!double.TryParse(
                        token.Value<string>() ?? string.Empty,
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture,
                        out value))
                {
                    return false;
                }

                break;
            default:
                return false;
        }

        return !double.IsNaN(value) && !double.IsInfinity(value);
    }

    private static IEnumerable<string> Candidates(string response)
    {
        foreach (string fenced in FencedBlocks(response, "json")) yield return fenced;
        foreach (string fenced in FencedBlocks(response, null)) yield return fenced;

        // Every balanced object in order, not just the first: a model that emits a
        // broken object and then a corrected one is common, and stopping at the
        // first candidate would discard the answer it actually meant.
        foreach (string balanced in BalancedObjects(response)) yield return balanced;

        yield return response.Trim();
    }

    private static IEnumerable<string> FencedBlocks(string response, string? requiredLabel)
    {
        int index = 0;
        while (index < response.Length)
        {
            int open = response.IndexOf("```", index, StringComparison.Ordinal);
            if (open < 0) yield break;

            int labelEnd = response.IndexOf('\n', open);
            if (labelEnd < 0) yield break;

            string label = response.Substring(open + 3, labelEnd - open - 3).Trim();
            int close = response.IndexOf("```", labelEnd + 1, StringComparison.Ordinal);
            if (close < 0) yield break;

            if (requiredLabel is null || string.Equals(label, requiredLabel, StringComparison.OrdinalIgnoreCase))
            {
                yield return response.Substring(labelEnd + 1, close - labelEnd - 1).Trim();
            }

            index = close + 3;
        }
    }

    private static IEnumerable<string> BalancedObjects(string response)
    {
        int depth = 0;
        int start = -1;
        bool inString = false;
        bool escaped = false;

        for (int index = 0; index < response.Length; index++)
        {
            char current = response[index];
            if (escaped)
            {
                escaped = false;
                continue;
            }

            if (current == '\\' && inString)
            {
                escaped = true;
                continue;
            }

            if (current == '"')
            {
                inString = !inString;
                continue;
            }

            // Braces inside a string value are data, not structure. A regular
            // expression cannot make that distinction, which is why the three
            // hand-rolled extractors this replaces mis-parse such replies.
            if (inString) continue;

            if (current == '{')
            {
                if (depth == 0) start = index;
                depth++;
                continue;
            }

            if (current != '}' || depth == 0) continue;
            depth--;
            if (depth == 0 && start >= 0)
            {
                yield return response.Substring(start, index - start + 1);
                start = -1;
            }
        }
    }

    private static bool TryParse(string candidate, out JObject json)
    {
        json = new JObject();
        if (candidate.Length == 0) return false;

        try
        {
            JToken token = JToken.Parse(candidate);
            if (token is not JObject parsed) return false;
            json = parsed;
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
