using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agents;

/// <summary>
/// Extracts structured hyperparameters from LLM text output using multiple parsing strategies.
/// </summary>
/// <remarks>
/// <para>
/// This parser uses a multi-strategy approach to extract hyperparameter key-value pairs from
/// free-form LLM responses. It tries the following strategies in order:
/// 1. JSON extraction (```json blocks or raw { } objects)
/// 2. Markdown bold pattern (**param:** value)
/// 3. Colon/equals-separated pattern (param: value or param = value)
/// </para>
/// <para>
/// Values are automatically type-inferred: int, double, bool, or string.
/// </para>
/// </remarks>
internal class HyperparameterResponseParser
{
    /// <summary>
    /// Parses LLM response text and extracts hyperparameter key-value pairs.
    /// </summary>
    /// <param name="llmResponse">The raw text response from the LLM.</param>
    /// <returns>A dictionary of hyperparameter names to their parsed values.</returns>
    public Dictionary<string, object> Parse(string llmResponse)
    {
        if (string.IsNullOrWhiteSpace(llmResponse))
        {
            return new Dictionary<string, object>();
        }

        // Strategy 1: Try JSON extraction first
        var result = TryParseJson(llmResponse);
        if (result.Count > 0)
        {
            return result;
        }

        // Strategy 2: Try markdown bold pattern
        result = TryParseMarkdownBold(llmResponse);
        if (result.Count > 0)
        {
            return result;
        }

        // Strategy 3: Fall back to colon/equals-separated
        result = TryParseColonSeparated(llmResponse);

        return result;
    }

    /// <summary>
    /// Tries to extract hyperparameters from JSON blocks in the LLM response.
    /// </summary>
    internal Dictionary<string, object> TryParseJson(string text)
    {
        var result = new Dictionary<string, object>();

        // Try ```json ... ``` blocks first
        var jsonBlockPattern = new Regex(@"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", RegexOptions.Compiled);
        foreach (Match match in jsonBlockPattern.Matches(text).Cast<Match>().Where(m => m.Success))
        {
            if (TryParseJsonObject(match.Groups[1].Value.Trim(), result))
            {
                return result;
            }
        }

        // Try raw { ... } objects
        var rawJsonPattern = new Regex(@"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", RegexOptions.Compiled);
        foreach (Match match in rawJsonPattern.Matches(text).Cast<Match>().Where(m => m.Success))
        {
            if (TryParseJsonObject(match.Value.Trim(), result))
            {
                return result;
            }
        }

        return result;
    }

    private static bool TryParseJsonObject(string json, Dictionary<string, object> result)
    {
        try
        {
            var jObject = JObject.Parse(json);
            foreach (var property in jObject.Properties())
            {
                var value = InferTypedValue(property.Value);
                if (value != null)
                {
                    result[property.Name] = value;
                }
            }
            return result.Count > 0;
        }
        catch (Newtonsoft.Json.JsonReaderException)
        {
            return false;
        }
    }

    /// <summary>
    /// Tries to extract hyperparameters from markdown bold patterns like **param:** value.
    /// </summary>
    internal Dictionary<string, object> TryParseMarkdownBold(string text)
    {
        var result = new Dictionary<string, object>();
        var pattern = new Regex(@"\*\*(\w[\w_]*?):\*\*\s*([^\s(]+)", RegexOptions.Compiled);

        foreach (Match match in pattern.Matches(text))
        {
            var name = match.Groups[1].Value;
            var rawValue = match.Groups[2].Value.TrimEnd(',', '.', ';');
            var value = InferTypedValue(rawValue);
            if (value != null)
            {
                result[name] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Tries to extract hyperparameters from colon or equals-separated lines.
    /// </summary>
    internal Dictionary<string, object> TryParseColonSeparated(string text)
    {
        var result = new Dictionary<string, object>();
        var pattern = new Regex(@"^\s*[-\*\s]*(\w[\w_]*?)\s*[:=][^\S\n]*([^\s(,]+)", RegexOptions.Multiline | RegexOptions.Compiled);

        foreach (Match match in pattern.Matches(text))
        {
            var name = match.Groups[1].Value;
            var rawValue = match.Groups[2].Value.TrimEnd(',', '.', ';');

            // Skip common non-hyperparameter patterns
            if (IsCommonNonParameter(name))
            {
                continue;
            }

            var value = InferTypedValue(rawValue);
            if (value != null)
            {
                result[name] = value;
            }
        }

        return result;
    }

    /// <summary>
    /// Infers the typed value from a raw string or JToken.
    /// </summary>
    internal static object? InferTypedValue(object rawValue)
    {
        if (rawValue is JToken jToken)
        {
            return jToken.Type switch
            {
                JTokenType.Integer => jToken.Value<long>() is var l && l >= int.MinValue && l <= int.MaxValue
                    ? (object)(int)l : (object)l,
                JTokenType.Float => jToken.Value<double>(),
                JTokenType.Boolean => jToken.Value<bool>(),
                JTokenType.String => InferTypedValue(jToken.Value<string>() ?? string.Empty),
                _ => jToken.ToString()
            };
        }

        if (rawValue is not string str || string.IsNullOrWhiteSpace(str))
        {
            return null;
        }

        // Remove quotes
        str = str.Trim('"', '\'');

        // Try int first
        if (int.TryParse(str, out var intVal))
        {
            return intVal;
        }

        // Try double
        if (double.TryParse(str, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var doubleVal))
        {
            return doubleVal;
        }

        // Try bool
        if (bool.TryParse(str, out var boolVal))
        {
            return boolVal;
        }

        // Keep as string
        return str;
    }

    private static bool IsCommonNonParameter(string name)
    {
        var nonParams = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "step", "note", "example", "reason", "because", "since",
            "model", "algorithm", "method", "approach", "result",
            "summary", "recommendation", "suggestion", "tip"
        };
        return nonParams.Contains(name);
    }
}
