using System.Text;
using AiDotNet.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serving.StructuredOutput;

/// <summary>
/// Compiles a (subset of) JSON Schema to a regular expression and builds a <see cref="RegexTokenConstraint"/>
/// from it, so a model can be forced to emit JSON conforming to the schema. This is the mechanism behind the
/// OpenAI-compatible <c>response_format: { type: "json_schema" }</c> guided-decoding mode.
/// </summary>
/// <remarks>
/// <para>
/// The output is <b>compact</b> JSON (no insignificant whitespace) so the constraint is tight and
/// deterministic. Supported schema constructs: <c>object</c> with ordered <c>properties</c> (all treated as
/// required, emitted in declared order), scalar types <c>string</c> / <c>integer</c> / <c>number</c> /
/// <c>boolean</c>, <c>string</c> with <c>enum</c> or <c>pattern</c>, and <c>array</c> of a scalar
/// <c>items</c> type. A top-level <c>enum</c> restricts to a fixed set of JSON literals.
/// </para>
/// <para><b>For Beginners:</b> hand it a schema describing the JSON you want (which fields, what types) and
/// it forces the model to fill in exactly that shape — no missing fields, no wrong types, always parseable.</para>
/// </remarks>
public static class JsonSchemaConstraint
{
    /// <summary>
    /// Builds a token constraint that forces output to match the given JSON Schema.
    /// </summary>
    /// <param name="schemaJson">The JSON Schema document as a JSON string.</param>
    /// <param name="tokenText">The vocabulary: token id -&gt; decoded string piece.</param>
    /// <param name="eosTokenId">The end-of-sequence token id, permitted once a full JSON value is complete.</param>
    public static RegexTokenConstraint FromSchema(string schemaJson, IReadOnlyList<string> tokenText, int eosTokenId)
    {
        Guard.NotNullOrWhiteSpace(schemaJson);
        JToken schema;
        try
        {
            schema = JToken.Parse(schemaJson);
        }
        catch (JsonException ex)
        {
            throw new ArgumentException($"Invalid JSON Schema document: {ex.Message}", nameof(schemaJson), ex);
        }
        string regex = CompileToRegex(schema);
        return new RegexTokenConstraint(regex, tokenText, eosTokenId);
    }

    /// <summary>
    /// Builds a token constraint that forces output to be any valid compact JSON value up to
    /// <paramref name="maxDepth"/> nesting (the OpenAI <c>json_object</c> mode). JSON is context-free, so a
    /// finite regex can only express bounded nesting; deeper structures require a grammar constraint.
    /// </summary>
    /// <param name="tokenText">The vocabulary: token id -&gt; decoded string piece.</param>
    /// <param name="eosTokenId">The end-of-sequence token id.</param>
    /// <param name="maxDepth">Maximum nesting depth of objects/arrays (default 3).</param>
    public static RegexTokenConstraint AnyJsonObject(IReadOnlyList<string> tokenText, int eosTokenId, int maxDepth = 3)
    {
        // The top level of json_object mode must be an object (per the OpenAI contract), with values that may
        // nest up to maxDepth.
        string value = JsonValueRegex(maxDepth);
        string obj = ObjectOfAnyRegex(value);
        return new RegexTokenConstraint(obj, tokenText, eosTokenId);
    }

    /// <summary>Compiles a JSON Schema token to a regex matching its compact JSON serialization.</summary>
    public static string CompileToRegex(JToken schema)
    {
        Guard.NotNull(schema);
        if (schema is not JObject obj)
        {
            // A bare `true`/`{}` schema accepts anything.
            return JsonValueRegex(3);
        }

        // enum takes precedence over type (it's the tightest constraint).
        if (obj["enum"] is JArray enumArr && enumArr.Count > 0)
        {
            return Alternation(enumArr.Select(JsonLiteralRegex));
        }

        string? type = obj["type"]?.ToString();
        switch (type)
        {
            case "object": return CompileObject(obj);
            case "array": return CompileArray(obj);
            case "string": return CompileString(obj);
            case "integer": return IntegerRegex;
            case "number": return NumberRegex;
            case "boolean": return BooleanRegex;
            case "null": return "null";
            case null: return JsonValueRegex(3); // untyped -> any value
            default:
                throw new ArgumentException($"Unsupported JSON Schema type '{type}'.", nameof(schema));
        }
    }

    private static string CompileObject(JObject obj)
    {
        var props = obj["properties"] as JObject;
        if (props is null || props.Count == 0)
        {
            // Object with no declared properties: match any compact object.
            return ObjectOfAnyRegex(JsonValueRegex(3));
        }

        // All declared properties, in declared order, are required (deterministic ordering keeps the DFA
        // small and avoids the exponential subset/permutation blow-up of optional-key ordering).
        var sb = new StringBuilder();
        sb.Append(Escape("{"));
        bool first = true;
        foreach (var prop in props.Properties())
        {
            if (!first) sb.Append(Escape(","));
            first = false;
            sb.Append('"').Append(EscapeLiteral(prop.Name)).Append('"');
            sb.Append(Escape(":"));
            sb.Append("(?:").Append(CompileToRegex(prop.Value)).Append(')');
        }
        sb.Append(Escape("}"));
        return sb.ToString();
    }

    private static string CompileArray(JObject obj)
    {
        string itemRegex = obj["items"] is JToken items ? CompileToRegex(items) : JsonValueRegex(3);
        string item = "(?:" + itemRegex + ")";
        // [] or [ item (,item)* ]
        return Escape("[") + "(?:" + item + "(?:" + Escape(",") + item + ")*)?" + Escape("]");
    }

    private static string CompileString(JObject obj)
    {
        if (obj["enum"] is JArray e && e.Count > 0)
        {
            return Alternation(e.Select(JsonLiteralRegex));
        }
        if (obj["pattern"]?.ToString() is { Length: > 0 } pattern)
        {
            // Embed the user pattern inside JSON quotes. The pattern constrains the string CONTENT.
            return "\"(?:" + pattern + ")\"";
        }
        return JsonStringRegex;
    }

    // ---- primitive JSON regexes (compact form) ----
    private const string JsonStringRegex = "\"(?:[^\"\\\\]|\\\\.)*\"";
    private const string IntegerRegex = "-?(?:0|[1-9][0-9]*)";
    private const string NumberRegex = "-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";
    private const string BooleanRegex = "(?:true|false)";

    // A JSON value with nesting bounded by depth. At depth 0, only scalars are allowed.
    private static string JsonValueRegex(int depth)
    {
        var parts = new List<string> { JsonStringRegex, NumberRegex, BooleanRegex, "null" };
        if (depth > 0)
        {
            string inner = JsonValueRegex(depth - 1);
            parts.Add(ObjectOfAnyRegex(inner));
            parts.Add(Escape("[") + "(?:" + "(?:" + inner + ")(?:" + Escape(",") + "(?:" + inner + "))*)?" + Escape("]"));
        }
        return "(?:" + string.Join("|", parts) + ")";
    }

    private static string ObjectOfAnyRegex(string valueRegex)
    {
        string member = JsonStringRegex + Escape(":") + "(?:" + valueRegex + ")";
        return Escape("{") + "(?:" + member + "(?:" + Escape(",") + member + ")*)?" + Escape("}");
    }

    private static string JsonLiteralRegex(JToken literal)
    {
        // Serialize the literal to its exact compact JSON form and escape it for the regex.
        string json = literal.ToString(Formatting.None);
        return EscapeLiteral(json);
    }

    private static string Alternation(IEnumerable<string> options) => "(?:" + string.Join("|", options) + ")";

    // Escapes a structural character for the regex (single-char metachars used by CompileObject/Array).
    private static string Escape(string s)
    {
        var sb = new StringBuilder(s.Length * 2);
        foreach (char c in s) sb.Append(EscapeChar(c));
        return sb.ToString();
    }

    // Escapes a literal string so it matches itself in the regex engine.
    private static string EscapeLiteral(string s)
    {
        var sb = new StringBuilder(s.Length * 2);
        foreach (char c in s) sb.Append(EscapeChar(c));
        return sb.ToString();
    }

    private static string EscapeChar(char c)
        => c is '\\' or '.' or '[' or ']' or '(' or ')' or '{' or '}' or '*' or '+' or '?' or '|' or '^' or '$'
            ? "\\" + c
            : c.ToString();
}
