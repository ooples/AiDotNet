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

    // Keywords that carry no constraint this compiler must enforce (pure annotations); always allowed.
    private static readonly HashSet<string> MetadataKeywords = new(StringComparer.Ordinal)
    {
        "type", "title", "description", "$schema", "$id", "$comment", "default", "examples", "enum", "const"
    };

    /// <summary>Compiles a JSON Schema token to a regex matching its compact JSON serialization.</summary>
    /// <remarks>
    /// Rejects — rather than silently ignores — any schema construct it cannot enforce, so generated output
    /// can never violate a keyword the caller supplied. In particular a boolean <c>false</c> schema (permits
    /// no value) and unsupported constraint keywords (numeric <c>minimum</c>/<c>maximum</c>/<c>multipleOf</c>,
    /// <c>uniqueItems</c>, <c>$ref</c>, <c>additionalProperties</c>, etc.) throw <see cref="ArgumentException"/>.
    /// </remarks>
    public static string CompileToRegex(JToken schema)
    {
        Guard.NotNull(schema);

        // Boolean schemas: `true` accepts any value; `false` permits NONE (no output can satisfy it), so it is
        // rejected explicitly rather than silently treated as "any JSON" (which would be the opposite meaning).
        if (schema.Type == JTokenType.Boolean)
        {
            if (schema.Value<bool>()) return JsonValueRegex(3);
            throw new ArgumentException(
                "A `false` JSON Schema permits no value; no generated output can satisfy it.", nameof(schema));
        }

        if (schema is not JObject obj)
        {
            // A bare `{}` schema accepts anything.
            return JsonValueRegex(3);
        }

        // const is the single-value form of enum — the tightest possible constraint.
        if (obj["const"] is JToken constVal)
        {
            return JsonLiteralRegex(constVal);
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
            case "integer":
            case "number":
                // Numeric range/step keywords (minimum, maximum, exclusiveMinimum, exclusiveMaximum,
                // multipleOf) cannot be enforced by a finite regex over the decimal encoding, so rather than
                // emit output that silently violates them, reject any schema that sets them.
                EnsureNoUnsupportedKeywords(obj, type!);
                return type == "integer" ? IntegerRegex : NumberRegex;
            case "boolean":
                EnsureNoUnsupportedKeywords(obj);
                return BooleanRegex;
            case "null":
                EnsureNoUnsupportedKeywords(obj);
                return "null";
            case null: return JsonValueRegex(3); // untyped -> any value
            default:
                throw new ArgumentException($"Unsupported JSON Schema type '{type}'.", nameof(schema));
        }
    }

    // Throws if <paramref name="obj"/> carries any keyword this compiler does not enforce for the given type
    // (metadata keywords and the ones named in <paramref name="supported"/> are allowed). This is what keeps
    // "silently ignored keyword -> schema-violating output" from happening.
    private static void EnsureNoUnsupportedKeywords(JObject obj, params string[] supported)
    {
        foreach (var prop in obj.Properties())
        {
            if (MetadataKeywords.Contains(prop.Name)) continue;
            if (Array.IndexOf(supported, prop.Name) >= 0) continue;
            throw new ArgumentException(
                $"JSON Schema keyword '{prop.Name}' is not supported by the guided-decoding compiler and cannot " +
                "be silently ignored (it would allow schema-violating output). Remove it or use a schema this " +
                "compiler can enforce.", nameof(obj));
        }
    }

    private static string CompileObject(JObject obj)
    {
        EnsureNoUnsupportedKeywords(obj, "object", "properties", "required", "additionalProperties");

        // Our output emits EXACTLY the declared properties, so additionalProperties:false is already honored;
        // additionalProperties:true (or a schema allowing extras) cannot be, so reject it.
        if (obj["additionalProperties"] is JToken ap &&
            !(ap.Type == JTokenType.Boolean && ap.Value<bool>() == false))
        {
            throw new ArgumentException(
                "JSON Schema 'additionalProperties' other than false is not supported by the guided-decoding " +
                "compiler (it emits exactly the declared properties).", nameof(obj));
        }

        var props = obj["properties"] as JObject;
        if (props is null || props.Count == 0)
        {
            // Object with no declared properties: match any compact object.
            return ObjectOfAnyRegex(JsonValueRegex(3));
        }

        // This compiler forces EVERY declared property (in declared order); it cannot express optional
        // properties without exponential key-permutation blow-up. So a `required` list, if present, must name
        // exactly the declared properties — otherwise reject rather than under/over-constrain the output.
        if (obj["required"] is JArray req)
        {
            var declared = new HashSet<string>(props.Properties().Select(p => p.Name), StringComparer.Ordinal);
            var required = new HashSet<string>(req.Select(t => t.ToString()), StringComparer.Ordinal);
            if (!declared.SetEquals(required))
            {
                throw new ArgumentException(
                    "The guided-decoding compiler emits every declared property as required; a 'required' list " +
                    "that does not cover exactly the declared properties (i.e. optional properties) is not " +
                    "supported.", nameof(obj));
            }
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
        EnsureNoUnsupportedKeywords(obj, "array", "items", "minItems", "maxItems");
        string itemRegex = obj["items"] is JToken items ? CompileToRegex(items) : JsonValueRegex(3);
        string item = "(?:" + itemRegex + ")";
        string comma = Escape(",");

        int min = obj["minItems"] is JToken miTok ? miTok.Value<int>() : 0;
        int? max = obj["maxItems"] is JToken mxTok ? mxTok.Value<int>() : (int?)null;
        if (min < 0) throw new ArgumentException("JSON Schema 'minItems' must be >= 0.", nameof(obj));
        if (max is { } mm && mm < min) throw new ArgumentException("JSON Schema 'maxItems' must be >= 'minItems'.", nameof(obj));

        // [] or [ item (,item)* ] with the item count bounded to [min, max].
        string body;
        if (min == 0 && max is null)
        {
            body = "(?:" + item + "(?:" + comma + item + ")*)?"; // 0..unbounded
        }
        else if (max is 0)
        {
            body = ""; // empty array only
        }
        else
        {
            // A non-empty run of `item` separated by commas: first item, then (min-1 .. max-1) more.
            int repMin = Math.Max(0, min - 1);
            string repMax = max is { } m ? (m - 1).ToString(System.Globalization.CultureInfo.InvariantCulture) : "";
            string nonEmpty = item + "(?:" + comma + item + "){" + repMin.ToString(System.Globalization.CultureInfo.InvariantCulture) + "," + repMax + "}";
            body = min == 0 ? "(?:" + nonEmpty + ")?" : nonEmpty;
        }
        return Escape("[") + body + Escape("]");
    }

    private static string CompileString(JObject obj)
    {
        if (obj["enum"] is JArray e && e.Count > 0)
        {
            return Alternation(e.Select(JsonLiteralRegex));
        }
        if (obj["pattern"]?.ToString() is { Length: > 0 } pattern)
        {
            EnsureNoUnsupportedKeywords(obj, "string", "pattern");
            // Embed the user pattern inside JSON quotes. The pattern constrains the string CONTENT.
            return "\"(?:" + pattern + ")\"";
        }
        // minLength/maxLength/format are not enforced by this compiler; reject them rather than silently emit
        // strings that violate them (callers can express bounds via `pattern` instead).
        EnsureNoUnsupportedKeywords(obj, "string", "pattern");
        return JsonStringRegex;
    }

    // ---- primitive JSON regexes (compact form) ----
    // A JSON string per RFC 8259: unescaped chars are anything except '"', '\', and control chars
    // U+0000-U+001F; escapes are exactly \" \\ \/ \b \f \n \r \t or \uXXXX (4 hex). The previous
    // "[^"\\]|\\." wrongly accepted raw control characters and invalid escapes like \q. The U+0000 and
    // U+001F literals below are the class range endpoints (this engine has no \u escape).
    private const string JsonStringRegex = "\"(?:[^\"\\\\\u0000-\u001f]|\\\\(?:[\"\\\\/bfnrt]|u[0-9a-fA-F]{4}))*\"";
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
