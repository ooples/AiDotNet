using System.Runtime.CompilerServices;
using AiDotNet.Serving.StructuredOutput;
using AiDotNet.Tokenization.Interfaces;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Builds an <see cref="ITokenConstraint"/> from an OpenAI-compatible <c>response_format</c> so guided /
/// structured decoding can be requested per call. Supported forms:
/// <list type="bullet">
/// <item><description><c>{ "type": "text" }</c> — no constraint (default free-form).</description></item>
/// <item><description><c>{ "type": "json_object" }</c> — any valid (bounded-depth) compact JSON object.</description></item>
/// <item><description><c>{ "type": "json_schema", "json_schema": { "schema": { ... } } }</c> — JSON matching the schema.</description></item>
/// <item><description><c>{ "type": "regex", "regex": "..." }</c> — an AiDotNet extension: output matches the regex.</description></item>
/// </list>
/// The vocabulary's per-token text is derived once per tokenizer and cached, since it is reused by every
/// constrained request served with that tokenizer.
/// </summary>
internal static class StructuredOutputFactory
{
    // token-piece strings per tokenizer, built lazily and kept alive only while the tokenizer is.
    private static readonly ConditionalWeakTable<ITokenizer, string[]> _vocabCache = new();

    /// <summary>
    /// Builds a constraint for <paramref name="responseFormat"/>, or returns null when no constraint applies
    /// (null / <c>type:"text"</c>). Throws <see cref="ArgumentException"/> for a malformed response_format so
    /// the controller can surface a 400.
    /// </summary>
    /// <param name="responseFormat">The parsed <c>response_format</c> value, or null.</param>
    /// <param name="tokenizer">The tokenizer whose vocabulary the constraint masks over.</param>
    /// <param name="eosTokenId">The model's end-of-sequence token id, or a negative value if none.</param>
    public static ITokenConstraint? Build(JToken? responseFormat, ITokenizer tokenizer, int eosTokenId)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (responseFormat is null || responseFormat.Type == JTokenType.Null)
        {
            return null;
        }
        if (responseFormat is not JObject obj)
        {
            throw new ArgumentException("'response_format' must be an object.");
        }

        string type = obj["type"]?.ToString() ?? "text";
        switch (type)
        {
            case "text":
                return null;

            case "json_object":
            {
                // Unbounded-depth JSON via the pushdown-automaton grammar (not the finite bounded-depth regex).
                var vocab = GetVocabStrings(tokenizer);
                return new JsonGrammarConstraint(vocab, eosTokenId);
            }

            case "json_schema":
            {
                // OpenAI nests the schema under json_schema.schema; accept a bare schema too for convenience.
                JToken? schema = obj["json_schema"]?["schema"] ?? obj["schema"] ?? obj["json_schema"];
                if (schema is null)
                {
                    throw new ArgumentException("'response_format.json_schema.schema' is required for type 'json_schema'.");
                }
                var vocab = GetVocabStrings(tokenizer);
                return BuildOrThrow(
                    () => JsonSchemaConstraint.FromSchema(schema.ToString(Newtonsoft.Json.Formatting.None), vocab, eosTokenId),
                    "json_schema");
            }

            case "regex":
            {
                string? pattern = obj["regex"]?.ToString();
                if (string.IsNullOrEmpty(pattern))
                {
                    throw new ArgumentException("'response_format.regex' is required for type 'regex'.");
                }
                var vocab = GetVocabStrings(tokenizer);
                return BuildOrThrow(() => new RegexTokenConstraint(pattern!, vocab, eosTokenId), "regex");
            }

            default:
                throw new ArgumentException($"Unsupported response_format type '{type}'. " +
                    "Supported: text, json_object, json_schema, regex.");
        }
    }

    // Runs a constraint compiler, normalizing malformed-input parse failures to ArgumentException so the
    // controller surfaces a 400 (Build's documented contract). A regex or JSON schema is request-controlled
    // and can throw FormatException/OverflowException (and, from the regex quantifier cap, ArgumentException)
    // deep in its parser; without this the controller would surface a 500 instead of a client 400.
    private static ITokenConstraint BuildOrThrow(Func<ITokenConstraint> compile, string kind)
    {
        try
        {
            return compile();
        }
        catch (ArgumentException)
        {
            throw; // already the documented 400-mapped type; keep its message
        }
        catch (Exception ex) when (ex is FormatException or OverflowException or ArgumentOutOfRangeException or InvalidOperationException)
        {
            throw new ArgumentException($"Invalid '{kind}' response_format: {ex.Message}", ex);
        }
    }

    // Builds (and caches) token id -> decoded text piece for the whole vocabulary. Used to judge, per token,
    // what characters it would append to the output. Shared with ToolConstraintFactory.
    internal static string[] GetVocabStrings(ITokenizer tokenizer)
    {
        if (_vocabCache.TryGetValue(tokenizer, out var cached))
        {
            return cached;
        }

        int size = tokenizer.VocabularySize;
        var pieces = new string[size];
        var single = new List<int>(1) { 0 };
        for (int id = 0; id < size; id++)
        {
            single[0] = id;
            try
            {
                pieces[id] = tokenizer.Decode(single, skipSpecialTokens: false) ?? string.Empty;
            }
            catch (Exception ex) when (ex is ArgumentException or ArgumentOutOfRangeException or KeyNotFoundException or FormatException or InvalidOperationException or NotSupportedException or IndexOutOfRangeException)
            {
                // Some tokenizers cannot decode certain ids in isolation (byte-fallback fragments, special
                // tokens). Log the offending id and treat it as an empty, un-emittable piece so the constraint
                // simply never permits it — rather than silently swallowing EVERY exception, which would hide
                // real tokenizer defects. Unexpected exception types propagate.
                AiDotNet.Helpers.InferenceDiagnostics.RecordException(
                    area: "Serving.StructuredOutput",
                    feature: "VocabDecode",
                    ex: ex,
                    reason: $"Tokenizer failed to decode token id {id} in isolation; treating it as an un-emittable empty piece.");
                pieces[id] = string.Empty;
            }
        }

        _vocabCache.AddOrUpdate(tokenizer, pieces);
        return pieces;
    }
}
