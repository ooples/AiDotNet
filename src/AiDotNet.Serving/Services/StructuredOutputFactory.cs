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
                var vocab = GetVocabStrings(tokenizer);
                return JsonSchemaConstraint.AnyJsonObject(vocab, eosTokenId);
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
                return JsonSchemaConstraint.FromSchema(schema.ToString(Newtonsoft.Json.Formatting.None), vocab, eosTokenId);
            }

            case "regex":
            {
                string? pattern = obj["regex"]?.ToString();
                if (string.IsNullOrEmpty(pattern))
                {
                    throw new ArgumentException("'response_format.regex' is required for type 'regex'.");
                }
                var vocab = GetVocabStrings(tokenizer);
                return new RegexTokenConstraint(pattern!, vocab, eosTokenId);
            }

            default:
                throw new ArgumentException($"Unsupported response_format type '{type}'. " +
                    "Supported: text, json_object, json_schema, regex.");
        }
    }

    // Builds (and caches) token id -> decoded text piece for the whole vocabulary. Used to judge, per token,
    // what characters it would append to the output.
    private static string[] GetVocabStrings(ITokenizer tokenizer)
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
            catch (Exception)
            {
                // A tokenizer may reject decoding certain ids in isolation; treat as an empty piece so the
                // constraint simply never permits that token rather than failing the whole request.
                pieces[id] = string.Empty;
            }
        }

        _vocabCache.AddOrUpdate(tokenizer, pieces);
        return pieces;
    }
}
