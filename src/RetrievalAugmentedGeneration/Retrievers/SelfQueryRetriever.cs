using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Self-querying retriever: uses a text generator to split a natural-language query into a semantic
    /// search string plus structured metadata filters over known fields, then delegates to a base retriever
    /// with those filters merged in. Mirrors the LangChain/LlamaIndex self-query retriever. Without a
    /// generator it passes the query through unchanged (filters from the caller only).
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class SelfQueryRetriever<T> : RetrieverBase<T>
    {
        private readonly IRetriever<T> _baseRetriever;
        private readonly IReadOnlyList<string> _metadataFields;
        private readonly ITextGenerator? _generator;

        /// <param name="baseRetriever">The retriever the cleaned query + extracted filters are passed to.</param>
        /// <param name="metadataFields">The filterable metadata field names the LLM may use.</param>
        /// <param name="generator">Optional real text generator; when null the query passes through unparsed.</param>
        /// <param name="defaultTopK">Default number of documents to return.</param>
        public SelfQueryRetriever(
            IRetriever<T> baseRetriever,
            IReadOnlyList<string> metadataFields,
            ITextGenerator? generator = null,
            int defaultTopK = 5)
            : base(defaultTopK)
        {
            _baseRetriever = baseRetriever ?? throw new ArgumentNullException(nameof(baseRetriever));
            _metadataFields = metadataFields ?? new List<string>();
            _generator = generator;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            string searchQuery = query;
            var merged = new Dictionary<string, object>(metadataFilters ?? new Dictionary<string, object>());

            if (_generator != null && _metadataFields.Count > 0)
            {
                var (cleaned, extracted) = ParseQuery(query);
                if (!string.IsNullOrWhiteSpace(cleaned)) searchQuery = cleaned;
                foreach (var kv in extracted)
                {
                    // Caller-supplied filters take precedence over LLM-extracted ones.
                    if (!merged.ContainsKey(kv.Key)) merged[kv.Key] = kv.Value;
                }
            }

            return _baseRetriever.Retrieve(searchQuery, topK, merged);
        }

        private (string cleanedQuery, Dictionary<string, object> filters) ParseQuery(string query)
        {
            var filters = new Dictionary<string, object>();
            var fields = string.Join(", ", _metadataFields);
            var prompt =
                $"Rewrite the user query for semantic search and extract any metadata filters that map to these " +
                $"fields: {fields}. Reply with ONLY a JSON object of the form " +
                $"{{\"query\": \"...\", \"filters\": {{\"field\": value}}}}. Use only the listed fields; omit " +
                $"filters if none apply.\n\nUser query: {query}\n\nJSON:";
            var reply = _generator!.Generate(prompt);
            if (string.IsNullOrWhiteSpace(reply)) return (query, filters);

            try
            {
                var json = ExtractJsonObject(reply);
                if (json == null) return (query, filters);

                var obj = JObject.Parse(json);
                var cleaned = obj.Value<string>("query");
                var filtersToken = obj["filters"] as JObject;
                if (filtersToken != null)
                {
                    foreach (var prop in filtersToken.Properties())
                    {
                        // Only accept declared fields to avoid injecting arbitrary filter keys.
                        if (!_metadataFields.Contains(prop.Name)) continue;
                        var val = ToClrValue(prop.Value);
                        if (val != null) filters[prop.Name] = val;
                    }
                }

                return (string.IsNullOrWhiteSpace(cleaned) ? query : cleaned!, filters);
            }
            catch (Exception)
            {
                // Malformed LLM JSON: degrade gracefully to the raw query with no extracted filters.
                return (query, filters);
            }
        }

        // Pull the first {...} block out of a possibly chatty LLM reply.
        private static string? ExtractJsonObject(string text)
        {
            int start = text.IndexOf('{');
            int end = text.LastIndexOf('}');
            return (start >= 0 && end > start) ? text.Substring(start, end - start + 1) : null;
        }

        private static object? ToClrValue(JToken token)
        {
            switch (token.Type)
            {
                case JTokenType.Integer: return token.Value<long>();
                case JTokenType.Float: return token.Value<double>();
                case JTokenType.Boolean: return token.Value<bool>();
                case JTokenType.String: return token.Value<string>();
                case JTokenType.Array: return token.Select(ToClrValue).Where(v => v != null).ToList();
                default: return null;
            }
        }
    }
}
