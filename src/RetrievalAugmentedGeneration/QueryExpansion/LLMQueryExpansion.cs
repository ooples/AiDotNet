using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// LLM-based query expansion: rewrites a query into several alternative phrasings that capture the
    /// same information need. When a real text generator is supplied the variations are produced by the
    /// LLM; otherwise a deterministic template set is used as an offline fallback.
    /// </summary>
    [ComponentType(ComponentType.QueryExpander)]
    [PipelineStage(PipelineStage.QueryProcessing)]
    public class LLMQueryExpansion : QueryExpansionBase
    {
        private readonly string _llmEndpoint;
        private readonly string _apiKey;
        private readonly int _numExpansions;
        private readonly ITextGenerator? _generator;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMQueryExpansion"/> class.
        /// </summary>
        /// <param name="llmEndpoint">Optional LLM API endpoint (informational; the actual call goes through <paramref name="generator"/>).</param>
        /// <param name="apiKey">Optional API key (informational; the actual call goes through <paramref name="generator"/>).</param>
        /// <param name="numExpansions">The number of query expansions to generate.</param>
        /// <param name="generator">
        /// Optional real text generator (e.g. a <c>ChatClientGenerator</c>). When provided, variations are
        /// LLM-generated; when <c>null</c>, a deterministic template fallback is used so the component still
        /// functions offline.
        /// </param>
        public LLMQueryExpansion(string? llmEndpoint = null, string? apiKey = null, int numExpansions = 3, ITextGenerator? generator = null)
        {
            _llmEndpoint = llmEndpoint ?? string.Empty;
            _apiKey = apiKey ?? string.Empty;
            _numExpansions = numExpansions > 0 ? numExpansions : throw new ArgumentOutOfRangeException(nameof(numExpansions));
            _generator = generator;
        }

        /// <summary>
        /// Expands a query using LLM-generated variations.
        /// </summary>
        /// <param name="query">The original query.</param>
        /// <returns>A list of expanded queries including the original.</returns>
        public override List<string> ExpandQuery(string query)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));

            var expansions = new List<string> { query };

            if (_generator != null)
            {
                var prompt =
                    $"Generate {_numExpansions} alternative search queries that capture the same information " +
                    $"need as the query below. Put each on its own line, with no numbering, quotes, or commentary.\n\n" +
                    $"Query: {query}";
                var text = _generator.Generate(prompt) ?? string.Empty;
                foreach (var line in text.Split('\n'))
                {
                    var cleaned = CleanVariation(line);
                    if (cleaned.Length > 0 && !expansions.Contains(cleaned))
                    {
                        expansions.Add(cleaned);
                    }

                    if (expansions.Count > _numExpansions) break;
                }
            }

            // No generator, or the LLM returned nothing usable: guarantee variations via the template.
            if (expansions.Count == 1)
            {
                for (int i = 0; i < _numExpansions; i++)
                {
                    expansions.Add(GenerateExpansion(query, i));
                }
            }

            return expansions;
        }

        // Strip common list decorations (numbering, bullets, surrounding quotes) from an LLM line.
        private static string CleanVariation(string line)
        {
            var s = line.Trim();
            if (s.Length == 0) return string.Empty;
            s = s.TrimStart('-', '*', '•', ' ', '\t');
            int dot = s.IndexOf(". ", StringComparison.Ordinal);
            if (dot > 0 && dot <= 3 && s.Take(dot).All(char.IsDigit)) s = s.Substring(dot + 2);
            return s.Trim().Trim('"').Trim();
        }

        private string GenerateExpansion(string query, int variant)
        {
            switch (variant % 5)
            {
                case 0:
                    return $"What are the details of {query}?";
                case 1:
                    return $"Explain {query} in detail";
                case 2:
                    return $"Tell me about {query}";
                case 3:
                    return $"Information regarding {query}";
                case 4:
                    return $"{query} explained";
                default:
                    return query;
            }
        }
    }
}
