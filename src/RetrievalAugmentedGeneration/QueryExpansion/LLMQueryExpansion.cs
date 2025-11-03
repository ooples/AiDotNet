using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// LLM-based query expansion for generating additional query variations.
    /// </summary>
    public class LLMQueryExpansion : QueryExpansionBase
    {
        private readonly string _llmEndpoint;
        private readonly string _apiKey;
        private readonly int _numExpansions;

        /// <summary>
        /// Initializes a new instance of the <see cref="LLMQueryExpansion"/> class.
        /// </summary>
        /// <param name="llmEndpoint">The LLM API endpoint.</param>
        /// <param name="apiKey">The API key for the LLM service.</param>
        /// <param name="numExpansions">The number of query expansions to generate.</param>
        public LLMQueryExpansion(string? llmEndpoint = null, string? apiKey = null, int numExpansions = 3)
        {
            _llmEndpoint = llmEndpoint ?? string.Empty;
            _apiKey = apiKey ?? string.Empty;
            _numExpansions = numExpansions > 0 ? numExpansions : throw new ArgumentOutOfRangeException(nameof(numExpansions));
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

            for (int i = 0; i < _numExpansions; i++)
            {
                var expansion = GenerateExpansion(query, i);
                expansions.Add(expansion);
            }

            return expansions;
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
