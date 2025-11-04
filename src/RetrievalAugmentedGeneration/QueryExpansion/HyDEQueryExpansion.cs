using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Hypothetical Document Embeddings (HyDE) query expansion strategy.
    /// </summary>
    public class HyDEQueryExpansion : QueryExpansionBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="HyDEQueryExpansion"/> class.
        /// </summary>
        public HyDEQueryExpansion()
        {
        }

        /// <summary>
        /// Expands a query by generating hypothetical documents.
        /// </summary>
        /// <param name="query">The original query.</param>
        /// <returns>A list of hypothetical document variations.</returns>
        public override List<string> ExpandQuery(string query)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));

            var expansions = new List<string> { query };

            var hypoDoc1 = GenerateHypotheticalDocument(query, "detailed answer");
            var hypoDoc2 = GenerateHypotheticalDocument(query, "concise explanation");
            var hypoDoc3 = GenerateHypotheticalDocument(query, "technical description");

            expansions.Add(hypoDoc1);
            expansions.Add(hypoDoc2);
            expansions.Add(hypoDoc3);

            return expansions;
        }

        private string GenerateHypotheticalDocument(string query, string style)
        {
            var words = query.Split(' ').Where(w => !string.IsNullOrWhiteSpace(w)).ToList();

            switch (style)
            {
                case "detailed answer":
                    return $"A comprehensive analysis of {query} reveals that it encompasses multiple aspects. " +
                           $"The key components include {string.Join(", ", words)}. " +
                           $"This topic is important because it addresses fundamental concepts.";

                case "concise explanation":
                    return $"{query} refers to {string.Join(" and ", words.Take(3))}. " +
                           $"It is characterized by its practical applications.";

                case "technical description":
                    return $"From a technical perspective, {query} involves {string.Join(", ", words)}. " +
                           $"The implementation requires careful consideration of these elements.";

                default:
                    return $"{query} is an important concept that involves {string.Join(", ", words)}. " +
                           $"Understanding this topic requires examination of its core principles.";
            }
        }
    }
}

