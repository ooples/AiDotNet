using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Hypothetical Document Embeddings (HyDE) query expansion strategy.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HyDEQueryExpansion<T> : QueryExpansionBase
    {
        private readonly INumericOperations<T> _numOps;
        private readonly IEmbeddingModel<T> _embeddingModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="HyDEQueryExpansion{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="embeddingModel">The embedding model for generating hypothetical documents.</param>
        public HyDEQueryExpansion(INumericOperations<T> numericOperations, IEmbeddingModel<T> embeddingModel)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
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

        /// <summary>
        /// Generates a hypothetical document embedding for the query.
        /// </summary>
        /// <param name="query">The query to expand.</param>
        /// <returns>An embedding representing the hypothetical document.</returns>
        public Vector<T> GenerateHypotheticalEmbedding(string query)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));

            var hypotheticalDoc = GenerateHypotheticalDocument(query, "comprehensive answer");
            return _embeddingModel.Embed(hypotheticalDoc);
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
