using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Hypothetical Document Embeddings (HyDE) query expansion (Gao et al., 2022). Generates hypothetical
    /// answer passages for a query so that a dense retriever matches document-shaped text rather than the
    /// short question. When a real text generator is supplied, the passages are written by the LLM (true
    /// HyDE); otherwise a deterministic template is used as an offline fallback.
    /// </summary>
    [ComponentType(ComponentType.QueryExpander)]
    [PipelineStage(PipelineStage.QueryProcessing)]
    public class HyDEQueryExpansion : QueryExpansionBase
    {
        private readonly ITextGenerator? _generator;
        private readonly int _numHypotheticals;

        // LLM prompt style descriptors (index-aligned with the offline template styles below).
        private static readonly string[] LlmStyles =
        {
            "a detailed, factual passage",
            "a concise explanation",
            "a technical description"
        };
        private static readonly string[] TemplateStyles =
        {
            "detailed answer",
            "concise explanation",
            "technical description"
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="HyDEQueryExpansion"/> class.
        /// </summary>
        /// <param name="generator">
        /// Optional real text generator (e.g. a <c>ChatClientGenerator</c> over any chat connector). When
        /// provided, hypothetical documents are LLM-generated (real HyDE). When <c>null</c>, a deterministic
        /// template fallback is used so the component still functions offline.
        /// </param>
        /// <param name="numHypotheticals">How many hypothetical passages to generate (default 3).</param>
        public HyDEQueryExpansion(ITextGenerator? generator = null, int numHypotheticals = 3)
        {
            _generator = generator;
            _numHypotheticals = Math.Max(1, numHypotheticals);
        }

        /// <summary>
        /// Expands a query by generating hypothetical documents.
        /// </summary>
        /// <param name="query">The original query.</param>
        /// <returns>The original query followed by hypothetical document variations.</returns>
        public override List<string> ExpandQuery(string query)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));

            var expansions = new List<string> { query };
            for (int i = 0; i < _numHypotheticals; i++)
            {
                expansions.Add(_generator != null
                    ? GenerateWithLlm(query, LlmStyles[i % LlmStyles.Length])
                    : GenerateHypotheticalDocument(query, TemplateStyles[i % TemplateStyles.Length]));
            }

            return expansions;
        }

        private string GenerateWithLlm(string query, string style)
        {
            var prompt =
                $"Write {style} that directly answers the following question, as if it were an excerpt from a " +
                $"relevant document. Return only the passage, with no preamble or commentary.\n\n" +
                $"Question: {query}\n\nPassage:";
            var doc = _generator!.Generate(prompt);
            return string.IsNullOrWhiteSpace(doc) ? query : doc.Trim();
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

