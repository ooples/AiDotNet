using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Multi-query expansion generating multiple variations of the query
    /// </summary>
    public class MultiQueryExpansion : QueryExpansionBase
    {
        private readonly int _numVariations;
        private readonly Dictionary<string, List<string>> _synonymMap;

        public MultiQueryExpansion(int numVariations = 3)
        {
            _numVariations = numVariations;
            _synonymMap = new Dictionary<string, List<string>>();
        }

        public void AddSynonyms(string term, List<string> synonyms)
        {
            if (string.IsNullOrEmpty(term))
                throw new ArgumentException("Term cannot be null or empty", nameof(term));
            if (synonyms == null)
                throw new ArgumentNullException(nameof(synonyms));

            _synonymMap[term.ToLowerInvariant()] = synonyms;
        }

        protected override Task<List<string>> ExpandCoreAsync(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Task.FromResult(new List<string>());

            var variations = new List<string> { query };
            var queryTerms = query.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < _numVariations - 1; i++)
            {
                var variation = new List<string>();
                foreach (var term in queryTerms)
                {
                    var normalizedTerm = term.ToLowerInvariant();
                    if (_synonymMap.TryGetValue(normalizedTerm, out var synonyms) && synonyms.Count > 0)
                    {
                        var synonymIndex = i % synonyms.Count;
                        variation.Add(synonyms[synonymIndex]);
                    }
                    else
                    {
                        variation.Add(term);
                    }
                }
                variations.Add(string.Join(" ", variation));
            }

            variations.Add(GenerateReorderedQuery(queryTerms));
            variations.Add(GenerateExpandedQuery(queryTerms));

            return Task.FromResult(variations.Distinct().ToList());
        }

        private string GenerateReorderedQuery(string[] terms)
        {
            var reordered = terms.Reverse().ToArray();
            return string.Join(" ", reordered);
        }

        private string GenerateExpandedQuery(string[] terms)
        {
            var expanded = new List<string>(terms);
            foreach (var term in terms)
            {
                var normalizedTerm = term.ToLowerInvariant();
                if (_synonymMap.TryGetValue(normalizedTerm, out var synonyms) && synonyms.Count > 0)
                {
                    expanded.AddRange(synonyms.Take(2));
                }
            }
            return string.Join(" ", expanded);
        }
    }
}
