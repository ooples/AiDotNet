using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Query expansion using learned sparse encoders
    /// </summary>
    public class LearnedSparseEncoderExpansion : QueryExpansionBase
    {
        private readonly Dictionary<string, double> _termWeights;

        public LearnedSparseEncoderExpansion()
        {
            _termWeights = new Dictionary<string, double>();
        }

        public void TrainWeights(Dictionary<string, double> termWeights)
        {
            if (termWeights == null)
                throw new ArgumentNullException(nameof(termWeights));

            _termWeights.Clear();
            foreach (var (term, weight) in termWeights)
            {
                _termWeights[term] = weight;
            }
        }

        protected override Task<List<string>> ExpandCoreAsync(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Task.FromResult(new List<string>());

            var queryTerms = query.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var expandedTerms = new Dictionary<string, double>();

            foreach (var term in queryTerms)
            {
                var normalizedTerm = term.ToLowerInvariant();
                expandedTerms[normalizedTerm] = 1.0;

                if (_termWeights.TryGetValue(normalizedTerm, out var weight))
                {
                    foreach (var (relatedTerm, relatedWeight) in _termWeights)
                    {
                        if (relatedTerm != normalizedTerm && relatedWeight > 0.1)
                        {
                            var combinedWeight = weight * relatedWeight;
                            if (!expandedTerms.ContainsKey(relatedTerm))
                            {
                                expandedTerms[relatedTerm] = combinedWeight;
                            }
                            else
                            {
                                expandedTerms[relatedTerm] = Math.Max(expandedTerms[relatedTerm], combinedWeight);
                            }
                        }
                    }
                }
            }

            var sortedTerms = expandedTerms
                .OrderByDescending(x => x.Value)
                .Take(20)
                .Select(x => x.Key)
                .ToList();

            var expandedQueries = new List<string> { query };
            if (sortedTerms.Count > queryTerms.Length)
            {
                expandedQueries.Add(string.Join(" ", sortedTerms));
            }

            return Task.FromResult(expandedQueries);
        }
    }
}
