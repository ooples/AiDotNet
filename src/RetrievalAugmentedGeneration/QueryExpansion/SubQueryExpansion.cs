using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryExpansion
{
    /// <summary>
    /// Sub-query expansion that breaks complex queries into simpler sub-queries
    /// </summary>
    public class SubQueryExpansion : QueryExpansionBase
    {
        private readonly int _maxSubQueries;

        public SubQueryExpansion(int maxSubQueries = 3)
        {
            _maxSubQueries = maxSubQueries;
        }

        protected override Task<List<string>> ExpandCoreAsync(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return Task.FromResult(new List<string>());

            var subQueries = new List<string> { query };

            var andSplits = query.Split(new[] { " and ", " AND " }, StringSplitOptions.RemoveEmptyEntries);
            if (andSplits.Length > 1)
            {
                foreach (var split in andSplits.Take(_maxSubQueries))
                {
                    if (!string.IsNullOrWhiteSpace(split))
                    {
                        subQueries.Add(split.Trim());
                    }
                }
            }

            var orSplits = query.Split(new[] { " or ", " OR " }, StringSplitOptions.RemoveEmptyEntries);
            if (orSplits.Length > 1)
            {
                foreach (var split in orSplits.Take(_maxSubQueries))
                {
                    if (!string.IsNullOrWhiteSpace(split))
                    {
                        subQueries.Add(split.Trim());
                    }
                }
            }

            if (query.Contains("?"))
            {
                var questions = query.Split(new[] { '?' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var question in questions.Take(_maxSubQueries))
                {
                    var trimmedQuestion = question.Trim();
                    if (!string.IsNullOrWhiteSpace(trimmedQuestion))
                    {
                        subQueries.Add(trimmedQuestion + "?");
                    }
                }
            }

            var terms = query.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            if (terms.Length >= 4)
            {
                var midpoint = terms.Length / 2;
                var firstHalf = string.Join(" ", terms.Take(midpoint));
                var secondHalf = string.Join(" ", terms.Skip(midpoint));

                if (!string.IsNullOrWhiteSpace(firstHalf))
                    subQueries.Add(firstHalf);
                if (!string.IsNullOrWhiteSpace(secondHalf))
                    subQueries.Add(secondHalf);
            }

            return Task.FromResult(subQueries.Distinct().Take(_maxSubQueries + 1).ToList());
        }
    }
}
