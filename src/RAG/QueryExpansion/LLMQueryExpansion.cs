using AiDotNet.Interfaces;

namespace AiDotNet.RAG.QueryExpansion
{
    public class LLMQueryExpansion<T> : QueryExpansionBase<T>
        where T : struct, IComparable, IConvertible, IFormattable
    {
        private readonly string _model;
        private readonly int _numVariations;

        public LLMQueryExpansion(string model = "gpt-3.5-turbo", int numVariations = 3)
        {
            _model = model;
            _numVariations = numVariations;
        }

        protected override List<string> ExpandCore(string query, Dictionary<string, object>? options = null)
        {
            var expandedQueries = new List<string> { query };
            
            // Generate paraphrases
            for (int i = 0; i < _numVariations; i++)
            {
                var paraphrase = GenerateParaphrase(query, i);
                expandedQueries.Add(paraphrase);
            }
            
            // Add synonyms
            var synonymExpansion = ExpandWithSynonyms(query);
            expandedQueries.Add(synonymExpansion);
            
            return expandedQueries.Distinct().ToList();
        }

        private string GenerateParaphrase(string query, int variation)
        {
            var words = query.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            
            var synonyms = new Dictionary<string, string[]>
            {
                { "find", new[] { "locate", "search for", "discover" } },
                { "show", new[] { "display", "present", "reveal" } },
                { "get", new[] { "retrieve", "obtain", "fetch" } },
                { "how", new[] { "in what way", "by what means", "what method" } },
                { "what", new[] { "which", "that which", "the thing that" } }
            };
            
            var paraphrased = words.Select(word =>
            {
                var lowerWord = word.ToLower();
                if (synonyms.ContainsKey(lowerWord) && variation < synonyms[lowerWord].Length)
                {
                    return synonyms[lowerWord][variation];
                }
                return word;
            });
            
            return string.Join(" ", paraphrased);
        }

        private string ExpandWithSynonyms(string query)
        {
            var words = query.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var expanded = new List<string>(words);
            
            var commonSynonyms = new Dictionary<string, string[]>
            {
                { "fast", new[] { "quick", "rapid", "swift" } },
                { "slow", new[] { "gradual", "leisurely" } },
                { "big", new[] { "large", "huge", "massive" } },
                { "small", new[] { "tiny", "little", "compact" } }
            };
            
            foreach (var word in words)
            {
                var lowerWord = word.ToLower();
                if (commonSynonyms.ContainsKey(lowerWord))
                {
                    expanded.AddRange(commonSynonyms[lowerWord]);
                }
            }
            
            return string.Join(" ", expanded.Distinct());
        }
    }
}
