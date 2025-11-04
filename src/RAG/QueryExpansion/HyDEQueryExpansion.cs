using AiDotNet.Interfaces;
using System.Text;

namespace AiDotNet.RAG.QueryExpansion
{
    public class HyDEQueryExpansion<T> : QueryExpansionBase<T>
    {
        private readonly string _model;
        private readonly int _numHypotheticalDocs;

        public HyDEQueryExpansion(string model = "gpt-3.5-turbo", int numHypotheticalDocs = 3)
        {
            _model = model;
            _numHypotheticalDocs = numHypotheticalDocs;
        }

        protected override List<string> ExpandCore(string query, Dictionary<string, object>? options = null)
        {
            var expandedQueries = new List<string> { query };
            
            // Generate hypothetical documents
            for (int i = 0; i < _numHypotheticalDocs; i++)
            {
                var hypotheticalDoc = GenerateHypotheticalDocument(query, i);
                expandedQueries.Add(hypotheticalDoc);
            }
            
            return expandedQueries;
        }

        private string GenerateHypotheticalDocument(string query, int variation)
        {
            var sb = new StringBuilder();
            
            // Generate a hypothetical answer based on the query
            if (query.ToLower().StartsWith("what is"))
            {
                var topic = query.Substring(7).Trim('?', ' ');
                sb.AppendLine($"{topic} is a concept that involves several key aspects.");
                sb.AppendLine($"The fundamental principles of {topic} include various components and features.");
                sb.AppendLine($"Understanding {topic} requires knowledge of its applications and implications.");
            }
            else if (query.ToLower().StartsWith("how to") || query.ToLower().StartsWith("how do"))
            {
                var task = query.Substring(query.IndexOf("to") + 2).Trim('?', ' ');
                sb.AppendLine($"To {task}, follow these general steps:");
                sb.AppendLine($"1. Prepare the necessary components and understand the requirements.");
                sb.AppendLine($"2. Execute the process systematically following best practices.");
                sb.AppendLine($"3. Verify the results and make adjustments as needed.");
            }
            else
            {
                sb.AppendLine($"Regarding the question about {query.Trim('?', ' ')}:");
                sb.AppendLine("This topic encompasses several important considerations and factors.");
                sb.AppendLine("Various approaches and methodologies can be applied to address this matter.");
                sb.AppendLine("Research and practical applications have shown different perspectives on this subject.");
            }
            
            // Add variation-specific details
            if (variation > 0)
            {
                sb.AppendLine($"Additionally, from perspective {variation + 1}:");
                sb.AppendLine("Alternative viewpoints and methodologies offer different insights.");
            }
            
            return sb.ToString().Trim();
        }
    }
}
