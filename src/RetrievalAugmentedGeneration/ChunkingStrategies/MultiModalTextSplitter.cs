using AiDotNet.Interfaces;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    public class MultiModalTextSplitter : ChunkingStrategyBase
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;

        public MultiModalTextSplitter(int chunkSize = 1000, int chunkOverlap = 200)
        {
            _chunkSize = chunkSize;
            _chunkOverlap = chunkOverlap;
        }

        protected override List<string> ChunkCore(string text)
        {
            var chunks = new List<string>();

            if (string.IsNullOrWhiteSpace(text))
            {
                return chunks;
            }

            // Extract code blocks
            var codeBlockPattern = @"```[\s\S]*?```";
            var codeBlocks = Regex.Matches(text, codeBlockPattern);
            var codeBlockIndices = new List<Tuple<int, int>>();

            foreach (Match match in codeBlocks)
            {
                codeBlockIndices.Add(Tuple.Create(match.Index, match.Index + match.Length));
                chunks.Add(match.Value);
            }

            // Extract tables
            var tablePattern = @"\|[^\n]+\|\n\|[-:\s|]+\|(?:\n\|[^\n]+\|)*";
            var tables = Regex.Matches(text, tablePattern);

            foreach (Match match in tables)
            {
                var index = match.Index;
                var endIndex = match.Index + match.Length;
                
                // Check if this table is not inside a code block
                var inCodeBlock = codeBlockIndices.Any(cb => index >= cb.Item1 && endIndex <= cb.Item2);
                
                if (!inCodeBlock)
                {
                    chunks.Add(match.Value);
                }
            }

            // Process remaining text
            var processedText = Regex.Replace(text, codeBlockPattern, "");
            processedText = Regex.Replace(processedText, tablePattern, "");

            var paragraphs = processedText.Split(new[] { "\n\n" }, StringSplitOptions.RemoveEmptyEntries);
            
            var currentChunk = "";
            foreach (var paragraph in paragraphs)
            {
                if (currentChunk.Length + paragraph.Length <= _chunkSize)
                {
                    currentChunk += paragraph + "\n\n";
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(currentChunk))
                    {
                        chunks.Add(currentChunk.Trim());
                    }
                    currentChunk = paragraph + "\n\n";
                }
            }

            if (!string.IsNullOrWhiteSpace(currentChunk))
            {
                chunks.Add(currentChunk.Trim());
            }

            return chunks;
        }
    }
}
