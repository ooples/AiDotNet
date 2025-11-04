using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    public class RecursiveCharacterChunkingStrategy : ChunkingStrategyBase
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;
        private readonly string[] _separators;

        public RecursiveCharacterChunkingStrategy(
            int chunkSize = 1000,
            int chunkOverlap = 200,
            string[]? separators = null)
        {
            _chunkSize = chunkSize;
            _chunkOverlap = chunkOverlap;
            _separators = separators ?? new[] { "\n\n", "\n", ". ", " ", "" };
        }

        protected override List<string> ChunkCore(string text)
        {
            var chunks = new List<string>();

            if (string.IsNullOrWhiteSpace(text))
            {
                return chunks;
            }

            var splits = RecursiveSplit(text, _separators, 0);
            
            // Merge small splits into chunks
            var currentChunk = "";
            foreach (var split in splits)
            {
                if (currentChunk.Length + split.Length <= _chunkSize)
                {
                    currentChunk += split;
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(currentChunk))
                    {
                        chunks.Add(currentChunk.Trim());
                    }
                    currentChunk = split;
                }
            }

            if (!string.IsNullOrWhiteSpace(currentChunk))
            {
                chunks.Add(currentChunk.Trim());
            }

            return chunks;
        }

        private List<string> RecursiveSplit(string text, string[] separators, int sepIndex)
        {
            if (sepIndex >= separators.Length || text.Length <= _chunkSize)
            {
                return new List<string> { text };
            }

            var separator = separators[sepIndex];
            var splits = text.Split(new[] { separator }, StringSplitOptions.None);
            var result = new List<string>();

            foreach (var split in splits)
            {
                if (split.Length > _chunkSize)
                {
                    result.AddRange(RecursiveSplit(split, separators, sepIndex + 1));
                }
                else if (!string.IsNullOrWhiteSpace(split))
                {
                    result.Add(split + separator);
                }
            }

            return result;
        }
    }
}
