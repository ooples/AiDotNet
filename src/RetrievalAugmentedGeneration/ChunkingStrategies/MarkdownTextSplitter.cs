using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Markdown-aware text splitter that respects markdown structure.
    /// </summary>
    public class MarkdownTextSplitter : ChunkingStrategyBase
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;

        /// <summary>
        /// Initializes a new instance of the <see cref="MarkdownTextSplitter"/> class.
        /// </summary>
        /// <param name="chunkSize">The maximum size of each chunk.</param>
        /// <param name="chunkOverlap">The overlap between consecutive chunks.</param>
        public MarkdownTextSplitter(int chunkSize = 1000, int chunkOverlap = 200)
        {
            _chunkSize = chunkSize > 0 ? chunkSize : throw new ArgumentOutOfRangeException(nameof(chunkSize));
            _chunkOverlap = chunkOverlap >= 0 ? chunkOverlap : throw new ArgumentOutOfRangeException(nameof(chunkOverlap));
        }

        /// <summary>
        /// Chunks markdown text while preserving structure.
        /// </summary>
        /// <param name="text">The markdown text to chunk.</param>
        /// <returns>A list of markdown chunks.</returns>
        public override List<string> ChunkText(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var separators = new[] { "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " " };
            return SplitTextRecursive(text, separators);
        }

        private List<string> SplitTextRecursive(string text, string[] separators)
        {
            var chunks = new List<string>();

            if (text.Length <= _chunkSize)
            {
                chunks.Add(text);
                return chunks;
            }

            foreach (var separator in separators)
            {
                var splits = text.Split(new[] { separator }, StringSplitOptions.None);
                var currentChunk = new StringBuilder();

                foreach (var split in splits)
                {
                    if (currentChunk.Length + split.Length + separator.Length > _chunkSize)
                    {
                        if (currentChunk.Length > 0)
                        {
                            chunks.Add(currentChunk.ToString());
                            var overlap = Math.Min(_chunkOverlap, currentChunk.Length);
                            currentChunk = new StringBuilder(currentChunk.ToString(currentChunk.Length - overlap, overlap));
                        }
                    }

                    if (currentChunk.Length > 0)
                    {
                        currentChunk.Append(separator);
                    }
                    currentChunk.Append(split);
                }

                if (currentChunk.Length > 0)
                {
                    chunks.Add(currentChunk.ToString());
                }

                if (chunks.Count > 1)
                {
                    return chunks;
                }
            }

            return chunks;
        }
    }
}
