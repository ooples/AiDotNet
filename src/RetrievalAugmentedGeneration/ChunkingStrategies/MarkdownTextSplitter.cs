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
        /// <summary>
        /// Initializes a new instance of the <see cref="MarkdownTextSplitter"/> class.
        /// </summary>
        /// <param name="chunkSize">The maximum size of each chunk.</param>
        /// <param name="chunkOverlap">The overlap between consecutive chunks.</param>
        public MarkdownTextSplitter(int chunkSize = 1000, int chunkOverlap = 200)
            : base(chunkSize, chunkOverlap)
        {
        }

        /// <summary>
        /// Chunks markdown text while preserving structure.
        /// </summary>
        /// <param name="text">The markdown text to chunk.</param>
        /// <returns>A collection of markdown chunks with positions.</returns>
        protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
        {
            var separators = new[] { "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " " };
            var chunks = SplitTextRecursive(text, separators);

            var position = 0;
            foreach (var chunk in chunks)
            {
                var endPos = position + chunk.Length;
                yield return (chunk, position, endPos);
                position = endPos - ChunkOverlap;
            }
        }

        private List<string> SplitTextRecursive(string text, string[] separators)
        {
            var chunks = new List<string>();

            if (text.Length <= ChunkSize)
            {
                chunks.Add(text);
                return chunks;
            }


            var currentChunk = new StringBuilder();

            foreach (var separator in separators)
            {
                var splits = text.Split(new[] { separator }, StringSplitOptions.None);
                currentChunk.Clear();

                foreach (var split in splits)
                {
                    if (currentChunk.Length + split.Length + separator.Length > ChunkSize)
                    {
                        if (currentChunk.Length > 0)
                        {
                            chunks.Add(currentChunk.ToString());
                            var overlap = Math.Min(ChunkOverlap, currentChunk.Length);
                            currentChunk = new StringBuilder(currentChunk.ToString(currentChunk.Length - overlap, overlap));
                        }
                    }

                    if (currentChunk.Length > 0)
                        currentChunk.Append(separator);

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
