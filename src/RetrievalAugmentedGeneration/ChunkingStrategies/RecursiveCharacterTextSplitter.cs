using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Recursive character-based text splitting that preserves semantic meaning.
    /// </summary>
    public class RecursiveCharacterTextSplitter : IChunkingStrategy
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;
        private readonly string[] _separators;

        /// <summary>
        /// Initializes a new instance of the <see cref="RecursiveCharacterTextSplitter"/> class.
        /// </summary>
        /// <param name="chunkSize">The maximum size of each chunk.</param>
        /// <param name="chunkOverlap">The overlap between consecutive chunks.</param>
        /// <param name="separators">The separators to use for splitting, in order of preference.</param>
        public RecursiveCharacterTextSplitter(
            int chunkSize = 1000,
            int chunkOverlap = 200,
            string[] separators = null)
        {
            _chunkSize = chunkSize > 0 ? chunkSize : throw new ArgumentOutOfRangeException(nameof(chunkSize));
            _chunkOverlap = chunkOverlap >= 0 ? chunkOverlap : throw new ArgumentOutOfRangeException(nameof(chunkOverlap));
            _separators = separators ?? new[] { "\n\n", "\n", ". ", " ", "" };
        }

        /// <summary>
        /// Splits the input text into chunks recursively.
        /// </summary>
        /// <param name="text">The text to split.</param>
        /// <returns>A list of text chunks.</returns>
        public List<string> ChunkText(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            return SplitTextRecursive(text, _separators);
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
                if (string.IsNullOrEmpty(separator))
                {
                    for (int i = 0; i < text.Length; i += _chunkSize - _chunkOverlap)
                    {
                        var length = Math.Min(_chunkSize, text.Length - i);
                        chunks.Add(text.Substring(i, length));
                    }
                    return chunks;
                }

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
