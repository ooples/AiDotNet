using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Code-aware text splitter that respects code structure and syntax.
    /// </summary>
    public class CodeAwareTextSplitter : ChunkingStrategyBase
    {
        private readonly string _language;

        /// <summary>
        /// Initializes a new instance of the <see cref="CodeAwareTextSplitter"/> class.
        /// </summary>
        /// <param name="chunkSize">The maximum size of each chunk.</param>
        /// <param name="chunkOverlap">The overlap between consecutive chunks.</param>
        /// <param name="language">The programming language (e.g., "csharp", "python", "javascript").</param>
        public CodeAwareTextSplitter(int chunkSize = 1000, int chunkOverlap = 200, string language = "csharp")
            : base(chunkSize, chunkOverlap)
        {
            _language = language ?? throw new ArgumentNullException(nameof(language));
        }

        /// <summary>
        /// Chunks code while preserving code structure.
        /// </summary>
        /// <param name="text">The code text to chunk.</param>
        /// <returns>A collection of code chunks with positions.</returns>
        protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
        {
            var separators = GetLanguageSeparators(_language);
            var chunks = SplitTextRecursive(text, separators);

            var position = 0;
            foreach (var chunk in chunks)
            {
                var endPos = position + chunk.Length;
                yield return (chunk, position, endPos);
                position = endPos - ChunkOverlap;
            }
        }

        private string[] GetLanguageSeparators(string language)
        {
            switch (language.ToLowerInvariant())
            {
                case "csharp":
                case "c#":
                    return new[] { "\n    }\n", "\n}\n", "\n\n", "\n", " " };
                case "python":
                    return new[] { "\ndef ", "\nclass ", "\n\n", "\n", " " };
                case "javascript":
                case "typescript":
                    return new[] { "\n}\n", "\n\n", "\n", " " };
                default:
                    return new[] { "\n\n", "\n", " " };
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

                    // Handle splits larger than ChunkSize by breaking them down
                    if (currentChunk.Length == 0 && split.Length > ChunkSize)
                    {
                        var step = Math.Max(ChunkSize - ChunkOverlap, 1);
                        for (int i = 0; i < split.Length; i += step)
                        {
                            var length = Math.Min(ChunkSize, split.Length - i);
                            chunks.Add(split.Substring(i, length));
                        }
                        currentChunk.Clear();
                        continue;
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
