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
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;
        private readonly string _language;

        /// <summary>
        /// Initializes a new instance of the <see cref="CodeAwareTextSplitter"/> class.
        /// </summary>
        /// <param name="chunkSize">The maximum size of each chunk.</param>
        /// <param name="chunkOverlap">The overlap between consecutive chunks.</param>
        /// <param name="language">The programming language (e.g., "csharp", "python", "javascript").</param>
        public CodeAwareTextSplitter(int chunkSize = 1000, int chunkOverlap = 200, string language = "csharp")
        {
            _chunkSize = chunkSize > 0 ? chunkSize : throw new ArgumentOutOfRangeException(nameof(chunkSize));
            _chunkOverlap = chunkOverlap >= 0 ? chunkOverlap : throw new ArgumentOutOfRangeException(nameof(chunkOverlap));
            _language = language ?? throw new ArgumentNullException(nameof(language));
        }

        /// <summary>
        /// Chunks code while preserving code structure.
        /// </summary>
        /// <param name="text">The code text to chunk.</param>
        /// <returns>A list of code chunks.</returns>
        public override List<string> ChunkText(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var separators = GetLanguageSeparators(_language);
            return SplitTextRecursive(text, separators);
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
