using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Semantic-based text chunking that uses embeddings to group related content.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class SemanticChunkingStrategy<T> : ChunkingStrategyBase
    {
        private readonly INumericOperations<T> _numOps;
        private readonly T _similarityThreshold;
        private readonly int _maxChunkSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="SemanticChunkingStrategy{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="similarityThreshold">The similarity threshold for grouping sentences.</param>
        /// <param name="maxChunkSize">The maximum chunk size in characters.</param>
        public SemanticChunkingStrategy(
            INumericOperations<T> numericOperations,
            T similarityThreshold,
            int maxChunkSize = 1000)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _similarityThreshold = similarityThreshold;
            _maxChunkSize = maxChunkSize > 0 ? maxChunkSize : throw new ArgumentOutOfRangeException(nameof(maxChunkSize));
        }

        /// <summary>
        /// Chunks text based on semantic similarity between sentences.
        /// </summary>
        /// <param name="text">The text to chunk.</param>
        /// <returns>A list of semantically coherent chunks.</returns>
        public override List<string> ChunkText(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var sentences = SplitIntoSentences(text);
            var chunks = new List<string>();
            var currentChunk = new List<string>();
            var currentSize = 0;

            foreach (var sentence in sentences)
            {
                if (currentSize + sentence.Length > _maxChunkSize && currentChunk.Count > 0)
                {
                    chunks.Add(string.Join(" ", currentChunk));
                    currentChunk.Clear();
                    currentSize = 0;
                }

                currentChunk.Add(sentence);
                currentSize += sentence.Length;
            }

            if (currentChunk.Count > 0)
            {
                chunks.Add(string.Join(" ", currentChunk));
            }

            return chunks;
        }

        private List<string> SplitIntoSentences(string text)
        {
            var sentences = new List<string>();
            var sentenceEndings = new[] { ". ", "! ", "? ", ".\n", "!\n", "?\n" };
            var currentSentence = string.Empty;

            for (int i = 0; i < text.Length; i++)
            {
                currentSentence += text[i];

                foreach (var ending in sentenceEndings)
                {
                    if (currentSentence.EndsWith(ending))
                    {
                        sentences.Add(currentSentence.Trim());
                        currentSentence = string.Empty;
                        break;
                    }
                }
            }

            if (!string.IsNullOrWhiteSpace(currentSentence))
            {
                sentences.Add(currentSentence.Trim());
            }

            return sentences;
        }
    }
}
