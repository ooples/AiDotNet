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

        /// <summary>
        /// Initializes a new instance of the <see cref="SemanticChunkingStrategy{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="similarityThreshold">The similarity threshold for grouping sentences.</param>
        /// <param name="maxChunkSize">The maximum chunk size in characters.</param>
        public SemanticChunkingStrategy(
            INumericOperations<T> numericOperations,
            T similarityThreshold,
            int maxChunkSize = 1000,
            int chunkOverlap = 200)
            : base(maxChunkSize, chunkOverlap)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _similarityThreshold = similarityThreshold;
        }

        /// <summary>
        /// Chunks text based on semantic similarity between sentences.
        /// </summary>
        /// <param name="text">The text to chunk.</param>
        /// <returns>A collection of semantically coherent chunks with positions.</returns>
        protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
        {
            var sentences = SplitIntoSentences(text);
            var currentChunk = new List<string>();
            var currentSize = 0;
            var position = 0;

            foreach (var sentence in sentences)
            {
                if (currentSize + sentence.Length > ChunkSize && currentChunk.Count > 0)
                {
                    var chunkText = string.Join(" ", currentChunk);
                    var endPos = position + chunkText.Length;
                    yield return (chunkText, position, endPos);
                    
                    position = endPos - ChunkOverlap;
                    currentChunk.Clear();
                    currentSize = 0;
                }

                currentChunk.Add(sentence);
                currentSize += sentence.Length;
            }

            if (currentChunk.Count > 0)
            {
                var chunkText = string.Join(" ", currentChunk);
                yield return (chunkText, position, position + chunkText.Length);
            }
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
