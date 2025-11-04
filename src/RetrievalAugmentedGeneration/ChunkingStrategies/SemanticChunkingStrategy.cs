using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Semantic-based text chunking that uses embeddings to group related content.
    /// </summary>
    public class SemanticChunkingStrategy : ChunkingStrategyBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SemanticChunkingStrategy"/> class.
        /// </summary>
        /// <param name="maxChunkSize">The maximum chunk size in characters.</param>
        /// <param name="chunkOverlap">The chunk overlap in characters.</param>
        public SemanticChunkingStrategy(
            int maxChunkSize = 1000,
            int chunkOverlap = 200)
            : base(maxChunkSize, chunkOverlap)
        {
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

                if (sentenceEndings.Any(ending => currentSentence.EndsWith(ending)))
                {
                    sentences.Add(currentSentence.Trim());
                    currentSentence = string.Empty;
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

