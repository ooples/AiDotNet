using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Sentence-based chunking strategy that preserves sentence boundaries
    /// </summary>
    public class SentenceChunkingStrategy : ChunkingStrategyBase
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;
        private static readonly Regex SentencePattern = new Regex(
            @"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?][""])\s+(?=[A-Z])",
            RegexOptions.Compiled);

        public SentenceChunkingStrategy(int chunkSize = 1000, int chunkOverlap = 200)
        {
            _chunkSize = chunkSize;
            _chunkOverlap = chunkOverlap;
        }

        protected override List<TextChunk> SplitCore(string text, Dictionary<string, string>? metadata = null)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<TextChunk>();

            var sentences = SplitIntoSentences(text);
            var chunks = new List<TextChunk>();
            var currentChunk = new List<string>();
            var currentLength = 0;
            var chunkIndex = 0;

            foreach (var sentence in sentences)
            {
                var sentenceLength = sentence.Length + 1;

                if (currentLength + sentenceLength > _chunkSize && currentChunk.Count > 0)
                {
                    var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                    {
                        ["chunk_index"] = chunkIndex.ToString(),
                        ["sentence_count"] = currentChunk.Count.ToString()
                    };

                    chunks.Add(new TextChunk
                    {
                        Text = string.Join(" ", currentChunk),
                        Metadata = chunkMetadata
                    });
                    chunkIndex++;

                    var overlapSentences = new List<string>();
                    var overlapLength = 0;

                    for (int i = currentChunk.Count - 1; i >= 0 && overlapLength < _chunkOverlap; i--)
                    {
                        overlapSentences.Insert(0, currentChunk[i]);
                        overlapLength += currentChunk[i].Length + 1;
                    }

                    currentChunk = overlapSentences;
                    currentLength = overlapLength;
                }

                currentChunk.Add(sentence);
                currentLength += sentenceLength;
            }

            if (currentChunk.Count > 0)
            {
                var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                {
                    ["chunk_index"] = chunkIndex.ToString(),
                    ["sentence_count"] = currentChunk.Count.ToString()
                };

                chunks.Add(new TextChunk
                {
                    Text = string.Join(" ", currentChunk),
                    Metadata = chunkMetadata
                });
            }

            return chunks;
        }

        private List<string> SplitIntoSentences(string text)
        {
            var sentences = new List<string>();
            var splits = SentencePattern.Split(text);

            foreach (var split in splits)
            {
                var trimmed = split.Trim();
                if (!string.IsNullOrWhiteSpace(trimmed))
                {
                    sentences.Add(trimmed);
                }
            }

            if (sentences.Count == 0 && !string.IsNullOrWhiteSpace(text))
            {
                sentences.Add(text.Trim());
            }

            return sentences;
        }
    }
}
