using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Recursive character-based text splitting strategy with multiple separators
    /// </summary>
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
            _separators = separators ?? new[] { "\n\n", "\n", " ", "" };
        }

        protected override List<TextChunk> SplitCore(string text, Dictionary<string, string>? metadata = null)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<TextChunk>();

            var chunks = new List<TextChunk>();
            SplitTextRecursive(text, 0, chunks, metadata);
            return chunks;
        }

        private void SplitTextRecursive(
            string text,
            int separatorIndex,
            List<TextChunk> chunks,
            Dictionary<string, string>? metadata)
        {
            if (text.Length <= _chunkSize)
            {
                if (!string.IsNullOrWhiteSpace(text))
                {
                    var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                    {
                        ["chunk_index"] = chunks.Count.ToString()
                    };

                    chunks.Add(new TextChunk
                    {
                        Text = text,
                        Metadata = chunkMetadata
                    });
                }
                return;
            }

            if (separatorIndex >= _separators.Length)
            {
                var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                {
                    ["chunk_index"] = chunks.Count.ToString(),
                    ["truncated"] = "true"
                };

                chunks.Add(new TextChunk
                {
                    Text = text.Substring(0, Math.Min(_chunkSize, text.Length)),
                    Metadata = chunkMetadata
                });
                return;
            }

            var separator = _separators[separatorIndex];
            var splits = !string.IsNullOrEmpty(separator)
                ? text.Split(new[] { separator }, StringSplitOptions.None)
                : text.Select(c => c.ToString()).ToArray();

            var currentChunk = new List<string>();
            var currentLength = 0;

            foreach (var split in splits)
            {
                var splitLength = split.Length + (string.IsNullOrEmpty(separator) ? 0 : separator.Length);

                if (currentLength + splitLength > _chunkSize && currentChunk.Count > 0)
                {
                    var combinedText = string.Join(separator, currentChunk);
                    
                    if (combinedText.Length > _chunkSize)
                    {
                        SplitTextRecursive(combinedText, separatorIndex + 1, chunks, metadata);
                    }
                    else if (!string.IsNullOrWhiteSpace(combinedText))
                    {
                        var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                        {
                            ["chunk_index"] = chunks.Count.ToString()
                        };

                        chunks.Add(new TextChunk
                        {
                            Text = combinedText,
                            Metadata = chunkMetadata
                        });
                    }

                    var overlapText = string.Join(separator, currentChunk.TakeLast(2));
                    if (overlapText.Length <= _chunkOverlap)
                    {
                        currentChunk = currentChunk.TakeLast(2).ToList();
                        currentLength = overlapText.Length;
                    }
                    else
                    {
                        currentChunk.Clear();
                        currentLength = 0;
                    }
                }

                currentChunk.Add(split);
                currentLength += splitLength;
            }

            if (currentChunk.Count > 0)
            {
                var combinedText = string.Join(separator, currentChunk);
                
                if (combinedText.Length > _chunkSize)
                {
                    SplitTextRecursive(combinedText, separatorIndex + 1, chunks, metadata);
                }
                else if (!string.IsNullOrWhiteSpace(combinedText))
                {
                    var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                    {
                        ["chunk_index"] = chunks.Count.ToString()
                    };

                    chunks.Add(new TextChunk
                    {
                        Text = combinedText,
                        Metadata = chunkMetadata
                    });
                }
            }
        }
    }
}
