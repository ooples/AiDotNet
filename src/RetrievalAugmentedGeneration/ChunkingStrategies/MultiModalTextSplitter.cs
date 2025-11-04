using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Chunking strategy for multi-modal content (text, images, tables, etc.)
    /// </summary>
    public class MultiModalTextSplitter : ChunkingStrategyBase
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;
        private readonly Dictionary<string, Func<string, List<TextChunk>>> _modalityHandlers;

        public MultiModalTextSplitter(int chunkSize = 1000, int chunkOverlap = 200)
        {
            _chunkSize = chunkSize;
            _chunkOverlap = chunkOverlap;
            _modalityHandlers = new Dictionary<string, Func<string, List<TextChunk>>>();
        }

        public void RegisterModalityHandler(string modalityType, Func<string, List<TextChunk>> handler)
        {
            if (string.IsNullOrEmpty(modalityType))
                throw new ArgumentException("Modality type cannot be null or empty", nameof(modalityType));
            if (handler == null)
                throw new ArgumentNullException(nameof(handler));

            _modalityHandlers[modalityType] = handler;
        }

        protected override List<TextChunk> SplitCore(string text, Dictionary<string, string>? metadata = null)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<TextChunk>();

            var chunks = new List<TextChunk>();
            var currentPosition = 0;
            var chunkIndex = 0;

            var sections = ExtractModalitySections(text);

            foreach (var (modality, content, start, end) in sections)
            {
                if (_modalityHandlers.TryGetValue(modality, out var handler))
                {
                    var modalityChunks = handler(content);
                    foreach (var chunk in modalityChunks)
                    {
                        chunk.Metadata ??= new Dictionary<string, string>();
                        chunk.Metadata["modality"] = modality;
                        chunk.Metadata["chunk_index"] = chunkIndex.ToString();
                        if (metadata != null)
                        {
                            foreach (var (key, value) in metadata)
                            {
                                chunk.Metadata[key] = value;
                            }
                        }
                        chunkIndex++;
                    }
                    chunks.AddRange(modalityChunks);
                }
                else
                {
                    var defaultChunks = ChunkText(content, modality, metadata, ref chunkIndex);
                    chunks.AddRange(defaultChunks);
                }

                currentPosition = end;
            }

            return chunks;
        }

        private List<(string modality, string content, int start, int end)> ExtractModalitySections(string text)
        {
            var sections = new List<(string, string, int, int)>();
            var currentPosition = 0;

            while (currentPosition < text.Length)
            {
                var nextMarker = text.IndexOf("[MODALITY:", currentPosition, StringComparison.Ordinal);
                
                if (nextMarker == -1)
                {
                    if (currentPosition < text.Length)
                    {
                        sections.Add(("text", text.Substring(currentPosition), currentPosition, text.Length));
                    }
                    break;
                }

                if (nextMarker > currentPosition)
                {
                    sections.Add(("text", text.Substring(currentPosition, nextMarker - currentPosition),
                        currentPosition, nextMarker));
                }

                var endMarker = text.IndexOf(']', nextMarker);
                if (endMarker == -1)
                {
                    sections.Add(("text", text.Substring(currentPosition), currentPosition, text.Length));
                    break;
                }

                var modalityType = text.Substring(nextMarker + 10, endMarker - nextMarker - 10);
                var contentEnd = text.IndexOf("[/MODALITY]", endMarker, StringComparison.Ordinal);
                
                if (contentEnd == -1)
                {
                    sections.Add(("text", text.Substring(currentPosition), currentPosition, text.Length));
                    break;
                }

                var content = text.Substring(endMarker + 1, contentEnd - endMarker - 1);
                sections.Add((modalityType, content, nextMarker, contentEnd + 11));
                currentPosition = contentEnd + 11;
            }

            return sections;
        }

        private List<TextChunk> ChunkText(string text, string modality, Dictionary<string, string>? metadata, ref int chunkIndex)
        {
            var chunks = new List<TextChunk>();
            var words = text.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            var currentChunk = new List<string>();
            var currentLength = 0;

            foreach (var word in words)
            {
                if (currentLength + word.Length + 1 > _chunkSize && currentChunk.Count > 0)
                {
                    var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                    {
                        ["modality"] = modality,
                        ["chunk_index"] = chunkIndex.ToString()
                    };

                    chunks.Add(new TextChunk
                    {
                        Text = string.Join(" ", currentChunk),
                        Metadata = chunkMetadata
                    });
                    chunkIndex++;

                    currentChunk.Clear();
                    currentLength = 0;
                }

                currentChunk.Add(word);
                currentLength += word.Length + 1;
            }

            if (currentChunk.Count > 0)
            {
                var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                {
                    ["modality"] = modality,
                    ["chunk_index"] = chunkIndex.ToString()
                };

                chunks.Add(new TextChunk
                {
                    Text = string.Join(" ", currentChunk),
                    Metadata = chunkMetadata
                });
                chunkIndex++;
            }

            return chunks;
        }
    }
}
