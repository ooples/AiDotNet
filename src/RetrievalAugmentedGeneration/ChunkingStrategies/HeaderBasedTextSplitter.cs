using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Chunking strategy that splits text based on header hierarchies
    /// </summary>
    public class HeaderBasedTextSplitter : ChunkingStrategyBase
    {
        private readonly Dictionary<int, string> _headerPatterns;
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;

        public HeaderBasedTextSplitter(int chunkSize = 1000, int chunkOverlap = 200)
        {
            _chunkSize = chunkSize;
            _chunkOverlap = chunkOverlap;
            _headerPatterns = new Dictionary<int, string>
            {
                { 1, @"^#\s+(.+)$" },
                { 2, @"^##\s+(.+)$" },
                { 3, @"^###\s+(.+)$" },
                { 4, @"^####\s+(.+)$" },
                { 5, @"^#####\s+(.+)$" },
                { 6, @"^######\s+(.+)$" }
            };
        }

        protected override List<TextChunk> SplitCore(string text, Dictionary<string, string>? metadata = null)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<TextChunk>();

            var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.None);
            var chunks = new List<TextChunk>();
            var currentSection = new List<string>();
            var currentHeaders = new Stack<(int level, string text)>();
            var chunkIndex = 0;

            foreach (var line in lines)
            {
                var headerMatch = false;
                foreach (var (level, pattern) in _headerPatterns.OrderBy(x => x.Key))
                {
                    var match = Regex.Match(line, pattern, RegexOptions.Multiline);
                    if (match.Success)
                    {
                        if (currentSection.Count > 0)
                        {
                            chunks.AddRange(CreateChunksFromSection(
                                string.Join(Environment.NewLine, currentSection),
                                currentHeaders,
                                metadata,
                                ref chunkIndex));
                            currentSection.Clear();
                        }

                        while (currentHeaders.Count > 0 && currentHeaders.Peek().level >= level)
                        {
                            currentHeaders.Pop();
                        }
                        currentHeaders.Push((level, match.Groups[1].Value));
                        headerMatch = true;
                        break;
                    }
                }

                currentSection.Add(line);
            }

            if (currentSection.Count > 0)
            {
                chunks.AddRange(CreateChunksFromSection(
                    string.Join(Environment.NewLine, currentSection),
                    currentHeaders,
                    metadata,
                    ref chunkIndex));
            }

            return chunks;
        }

        private List<TextChunk> CreateChunksFromSection(
            string sectionText,
            Stack<(int level, string text)> headers,
            Dictionary<string, string>? metadata,
            ref int chunkIndex)
        {
            var chunks = new List<TextChunk>();
            var headerPath = string.Join(" > ", headers.Reverse().Select(h => h.text));

            if (sectionText.Length <= _chunkSize)
            {
                var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                {
                    ["header_path"] = headerPath,
                    ["chunk_index"] = chunkIndex.ToString()
                };

                chunks.Add(new TextChunk
                {
                    Text = sectionText,
                    Metadata = chunkMetadata
                });
                chunkIndex++;
            }
            else
            {
                var words = sectionText.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                var currentChunk = new List<string>();
                var currentLength = 0;

                foreach (var word in words)
                {
                    if (currentLength + word.Length + 1 > _chunkSize && currentChunk.Count > 0)
                    {
                        var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                        {
                            ["header_path"] = headerPath,
                            ["chunk_index"] = chunkIndex.ToString()
                        };

                        chunks.Add(new TextChunk
                        {
                            Text = string.Join(" ", currentChunk),
                            Metadata = chunkMetadata
                        });
                        chunkIndex++;

                        var overlapWords = currentChunk.TakeLast((int)(_chunkOverlap / 5.0)).ToList();
                        currentChunk = overlapWords;
                        currentLength = overlapWords.Sum(w => w.Length + 1);
                    }

                    currentChunk.Add(word);
                    currentLength += word.Length + 1;
                }

                if (currentChunk.Count > 0)
                {
                    var chunkMetadata = new Dictionary<string, string>(metadata ?? new Dictionary<string, string>())
                    {
                        ["header_path"] = headerPath,
                        ["chunk_index"] = chunkIndex.ToString()
                    };

                    chunks.Add(new TextChunk
                    {
                        Text = string.Join(" ", currentChunk),
                        Metadata = chunkMetadata
                    });
                    chunkIndex++;
                }
            }

            return chunks;
        }
    }
}
