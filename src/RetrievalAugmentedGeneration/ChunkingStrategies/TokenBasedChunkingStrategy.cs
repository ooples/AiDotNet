using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Chunks text by TOKEN count rather than character count. <see cref="ChunkingStrategyBase.ChunkSize"/>
    /// and <see cref="ChunkingStrategyBase.ChunkOverlap"/> are interpreted as tokens.
    /// </summary>
    /// <remarks>
    /// <para>
    /// LLM context windows and embedding limits are measured in tokens, not characters, so token-aware
    /// splitting (LangChain <c>TokenTextSplitter</c>, LlamaIndex token node parser) is the correct default
    /// for RAG. A token-counter delegate is injectable so callers can plug an exact tokenizer (tiktoken /
    /// HF); the built-in default counts whitespace-separated words as an approximate proxy.
    /// </para>
    /// <para><b>For Beginners:</b> models "see" text as tokens (~¾ of a word each). This splitter keeps each
    /// chunk under a token budget so it always fits the model, instead of guessing with character counts.</para>
    /// </remarks>
    [ComponentType(ComponentType.Chunker)]
    [PipelineStage(PipelineStage.DataIngestion)]
    public class TokenBasedChunkingStrategy : ChunkingStrategyBase
    {
        private readonly Func<string, int> _tokenCounter;

        /// <param name="maxTokens">Maximum tokens per chunk.</param>
        /// <param name="overlapTokens">Approximate token overlap between consecutive chunks.</param>
        /// <param name="tokenCounter">
        /// Optional exact token counter (e.g. a tiktoken/HF adapter). When null, whitespace word count is
        /// used as an approximate proxy; with a custom multi-token-per-word counter the overlap is approximate.
        /// </param>
        public TokenBasedChunkingStrategy(int maxTokens = 256, int overlapTokens = 32, Func<string, int>? tokenCounter = null)
            : base(maxTokens, overlapTokens)
        {
            _tokenCounter = tokenCounter ?? DefaultWordCount;
        }

        private static int DefaultWordCount(string s)
            => string.IsNullOrWhiteSpace(s) ? 0 : s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;

        protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
        {
            var words = SplitWordsWithPositions(text);
            if (words.Count == 0)
            {
                yield break;
            }

            int chunkStart = 0;
            while (chunkStart < words.Count)
            {
                int end = chunkStart;
                int tokens = 0;
                while (end < words.Count)
                {
                    int wt = Math.Max(1, _tokenCounter(words[end].Word));
                    // Always take at least one word so a single over-budget word can't stall progress.
                    if (end > chunkStart && tokens + wt > ChunkSize)
                    {
                        break;
                    }

                    tokens += wt;
                    end++;
                }

                int startPos = words[chunkStart].Start;
                int endPos = words[end - 1].End;
                yield return (text.Substring(startPos, endPos - startPos), startPos, endPos);

                if (end >= words.Count)
                {
                    break;
                }

                // Advance, leaving ~ChunkOverlap tokens of overlap. Guaranteed forward progress (>= 1 word).
                int wordsInChunk = end - chunkStart;
                int advance = Math.Max(1, wordsInChunk - Math.Min(ChunkOverlap, wordsInChunk - 1));
                chunkStart += advance;
            }
        }

        private static List<(string Word, int Start, int End)> SplitWordsWithPositions(string text)
        {
            var words = new List<(string, int, int)>();
            int i = 0;
            while (i < text.Length)
            {
                while (i < text.Length && char.IsWhiteSpace(text[i])) i++;
                if (i >= text.Length) break;
                int start = i;
                while (i < text.Length && !char.IsWhiteSpace(text[i])) i++;
                words.Add((text.Substring(start, i - start), start, i));
            }

            return words;
        }
    }
}
