using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Sliding window chunking strategy with configurable window size and stride.
    /// </summary>
    public class SlidingWindowChunkingStrategy : ChunkingStrategyBase
    {
        private readonly int _stride;

        /// <summary>
        /// Initializes a new instance of the <see cref="SlidingWindowChunkingStrategy"/> class.
        /// </summary>
        /// <param name="windowSize">The size of the sliding window.</param>
        /// <param name="stride">The stride (step size) of the window.</param>
        public SlidingWindowChunkingStrategy(int windowSize = 1000, int stride = 500)
            : base(windowSize, Math.Max(0, windowSize - stride))
        {
            if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
            if (stride > windowSize)
                throw new ArgumentOutOfRangeException(nameof(stride), "Stride cannot exceed the window size because chunk overlap must stay non-negative.");
            _stride = stride;
        }

        /// <summary>
        /// Chunks text using a sliding window approach.
        /// </summary>
        /// <param name="text">The text to chunk.</param>
        /// <returns>A collection of overlapping text chunks with positions.</returns>
        protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
        {
            for (int i = 0; i < text.Length; i += _stride)
            {
                var length = Math.Min(ChunkSize, text.Length - i);
                var chunk = text.Substring(i, length);
                yield return (chunk, i, i + length);

                if (i + length >= text.Length)
                {
                    break;
                }
            }
        }
    }
}
