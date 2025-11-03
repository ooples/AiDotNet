using System;
using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Sliding window chunking strategy with configurable window size and stride.
    /// </summary>
    public class SlidingWindowChunkingStrategy : ChunkingStrategyBase
    {
        private readonly int _windowSize;
        private readonly int _stride;

        /// <summary>
        /// Initializes a new instance of the <see cref="SlidingWindowChunkingStrategy"/> class.
        /// </summary>
        /// <param name="windowSize">The size of the sliding window.</param>
        /// <param name="stride">The stride (step size) of the window.</param>
        public SlidingWindowChunkingStrategy(int windowSize = 1000, int stride = 500)
        {
            _windowSize = windowSize > 0 ? windowSize : throw new ArgumentOutOfRangeException(nameof(windowSize));
            _stride = stride > 0 ? stride : throw new ArgumentOutOfRangeException(nameof(stride));
        }

        /// <summary>
        /// Chunks text using a sliding window approach.
        /// </summary>
        /// <param name="text">The text to chunk.</param>
        /// <returns>A list of overlapping text chunks.</returns>
        public override List<string> ChunkText(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var chunks = new List<string>();

            for (int i = 0; i < text.Length; i += _stride)
            {
                var length = Math.Min(_windowSize, text.Length - i);
                chunks.Add(text.Substring(i, length));

                if (i + length >= text.Length)
                {
                    break;
                }
            }

            return chunks;
        }
    }
}
