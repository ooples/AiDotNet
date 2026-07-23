using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;


namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies
{
    /// <summary>
    /// Semantic text chunking. When an embedding function is supplied, it embeds each sentence and starts a
    /// new chunk at semantic breakpoints — consecutive-sentence cosine distances above a percentile
    /// threshold (Kamradt / LlamaIndex SemanticSplitter). Without an embedding function it falls back to
    /// size-based sentence packing.
    /// </summary>
    /// <remarks>
    /// Previously this "semantic" splitter used no embeddings at all — it just packed sentences to a size
    /// budget. It now performs real embedding-distance breakpoint detection when given an embedder.
    /// </remarks>
    [ComponentType(ComponentType.Chunker)]
    [PipelineStage(PipelineStage.DataIngestion)]
    public class SemanticChunkingStrategy : ChunkingStrategyBase
    {
        private readonly Func<IReadOnlyList<string>, IReadOnlyList<double[]>>? _embedBatch;
        private readonly double _breakpointPercentile;

        /// <param name="maxChunkSize">Maximum chunk size in characters (hard cap even in semantic mode).</param>
        /// <param name="chunkOverlap">Chunk overlap in characters (size-based fallback only).</param>
        /// <param name="embedBatch">
        /// Optional sentence embedder (batch: sentences → vectors). When provided, chunk boundaries are placed
        /// at semantic breakpoints; when null, a size-based sentence-packing fallback is used.
        /// </param>
        /// <param name="breakpointPercentile">
        /// Distance percentile (0–100) above which a sentence boundary becomes a chunk breakpoint. Higher =
        /// fewer, larger chunks. Default 95.
        /// </param>
        public SemanticChunkingStrategy(
            int maxChunkSize = 1000,
            int chunkOverlap = 200,
            Func<IReadOnlyList<string>, IReadOnlyList<double[]>>? embedBatch = null,
            double breakpointPercentile = 95.0)
            : base(maxChunkSize, chunkOverlap)
        {
            _embedBatch = embedBatch;
            _breakpointPercentile = Math.Min(100.0, Math.Max(0.0, breakpointPercentile));
        }

        protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
        {
            var sentences = SplitIntoSentences(text);
            if (_embedBatch == null || sentences.Count < 3)
            {
                return SizeBasedChunks(sentences);
            }

            return SemanticChunks(sentences, text);
        }

        // Real semantic chunking: break where the cosine DISTANCE between consecutive sentence embeddings
        // exceeds the configured percentile of all such distances (a topic shift), also capping chunk size.
        private IEnumerable<(string Chunk, int StartPosition, int EndPosition)> SemanticChunks(List<string> sentences, string text)
        {
            var spans = LocateSentenceSpans(sentences, text);
            var embeddings = _embedBatch!(sentences);

            var distances = new List<double>(sentences.Count - 1);
            for (int i = 0; i < sentences.Count - 1; i++)
            {
                distances.Add(1.0 - CosineSimilarity(embeddings[i], embeddings[i + 1]));
            }

            double threshold = Percentile(distances, _breakpointPercentile);

            var results = new List<(string, int, int)>();
            int chunkStart = 0;
            for (int i = 0; i < sentences.Count; i++)
            {
                bool lastSentence = i == sentences.Count - 1;
                int startPos = spans[chunkStart].Start;
                int endPos = spans[i].End;
                bool sizeExceeded = (endPos - startPos) >= ChunkSize && i > chunkStart;
                bool semanticBreak = !lastSentence && distances[i] > threshold;

                if (sizeExceeded)
                {
                    // Emit up to the previous sentence, then reconsider the current one as a chunk start.
                    int prevEnd = spans[i - 1].End;
                    results.Add((text.Substring(startPos, prevEnd - startPos), startPos, prevEnd));
                    chunkStart = i;
                    startPos = spans[chunkStart].Start;
                    endPos = spans[i].End;
                }

                if (lastSentence)
                {
                    results.Add((text.Substring(startPos, endPos - startPos), startPos, endPos));
                }
                else if (semanticBreak)
                {
                    results.Add((text.Substring(startPos, endPos - startPos), startPos, endPos));
                    chunkStart = i + 1;
                }
            }

            return results;
        }

        // Size-based fallback (original behavior) when no embedder is supplied.
        private IEnumerable<(string Chunk, int StartPosition, int EndPosition)> SizeBasedChunks(List<string> sentences)
        {
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

                    position = Math.Max(0, endPos - ChunkOverlap);
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

        // Find each sentence's [start,end) span in the original text, scanning left-to-right so repeated
        // sentences map to distinct occurrences.
        private static List<(int Start, int End)> LocateSentenceSpans(List<string> sentences, string text)
        {
            var spans = new List<(int, int)>(sentences.Count);
            int searchFrom = 0;
            foreach (var sentence in sentences)
            {
                int idx = text.IndexOf(sentence, searchFrom, StringComparison.Ordinal);
                if (idx < 0)
                {
                    // Sentence normalization changed the text; approximate with the running cursor.
                    idx = Math.Min(searchFrom, Math.Max(0, text.Length - 1));
                }

                int end = Math.Min(text.Length, idx + sentence.Length);
                spans.Add((idx, end));
                searchFrom = end;
            }

            return spans;
        }

        private static double CosineSimilarity(double[] a, double[] b)
        {
            if (a == null || b == null || a.Length == 0 || a.Length != b.Length) return 0.0;
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }

            double denom = Math.Sqrt(na) * Math.Sqrt(nb);
            return denom > 1e-12 ? dot / denom : 0.0;
        }

        private static double Percentile(List<double> values, double percentile)
        {
            if (values.Count == 0) return 0.0;
            var sorted = values.OrderBy(v => v).ToList();
            double rank = (percentile / 100.0) * (sorted.Count - 1);
            int lo = (int)Math.Floor(rank);
            int hi = (int)Math.Ceiling(rank);
            if (lo == hi) return sorted[lo];
            double frac = rank - lo;
            return sorted[lo] * (1 - frac) + sorted[hi] * frac;
        }

        private List<string> SplitIntoSentences(string text)
        {
            return Helpers.TextProcessingHelper.SplitIntoSentences(text);
        }
    }
}
