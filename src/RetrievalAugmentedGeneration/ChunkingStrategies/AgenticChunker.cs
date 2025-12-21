using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Production-ready intelligent chunker that decides where to split text based on semantic boundaries.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This chunker analyzes text structure to identify optimal split points based on:
/// - Paragraph boundaries
/// - Topic transitions (detected via sentence similarity)
/// - Natural breaks in content flow (headers, lists, code blocks)
/// - Semantic coherence within chunks
/// </para>
/// <para><b>For Beginners:</b> This is like a smart text splitter that understands content structure.
/// 
/// Think of it like organizing a book:
/// - Don't split in the middle of a sentence or paragraph
/// - Keep related ideas together in the same chunk
/// - Start new chunks at natural topic boundaries
/// - Maintain context with overlapping content
/// 
/// How it works:
/// 1. Identifies structural elements (paragraphs, sections, lists)
/// 2. Calculates semantic coherence scores for potential splits
/// 3. Creates chunks at natural boundaries
/// 4. Adds overlap for context preservation
/// 
/// Example:
/// - Input: Long article about climate change
/// - Output: Chunks at section boundaries, keeping introduction separate from data analysis,
///   solutions separate from problems, etc.
/// 
/// Unlike simple fixed-size chunking:
/// - ✓ Respects paragraph boundaries
/// - ✓ Keeps related sentences together  
/// - ✓ Detects topic changes
/// - ✓ Preserves document structure
/// 
/// Production features:
/// - No external API dependencies
/// - Fast heuristic-based topic detection
/// - Configurable chunk sizes and overlap
/// - Handles multiple document formats
/// - Maintains semantic coherence
/// </para>
/// </remarks>
public class AgenticChunker : ChunkingStrategyBase
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly int _maxChunkSize;
    private readonly double _coherenceThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="AgenticChunker"/> class.
    /// </summary>
    /// <param name="maxChunkSize">Maximum size of each chunk in characters (default: 1000).</param>
    /// <param name="overlap">Number of overlapping characters between chunks (default: 200).</param>
    /// <param name="coherenceThreshold">Minimum coherence score to keep sentences together (0-1, default: 0.3).</param>
    public AgenticChunker(
        int maxChunkSize = 1000,
        int overlap = 200,
        double coherenceThreshold = 0.3)
        : base(maxChunkSize, overlap)
    {
        if (maxChunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxChunkSize), "Max chunk size must be positive");
        if (overlap < 0)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap cannot be negative");
        if (overlap >= maxChunkSize)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Overlap must be less than max chunk size");
        if (coherenceThreshold < 0 || coherenceThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(coherenceThreshold), "Coherence threshold must be between 0 and 1");

        _maxChunkSize = maxChunkSize;
        _coherenceThreshold = coherenceThreshold;
    }

    /// <summary>
    /// Splits text into chunks using intelligent boundary detection.
    /// </summary>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            yield break;

        // Detect structural boundaries (paragraphs, sections)
        var boundaries = DetectBoundaries(text);

        // Create chunks at natural boundaries
        var chunks = CreateSemanticChunks(text, boundaries);

        foreach (var chunk in chunks)
        {
            yield return chunk;
        }
    }

    private List<int> DetectBoundaries(string text)
    {
        var boundaries = new List<int> { 0 };

        // Detect paragraph breaks (double newlines)
        var paragraphPattern = @"\n\s*\n";
        var paragraphMatches = Regex.Matches(text, paragraphPattern, RegexOptions.None, RegexTimeout);
        foreach (Match match in paragraphMatches)
        {
            boundaries.Add(match.Index);
        }

        // Detect section headers (lines starting with #, or all caps)
        var lineEndings = text.Contains("\r\n") ? "\r\n" : "\n";
        var lineEndingLength = lineEndings.Length;
        var lines = text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
        var position = 0;
        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            // Markdown headers or all-caps headers
            if (trimmed.StartsWith("#") ||
                (trimmed.Length > 3 && trimmed.Length < 100 && trimmed == trimmed.ToUpperInvariant() && !trimmed.All(char.IsDigit)))
            {
                boundaries.Add(position);
            }

            position += line.Length + lineEndingLength;
        }

        // Detect list boundaries
        var listPattern = @"^\s*[\d\-\*]\s+";
        position = 0;
        foreach (var line in lines)
        {
            if (Regex.IsMatch(line, listPattern, RegexOptions.None, RegexTimeout))
            {
                boundaries.Add(position);
            }
            position += line.Length + lineEndingLength;
        }

        // Sort and deduplicate boundaries
        boundaries = boundaries.Distinct().OrderBy(b => b).ToList();

        // Add end boundary
        if (boundaries[boundaries.Count - 1] < text.Length)
            boundaries.Add(text.Length);

        return boundaries;
    }

    private List<(string Chunk, int StartPosition, int EndPosition)> CreateSemanticChunks(
        string text,
        List<int> boundaries)
    {
        var chunks = new List<(string, int, int)>();
        var currentChunkStart = 0;
        var currentChunkEnd = 0;

        for (int i = 1; i < boundaries.Count; i++)
        {
            var boundaryPos = boundaries[i];
            var potentialChunkEnd = boundaryPos;

            // Check if adding this boundary would exceed max size
            if (potentialChunkEnd - currentChunkStart > _maxChunkSize)
            {
                // Check semantic coherence before splitting
                var shouldSplit = true;
                if (currentChunkEnd > currentChunkStart)
                {
                    // Calculate coherence between current chunk and the segment up to boundary
                    var coherence = CalculateSemanticCoherence(
                        text.Substring(currentChunkStart, currentChunkEnd - currentChunkStart),
                        text.Substring(currentChunkEnd, Math.Min(boundaryPos - currentChunkEnd, 500)));

                    // If coherence is high, allow slight size overflow
                    if (coherence >= _coherenceThreshold &&
                        (potentialChunkEnd - currentChunkStart) <= (_maxChunkSize * 1.2))
                    {
                        shouldSplit = false;
                        currentChunkEnd = potentialChunkEnd;
                    }
                }

                if (shouldSplit)
                {
                    // Finalize current chunk
                    if (currentChunkEnd > currentChunkStart)
                    {
                        var chunkText = text.Substring(currentChunkStart, currentChunkEnd - currentChunkStart).Trim();
                        if (!string.IsNullOrWhiteSpace(chunkText))
                        {
                            chunks.Add((chunkText, currentChunkStart, currentChunkEnd));
                        }

                        // Start new chunk with overlap
                        currentChunkStart = Math.Max(currentChunkStart, currentChunkEnd - ChunkOverlap);
                        currentChunkEnd = boundaryPos;
                    }
                    else
                    {
                        // First chunk, split at max size
                        var splitPos = Math.Min(currentChunkStart + _maxChunkSize, text.Length);
                        var chunkText = text.Substring(currentChunkStart, splitPos - currentChunkStart).Trim();
                        if (!string.IsNullOrWhiteSpace(chunkText))
                        {
                            chunks.Add((chunkText, currentChunkStart, splitPos));
                        }
                        currentChunkStart = Math.Max(currentChunkStart, splitPos - ChunkOverlap);
                        currentChunkEnd = boundaryPos;
                    }
                }
            }
            else
            {
                // Extend current chunk
                currentChunkEnd = potentialChunkEnd;
            }
        }

        // Add final chunk with size enforcement
        if (currentChunkEnd > currentChunkStart)
        {
            var finalChunkLength = currentChunkEnd - currentChunkStart;

            // If final chunk exceeds max size, split it recursively
            if (finalChunkLength > _maxChunkSize * 1.2)
            {
                var remainingStart = currentChunkStart;
                while (remainingStart < currentChunkEnd)
                {
                    var chunkEnd = Math.Min(remainingStart + _maxChunkSize, currentChunkEnd);
                    var chunkText = text.Substring(remainingStart, chunkEnd - remainingStart).Trim();
                    if (!string.IsNullOrWhiteSpace(chunkText))
                    {
                        chunks.Add((chunkText, remainingStart, chunkEnd));
                    }
                    // Ensure progress: move forward by at least 1 character if overlap would prevent it
                    var nextStart = chunkEnd - ChunkOverlap;
                    remainingStart = nextStart > remainingStart ? nextStart : remainingStart + 1;
                }
            }
            else
            {
                var chunkText = text.Substring(currentChunkStart, finalChunkLength).Trim();
                if (!string.IsNullOrWhiteSpace(chunkText))
                {
                    chunks.Add((chunkText, currentChunkStart, currentChunkEnd));
                }
            }
        }

        return chunks;
    }

    /// <summary>
    /// Calculates semantic coherence between two text segments using lexical overlap and connectivity.
    /// </summary>
    /// <param name="segment1">First text segment.</param>
    /// <param name="segment2">Second text segment.</param>
    /// <returns>Coherence score between 0 and 1, where higher values indicate stronger semantic connection.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines how related two pieces of text are without using AI models.
    /// It uses several heuristics:
    /// 
    /// 1. Word Overlap - How many words appear in both segments?
    ///    Example: "The cat sat" and "The cat ran" share 67% of words (2 of 3)
    /// 
    /// 2. Entity Continuity - Are the same proper nouns/entities mentioned?
    ///    Example: "John went home" and "He was tired" → connected by pronoun reference
    /// 
    /// 3. Transition Words - Does segment2 start with connecting words like "However", "Therefore"?
    ///    These indicate the segments are logically connected
    /// 
    /// The final score combines these factors. High scores (>0.3) mean keep segments together.
    /// Low scores mean a topic change occurred, so splitting is safe.
    /// </para>
    /// </remarks>
    private double CalculateSemanticCoherence(string segment1, string segment2)
    {
        if (string.IsNullOrWhiteSpace(segment1) || string.IsNullOrWhiteSpace(segment2))
            return 0.0;

        // Extract words (excluding common stop words for better signal)
        var stopWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be", "been"
        };

        var words1 = new HashSet<string>(
            segment1.Split(new[] { ' ', '\n', '\r', '\t', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length > 2 && !stopWords.Contains(w))
                .Select(w => w.ToLowerInvariant()),
            StringComparer.OrdinalIgnoreCase);

        var words2 = new HashSet<string>(
            segment2.Split(new[] { ' ', '\n', '\r', '\t', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(w => w.Length > 2 && !stopWords.Contains(w))
                .Select(w => w.ToLowerInvariant()),
            StringComparer.OrdinalIgnoreCase);

        if (words1.Count == 0 || words2.Count == 0)
            return 0.0;

        // Calculate Jaccard similarity (intersection over union)
        var intersection = words1.Intersect(words2).Count();
        var union = words1.Union(words2).Count();
        var jaccardScore = union > 0 ? (double)intersection / union : 0.0;

        // Check for discourse markers (transition words) at start of segment2
        var segment2Trimmed = segment2.TrimStart();
        var discourseMarkers = new[]
        {
            "however", "therefore", "thus", "furthermore", "moreover", "additionally",
            "consequently", "nevertheless", "meanwhile", "similarly", "likewise",
            "in contrast", "on the other hand", "as a result", "for example", "for instance"
        };

        var hasTransition = discourseMarkers.Any(marker =>
            segment2Trimmed.StartsWith(marker, StringComparison.OrdinalIgnoreCase));
        var transitionBonus = hasTransition ? 0.15 : 0.0;

        // Check for entity continuity (capitalized words that might be names/entities)
        var entities1 = new HashSet<string>(
            Regex.Matches(segment1, @"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", RegexOptions.None, RegexTimeout)
                .Cast<Match>()
                .Select(m => m.Value));

        var entities2 = new HashSet<string>(
            Regex.Matches(segment2, @"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", RegexOptions.None, RegexTimeout)
                .Cast<Match>()
                .Select(m => m.Value));

        var entityOverlap = entities1.Count > 0 && entities2.Count > 0
            ? (double)entities1.Intersect(entities2).Count() / Math.Max(entities1.Count, entities2.Count)
            : 0.0;

        // Weighted combination of signals
        var coherenceScore = (jaccardScore * 0.5) + (entityOverlap * 0.35) + transitionBonus;

        return Math.Min(1.0, coherenceScore);
    }
}
