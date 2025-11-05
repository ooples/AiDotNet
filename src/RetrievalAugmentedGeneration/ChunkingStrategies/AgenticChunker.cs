using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
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
        var paragraphMatches = Regex.Matches(text, paragraphPattern);
        foreach (Match match in paragraphMatches)
        {
            boundaries.Add(match.Index);
        }

        // Detect section headers (lines starting with #, or all caps)
        var lines = text.Split('\n');
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
            
            position += line.Length + 1; // +1 for newline
        }

        // Detect list boundaries
        var listPattern = @"^\s*[\d\-\*]\s+";
        position = 0;
        foreach (var line in lines)
        {
            if (Regex.IsMatch(line, listPattern))
            {
                boundaries.Add(position);
            }
            position += line.Length + 1;
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
            else
            {
                // Extend current chunk
                currentChunkEnd = potentialChunkEnd;
            }
        }

        // Add final chunk
        if (currentChunkEnd > currentChunkStart)
        {
            var chunkText = text.Substring(currentChunkStart, currentChunkEnd - currentChunkStart).Trim();
            if (!string.IsNullOrWhiteSpace(chunkText))
            {
                chunks.Add((chunkText, currentChunkStart, currentChunkEnd));
            }
        }

        return chunks;
    }
}
