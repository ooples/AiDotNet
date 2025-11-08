using System.Collections;
using AiDotNet.Helpers;

namespace AiDotNet.ModelCompression;

/// <summary>
/// Implements Huffman encoding compression for model weights using variable-length encoding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Huffman encoding is a lossless compression technique that assigns shorter codes to more frequent
/// values and longer codes to less frequent values. This is particularly effective when combined
/// with weight clustering, where cluster indices have non-uniform frequency distributions.
/// </para>
/// <para><b>For Beginners:</b> Huffman encoding is like creating custom abbreviations.
///
/// Imagine you're taking notes in a lecture:
/// - Words you hear often (like "the", "and", "is") you abbreviate with single letters
/// - Rare words you write out in full
/// - This makes your notes much shorter overall
///
/// For neural networks:
/// - Some weight values appear much more frequently than others
/// - Frequent values get short binary codes (like "01")
/// - Rare values get longer codes (like "110101")
/// - Since frequent values appear often, using short codes saves a lot of space
///
/// The magic is that Huffman encoding is "lossless":
/// - You can perfectly reconstruct the original values
/// - No accuracy is lost (unlike clustering which is lossy)
/// - It's often combined with clustering for even better compression
///
/// Example:
/// - Value appearing 1000 times: code "1" (1 bit each = 1000 bits total)
/// - Value appearing 10 times: code "01001" (5 bits each = 50 bits total)
/// - Total: 1050 bits instead of possibly much more with fixed-length codes
/// </para>
/// </remarks>
public class HuffmanEncodingCompression<T> : ModelCompressionBase<T>
{
    private readonly int _precision;

    /// <summary>
    /// Initializes a new instance of the HuffmanEncodingCompression class.
    /// </summary>
    /// <param name="precision">The number of decimal places to round to before encoding (default: 6).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Precision controls how many decimal places to keep.
    ///
    /// - Higher precision = more accurate but less compression
    ///   * precision=8: 0.12345678 (very precise, more unique values, less compression)
    ///
    /// - Lower precision = more compression but slightly less accurate
    ///   * precision=4: 0.1235 (rounded, fewer unique values, better compression)
    ///
    /// The default of 6 decimal places is usually a good balance.
    ///
    /// Why round at all?
    /// - Without rounding, you might have millions of unique values
    /// - Huffman encoding works best when many values are the same
    /// - Rounding creates more duplicate values, improving compression
    /// </para>
    /// </remarks>
    public HuffmanEncodingCompression(int precision = 6)
    {
        if (precision < 0)
        {
            throw new ArgumentException("Precision must be non-negative.", nameof(precision));
        }

        _precision = precision;
    }

    /// <summary>
    /// Compresses weights using Huffman encoding.
    /// </summary>
    /// <param name="weights">The original model weights.</param>
    /// <returns>Compressed weights and metadata containing the Huffman tree and encoding table.</returns>
    public override (T[] compressedWeights, object metadata) Compress(T[] weights)
    {
        if (weights == null || weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be null or empty.", nameof(weights));
        }

        // Round weights to specified precision to reduce unique values
        var roundedWeights = weights.Select(w => RoundToPrecision(w)).ToArray();

        // Build frequency table
        var frequencies = BuildFrequencyTable(roundedWeights);

        // Build Huffman tree
        var huffmanTree = BuildHuffmanTree(frequencies);

        // Generate encoding table
        var encodingTable = GenerateEncodingTable(huffmanTree);

        // Encode the weights
        var encodedBits = EncodeWeights(roundedWeights, encodingTable);

        // Create metadata
        var metadata = new HuffmanEncodingMetadata<T>
        {
            HuffmanTree = huffmanTree,
            EncodingTable = encodingTable,
            OriginalLength = weights.Length,
            BitLength = encodedBits.Count
        };

        // Convert BitArray to byte array and then to T[] for storage
        var compressedBytes = ConvertBitArrayToBytes(encodedBits);
        var compressedWeights = new T[compressedBytes.Length];
        for (int i = 0; i < compressedBytes.Length; i++)
        {
            compressedWeights[i] = NumOps.FromDouble(compressedBytes[i]);
        }

        return (compressedWeights, metadata);
    }

    /// <summary>
    /// Decompresses weights by decoding the Huffman-encoded bit stream.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights (encoded as bytes).</param>
    /// <param name="metadata">The metadata containing the Huffman tree.</param>
    /// <returns>The decompressed weights.</returns>
    public override T[] Decompress(T[] compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        if (metadata is not HuffmanEncodingMetadata<T> huffmanMetadata)
        {
            throw new ArgumentException("Invalid metadata type for Huffman encoding.", nameof(metadata));
        }

        // Convert T[] back to byte array
        var compressedBytes = new byte[compressedWeights.Length];
        for (int i = 0; i < compressedWeights.Length; i++)
        {
            compressedBytes[i] = (byte)NumOps.ToDouble(compressedWeights[i]);
        }

        // Convert bytes to BitArray
        var encodedBits = new BitArray(compressedBytes);

        // Decode the weights
        var decodedWeights = DecodeWeights(encodedBits, huffmanMetadata.HuffmanTree,
            huffmanMetadata.OriginalLength, huffmanMetadata.BitLength);

        return decodedWeights;
    }

    /// <summary>
    /// Gets the compressed size including the Huffman tree and encoded bits.
    /// </summary>
    public override long GetCompressedSize(T[] compressedWeights, object metadata)
    {
        if (metadata is not HuffmanEncodingMetadata<T> huffmanMetadata)
        {
            throw new ArgumentException("Invalid metadata type.", nameof(metadata));
        }

        // Size of encoded bits (in bytes)
        long encodedBitsSize = compressedWeights.Length;

        // Size of Huffman tree (approximate - depends on tree structure)
        // Each unique value + frequency + tree structure
        long treeSize = EstimateHuffmanTreeSize(huffmanMetadata.HuffmanTree);

        // Metadata overhead
        long metadataSize = sizeof(int) * 2; // OriginalLength, BitLength

        return encodedBitsSize + treeSize + metadataSize;
    }

    /// <summary>
    /// Rounds a weight value to the specified precision.
    /// </summary>
    private T RoundToPrecision(T weight)
    {
        double value = NumOps.ToDouble(weight);
        double multiplier = Math.Pow(10, _precision);
        double rounded = Math.Round(value * multiplier) / multiplier;
        return NumOps.FromDouble(rounded);
    }

    /// <summary>
    /// Builds a frequency table for the weights.
    /// </summary>
    private Dictionary<T, int> BuildFrequencyTable(T[] weights)
    {
        var frequencies = new Dictionary<T, int>();
        foreach (var weight in weights)
        {
            if (frequencies.TryGetValue(weight, out int count))
            {
                frequencies[weight] = count + 1;
            }
            else
            {
                frequencies[weight] = 1;
            }
        }
        return frequencies;
    }

    /// <summary>
    /// Builds a Huffman tree from the frequency table.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Huffman tree determines which codes are assigned to which values.
    ///
    /// The algorithm works like a tournament:
    /// 1. Start with all values as individual "trees"
    /// 2. Repeatedly combine the two trees with the lowest frequencies
    /// 3. Continue until there's only one tree left
    ///
    /// This ensures that:
    /// - Values with high frequency end up near the top (short codes)
    /// - Values with low frequency end up deep down (long codes)
    /// - The total encoded size is minimized
    /// </para>
    /// </remarks>
    private HuffmanNode<T> BuildHuffmanTree(Dictionary<T, int> frequencies)
    {
        var priorityQueue = new SortedSet<HuffmanNode<T>>(Comparer<HuffmanNode<T>>.Create((a, b) =>
        {
            int freqCompare = a.Frequency.CompareTo(b.Frequency);
            return freqCompare != 0 ? freqCompare : a.Id.CompareTo(b.Id);
        }));

        int nodeIdCounter = 0;
        foreach (var kvp in frequencies)
        {
            priorityQueue.Add(new HuffmanNode<T>
            {
                Value = kvp.Key,
                Frequency = kvp.Frequency,
                IsLeaf = true,
                Id = nodeIdCounter++
            });
        }

        while (priorityQueue.Count > 1)
        {
            var left = priorityQueue.Min!;
            priorityQueue.Remove(left);
            var right = priorityQueue.Min!;
            priorityQueue.Remove(right);

            var parent = new HuffmanNode<T>
            {
                Frequency = left.Frequency + right.Frequency,
                Left = left,
                Right = right,
                IsLeaf = false,
                Id = nodeIdCounter++
            };

            priorityQueue.Add(parent);
        }

        return priorityQueue.Min!;
    }

    /// <summary>
    /// Generates an encoding table from the Huffman tree.
    /// </summary>
    private Dictionary<T, string> GenerateEncodingTable(HuffmanNode<T> root)
    {
        var encodingTable = new Dictionary<T, string>();
        GenerateEncodingTableRecursive(root, "", encodingTable);
        return encodingTable;
    }

    private void GenerateEncodingTableRecursive(HuffmanNode<T>? node, string code, Dictionary<T, string> table)
    {
        if (node == null) return;

        if (node.IsLeaf && node.Value != null)
        {
            table[node.Value] = string.IsNullOrEmpty(code) ? "0" : code;
        }
        else
        {
            GenerateEncodingTableRecursive(node.Left, code + "0", table);
            GenerateEncodingTableRecursive(node.Right, code + "1", table);
        }
    }

    /// <summary>
    /// Encodes the weights using the encoding table.
    /// </summary>
    private BitArray EncodeWeights(T[] weights, Dictionary<T, string> encodingTable)
    {
        var bits = new List<bool>();
        foreach (var weight in weights)
        {
            string code = encodingTable[weight];
            bits.AddRange(code.Select(c => c == '1'));
        }
        return new BitArray(bits.ToArray());
    }

    /// <summary>
    /// Decodes the encoded bits using the Huffman tree.
    /// </summary>
    private T[] DecodeWeights(BitArray encodedBits, HuffmanNode<T> huffmanTree, int originalLength, int bitLength)
    {
        var decodedWeights = new List<T>();
        var currentNode = huffmanTree;

        for (int i = 0; i < bitLength && decodedWeights.Count < originalLength; i++)
        {
            currentNode = encodedBits[i] ? currentNode.Right! : currentNode.Left!;

            if (currentNode.IsLeaf && currentNode.Value != null)
            {
                decodedWeights.Add(currentNode.Value);
                currentNode = huffmanTree;
            }
        }

        return decodedWeights.ToArray();
    }

    /// <summary>
    /// Converts a BitArray to a byte array.
    /// </summary>
    private byte[] ConvertBitArrayToBytes(BitArray bits)
    {
        int numBytes = (bits.Count + 7) / 8;
        byte[] bytes = new byte[numBytes];
        bits.CopyTo(bytes, 0);
        return bytes;
    }

    /// <summary>
    /// Estimates the size of the Huffman tree structure.
    /// </summary>
    private long EstimateHuffmanTreeSize(HuffmanNode<T> root)
    {
        if (root == null) return 0;

        long size = sizeof(int) + sizeof(bool); // Frequency + IsLeaf flag

        if (root.IsLeaf)
        {
            size += GetElementSize(); // Value
        }
        else
        {
            size += EstimateHuffmanTreeSize(root.Left!);
            size += EstimateHuffmanTreeSize(root.Right!);
        }

        return size;
    }
}

/// <summary>
/// Represents a node in the Huffman tree.
/// </summary>
public class HuffmanNode<T>
{
    public T? Value { get; init; }
    public int Frequency { get; init; }
    public bool IsLeaf { get; init; }
    public HuffmanNode<T>? Left { get; init; }
    public HuffmanNode<T>? Right { get; init; }
    public int Id { get; init; } // For stable sorting
}

/// <summary>
/// Metadata for Huffman encoding compression.
/// </summary>
public class HuffmanEncodingMetadata<T>
{
    public required HuffmanNode<T> HuffmanTree { get; init; }
    public required Dictionary<T, string> EncodingTable { get; init; }
    public required int OriginalLength { get; init; }
    public required int BitLength { get; init; }
}
