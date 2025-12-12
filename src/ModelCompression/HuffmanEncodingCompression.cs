using System.Collections;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
    private readonly object _lockObject = new object();

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
    public override (Vector<T> compressedWeights, object metadata) Compress(Vector<T> weights)
    {
        if (weights == null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        if (weights.Length == 0)
        {
            throw new ArgumentException("Weights cannot be empty.", nameof(weights));
        }

        lock (_lockObject)
        {
            // Round weights to specified precision to reduce unique values
            var roundedWeights = new T[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                roundedWeights[i] = RoundToPrecision(weights[i]);
            }

            // Build frequency table
            var frequencies = BuildFrequencyTable(roundedWeights);

            if (frequencies.Count == 0)
            {
                throw new InvalidOperationException("No valid weights to compress.");
            }

            // Build Huffman tree
            var huffmanTree = BuildHuffmanTree(frequencies);

            // Generate encoding table
            var encodingTable = GenerateEncodingTable(huffmanTree);

            // Encode the weights
            var encodedBits = EncodeWeights(roundedWeights, encodingTable);

            // Create metadata
            var metadata = new HuffmanEncodingMetadata<T>(
                huffmanTree,
                encodingTable,
                weights.Length,
                encodedBits.Count);

            // Convert BitArray to byte array and then to Vector<T> for storage
            var compressedBytes = ConvertBitArrayToBytes(encodedBits);
            var compressedArray = new T[compressedBytes.Length];
            for (int i = 0; i < compressedBytes.Length; i++)
            {
                compressedArray[i] = NumOps.FromDouble(compressedBytes[i]);
            }

            return (new Vector<T>(compressedArray), metadata);
        }
    }

    /// <summary>
    /// Decompresses weights by decoding the Huffman-encoded bit stream.
    /// </summary>
    /// <param name="compressedWeights">The compressed weights (encoded as bytes).</param>
    /// <param name="metadata">The metadata containing the Huffman tree.</param>
    /// <returns>The decompressed weights.</returns>
    public override Vector<T> Decompress(Vector<T> compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        var huffmanMetadata = metadata as HuffmanEncodingMetadata<T>;
        if (huffmanMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type for Huffman encoding.", nameof(metadata));
        }

        lock (_lockObject)
        {
            // Convert Vector<T> back to byte array
            var compressedBytes = new byte[compressedWeights.Length];
            for (int i = 0; i < compressedWeights.Length; i++)
            {
                double value = NumOps.ToDouble(compressedWeights[i]);
                if (value < 0 || value > 255)
                {
                    throw new InvalidOperationException($"Invalid compressed byte value {value} at position {i}.");
                }
                compressedBytes[i] = (byte)value;
            }

            // Convert bytes to BitArray
            var encodedBits = new BitArray(compressedBytes);

            // Decode the weights
            var decodedArray = DecodeWeights(encodedBits, huffmanMetadata.HuffmanTree,
                huffmanMetadata.OriginalLength, huffmanMetadata.BitLength);

            return new Vector<T>(decodedArray);
        }
    }

    /// <summary>
    /// Gets the compressed size including the Huffman tree and encoded bits.
    /// </summary>
    public override long GetCompressedSize(Vector<T> compressedWeights, object metadata)
    {
        if (compressedWeights == null)
        {
            throw new ArgumentNullException(nameof(compressedWeights));
        }

        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        var huffmanMetadata = metadata as HuffmanEncodingMetadata<T>;
        if (huffmanMetadata == null)
        {
            throw new ArgumentException("Invalid metadata type.", nameof(metadata));
        }

        // Size of encoded bits (in bytes)
        long encodedBitsSize = compressedWeights.Length;

        // Size of Huffman tree (approximate - depends on tree structure)
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
    private NumericDictionary<T, int> BuildFrequencyTable(T[] weights)
    {
        var frequencies = new NumericDictionary<T, int>(weights.Length);
        foreach (var weight in weights)
        {
            frequencies[weight] = frequencies.TryGetValue(weight, out int count) ? count + 1 : 1;
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
    private HuffmanNode<T> BuildHuffmanTree(NumericDictionary<T, int> frequencies)
    {
        var priorityQueue = new SortedSet<HuffmanNode<T>>(Comparer<HuffmanNode<T>>.Create((a, b) =>
        {
            int freqCompare = a.Frequency.CompareTo(b.Frequency);
            return freqCompare != 0 ? freqCompare : a.Id.CompareTo(b.Id);
        }));

        int nodeIdCounter = 0;
        foreach (var kvp in frequencies)
        {
            priorityQueue.Add(new HuffmanNode<T>(kvp.Key, kvp.Value, true, nodeIdCounter++, null, null));
        }

        while (priorityQueue.Count > 1)
        {
            var left = priorityQueue.Min;
            if (left == null)
            {
                throw new InvalidOperationException("Priority queue minimum is null.");
            }
            priorityQueue.Remove(left);

            var right = priorityQueue.Min;
            if (right == null)
            {
                throw new InvalidOperationException("Priority queue minimum is null.");
            }
            priorityQueue.Remove(right);

            var parent = new HuffmanNode<T>(
                default(T),
                left.Frequency + right.Frequency,
                false,
                nodeIdCounter++,
                left,
                right);

            priorityQueue.Add(parent);
        }

        var result = priorityQueue.Min;
        if (result == null)
        {
            throw new InvalidOperationException("Failed to build Huffman tree.");
        }
        return result;
    }

    /// <summary>
    /// Generates an encoding table from the Huffman tree.
    /// </summary>
    private NumericDictionary<T, string> GenerateEncodingTable(HuffmanNode<T> root)
    {
        var encodingTable = new NumericDictionary<T, string>();
        GenerateEncodingTableRecursive(root, "", encodingTable);
        return encodingTable;
    }

    private void GenerateEncodingTableRecursive(HuffmanNode<T> node, string code, NumericDictionary<T, string> table)
    {
        if (node == null) return;

        if (node.IsLeaf && node.Value != null)
        {
            table[node.Value] = string.IsNullOrEmpty(code) ? "0" : code;
        }
        else
        {
            if (node.Left != null)
            {
                GenerateEncodingTableRecursive(node.Left, code + "0", table);
            }
            if (node.Right != null)
            {
                GenerateEncodingTableRecursive(node.Right, code + "1", table);
            }
        }
    }

    /// <summary>
    /// Encodes the weights using the encoding table.
    /// </summary>
    private BitArray EncodeWeights(T[] weights, NumericDictionary<T, string> encodingTable)
    {
        var bits = new List<bool>();
        foreach (var weight in weights)
        {
            if (!encodingTable.TryGetValue(weight, out string code))
            {
                throw new InvalidOperationException($"Weight value {weight} not found in encoding table.");
            }
            foreach (char c in code)
            {
                bits.Add(c == '1');
            }
        }
        return new BitArray(bits.ToArray());
    }

    /// <summary>
    /// Decodes the encoded bits using the Huffman tree.
    /// </summary>
    private T[] DecodeWeights(BitArray encodedBits, HuffmanNode<T> huffmanTree, int originalLength, int bitLength)
    {
        var decodedWeights = new List<T>();

        // Handle degenerate case: single unique value (root is the only leaf)
        if (huffmanTree.IsLeaf)
        {
            if (huffmanTree.Value == null)
            {
                throw new InvalidOperationException("Leaf node has null value.");
            }
            for (int i = 0; i < originalLength; i++)
            {
                decodedWeights.Add(huffmanTree.Value);
            }
            return decodedWeights.ToArray();
        }

        var currentNode = huffmanTree;

        for (int i = 0; i < bitLength && decodedWeights.Count < originalLength; i++)
        {
            if (currentNode == null)
            {
                throw new InvalidOperationException("Huffman tree node is null during decoding.");
            }

            currentNode = encodedBits[i] ? currentNode.Right : currentNode.Left;

            if (currentNode == null)
            {
                throw new InvalidOperationException("Invalid Huffman tree structure during decoding.");
            }

            if (currentNode.IsLeaf)
            {
                if (currentNode.Value == null)
                {
                    throw new InvalidOperationException("Leaf node has null value.");
                }
                decodedWeights.Add(currentNode.Value);
                currentNode = huffmanTree;
            }
        }

        if (decodedWeights.Count != originalLength)
        {
            throw new InvalidOperationException(
                $"Decoded {decodedWeights.Count} weights but expected {originalLength}.");
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

        // Use ternary for clarity - IsLeaf determines which size calculation to use
        size += root.IsLeaf
            ? GetElementSize()
            : (root.Left != null ? EstimateHuffmanTreeSize(root.Left) : 0)
              + (root.Right != null ? EstimateHuffmanTreeSize(root.Right) : 0);

        return size;
    }
}

/// <summary>
/// Represents a node in the Huffman tree.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class HuffmanNode<T>
{
    /// <summary>
    /// Initializes a new instance of the HuffmanNode class.
    /// </summary>
    /// <param name="value">The value stored in this node (for leaf nodes, default for internal nodes).</param>
    /// <param name="frequency">The frequency of this value.</param>
    /// <param name="isLeaf">Whether this is a leaf node.</param>
    /// <param name="id">Unique identifier for stable sorting.</param>
    /// <param name="left">Left child node (null for leaf nodes).</param>
    /// <param name="right">Right child node (null for leaf nodes).</param>
    public HuffmanNode(T? value, int frequency, bool isLeaf, int id, HuffmanNode<T>? left, HuffmanNode<T>? right)
    {
        if (frequency < 0)
        {
            throw new ArgumentException("Frequency cannot be negative.", nameof(frequency));
        }

        Value = value;
        Frequency = frequency;
        IsLeaf = isLeaf;
        Id = id;
        Left = left;
        Right = right;
    }

    /// <summary>
    /// Gets the value stored in this node (for leaf nodes, default for internal nodes).
    /// </summary>
    public T? Value { get; private set; }

    /// <summary>
    /// Gets the frequency of this value or subtree.
    /// </summary>
    public int Frequency { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this is a leaf node.
    /// </summary>
    public bool IsLeaf { get; private set; }

    /// <summary>
    /// Gets the unique identifier for stable sorting.
    /// </summary>
    public int Id { get; private set; }

    /// <summary>
    /// Gets the left child node (null for leaf nodes).
    /// </summary>
    public HuffmanNode<T>? Left { get; private set; }

    /// <summary>
    /// Gets the right child node (null for leaf nodes).
    /// </summary>
    public HuffmanNode<T>? Right { get; private set; }
}

/// <summary>
/// Metadata for Huffman encoding compression.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This metadata stores the information needed to decompress Huffman-encoded weights:
/// - The Huffman tree (for decoding variable-length codes back to values)
/// - The encoding table (mapping values to their codes, used during compression)
/// - The original length and bit length (for proper reconstruction)
///
/// Huffman encoding is lossless - you get exactly the original values back when decompressing.
/// </para>
/// </remarks>
public class HuffmanEncodingMetadata<T> : ICompressionMetadata<T>
{
    /// <summary>
    /// Initializes a new instance of the HuffmanEncodingMetadata class.
    /// </summary>
    /// <param name="huffmanTree">The Huffman tree used for encoding.</param>
    /// <param name="encodingTable">The encoding table mapping values to codes.</param>
    /// <param name="originalLength">The original length of the weights vector.</param>
    /// <param name="bitLength">The length of the encoded bit stream.</param>
    public HuffmanEncodingMetadata(
        HuffmanNode<T> huffmanTree,
        NumericDictionary<T, string> encodingTable,
        int originalLength,
        int bitLength)
    {
        if (huffmanTree == null)
        {
            throw new ArgumentNullException(nameof(huffmanTree));
        }

        if (encodingTable == null)
        {
            throw new ArgumentNullException(nameof(encodingTable));
        }

        if (originalLength <= 0)
        {
            throw new ArgumentException("Original length must be positive.", nameof(originalLength));
        }

        if (bitLength < 0)
        {
            throw new ArgumentException("Bit length cannot be negative.", nameof(bitLength));
        }

        HuffmanTree = huffmanTree;
        EncodingTable = encodingTable;
        OriginalLength = originalLength;
        BitLength = bitLength;
    }

    /// <summary>
    /// Gets the compression type.
    /// </summary>
    public CompressionType Type => CompressionType.HuffmanEncoding;

    /// <summary>
    /// Gets the Huffman tree used for encoding.
    /// </summary>
    public HuffmanNode<T> HuffmanTree { get; private set; }

    /// <summary>
    /// Gets the encoding table mapping values to codes.
    /// </summary>
    public NumericDictionary<T, string> EncodingTable { get; private set; }

    /// <summary>
    /// Gets the original length of the weights vector.
    /// </summary>
    public int OriginalLength { get; private set; }

    /// <summary>
    /// Gets the length of the encoded bit stream.
    /// </summary>
    public int BitLength { get; private set; }

    /// <summary>
    /// Gets the size in bytes of this metadata structure.
    /// </summary>
    public long GetMetadataSize()
    {
        // Approximate size: tree structure + encoding table + original length + bit length
        long treeSize = EstimateTreeSize(HuffmanTree);
        long tableSize = EncodingTable.Count * (sizeof(int) + 10); // Average code length ~10 chars
        return treeSize + tableSize + sizeof(int) + sizeof(int);
    }

    private long EstimateTreeSize(HuffmanNode<T>? node)
    {
        if (node == null) return 0;

        int elementSize = typeof(T) == typeof(float) ? 4 :
                          typeof(T) == typeof(double) ? 8 :
                          System.Runtime.InteropServices.Marshal.SizeOf(typeof(T));

        long size = sizeof(int) + sizeof(bool); // Frequency + IsLeaf flag

        // Use ternary for clarity - IsLeaf determines which size calculation to use
        size += node.IsLeaf
            ? elementSize
            : EstimateTreeSize(node.Left) + EstimateTreeSize(node.Right);

        return size;
    }
}
