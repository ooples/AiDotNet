namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of model compression strategies available in the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Model compression reduces the size of AI models while trying to maintain their accuracy.
/// Think of it like compressing a photo - you want a smaller file size but still a recognizable image.
/// Different compression techniques work better for different scenarios and model types.
/// </para>
/// </remarks>
public enum CompressionType
{
    /// <summary>
    /// No compression applied to the model.
    /// </summary>
    None,

    /// <summary>
    /// Weight clustering groups similar weight values together and replaces them with cluster representatives.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weight clustering is like organizing a messy drawer by grouping similar items.
    /// Instead of storing thousands of slightly different weight values (like 0.501, 0.502, 0.503),
    /// the model groups them into clusters and stores just the cluster centers (like 0.5).
    /// This dramatically reduces model size while maintaining most of the model's intelligence.
    /// </para>
    /// </remarks>
    WeightClustering,

    /// <summary>
    /// Huffman encoding uses variable-length codes where frequent values get shorter codes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Huffman encoding is like text message abbreviations. Common words like
    /// "you" become "u" (shorter), while rare words keep their full spelling. Similarly, weights that
    /// appear often in your model get stored with fewer bits, and rare weights use more bits.
    /// This creates an efficient compression without losing any information.
    /// </para>
    /// </remarks>
    HuffmanEncoding,

    /// <summary>
    /// Product quantization divides weight vectors into sub-vectors and quantizes each separately.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Product quantization is like describing a color by breaking it into
    /// red, green, and blue components separately, then rounding each component to the nearest
    /// standard value. For model weights, it divides weight vectors into smaller pieces, compresses
    /// each piece independently, then combines them. This provides better compression than treating
    /// all weights the same way.
    /// </para>
    /// </remarks>
    ProductQuantization,

    /// <summary>
    /// Combines weight clustering with quantization for improved compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This hybrid approach first groups similar weights (clustering) and then
    /// further compresses the cluster centers using quantization. It's like first organizing your
    /// closet by type (shirts, pants, etc.), then within each type, arranging by color codes.
    /// This two-stage process achieves better compression than either technique alone.
    /// </para>
    /// </remarks>
    HybridClusteringQuantization,

    /// <summary>
    /// Combines weight clustering with pruning (removing unimportant weights).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This combines two powerful techniques: clustering (grouping similar weights)
    /// and pruning (removing weights that barely affect the output). It's like cleaning and organizing
    /// a room - you throw away things you don't need (pruning) and organize what's left (clustering).
    /// This can achieve extreme compression while maintaining good accuracy.
    /// </para>
    /// </remarks>
    HybridClusteringPruning,

    /// <summary>
    /// Combines Huffman encoding with weight clustering for maximum compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This technique first groups weights into clusters, then uses Huffman encoding
    /// to efficiently store which cluster each weight belongs to. It's like first organizing books by
    /// category, then creating a shorthand code where popular categories get short codes (like "F" for
    /// Fiction) and rare categories get longer codes. This layered approach maximizes compression.
    /// </para>
    /// </remarks>
    HybridHuffmanClustering
}
