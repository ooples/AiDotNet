using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    /// <summary>
    /// Common interface for vector quantizers used to compress dense embeddings for
    /// memory- and latency-efficient similarity search.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Quantization trades a small amount of accuracy for large reductions in memory
    /// footprint (and often faster distance computation). This is how libraries such as
    /// FAISS, Qdrant and pgvector serve billions of vectors on commodity hardware.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A raw embedding stores every dimension as a full
    /// floating-point number (4 or 8 bytes each). Quantization replaces those numbers
    /// with much smaller codes (for example a single byte, or even a single bit, per
    /// dimension) plus a small "dictionary" that lets you approximately reconstruct the
    /// original values. You lose a little precision but save a lot of memory.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public interface IVectorQuantizer<T>
    {
        /// <summary>
        /// Gets a value indicating whether the quantizer has been trained and is ready to encode.
        /// </summary>
        bool IsTrained { get; }

        /// <summary>
        /// Gets the dimensionality of the vectors this quantizer was trained on, or 0 if untrained.
        /// </summary>
        int Dimension { get; }

        /// <summary>
        /// Gets the number of bytes produced by <see cref="Encode"/> for a single vector, or 0 if untrained.
        /// </summary>
        int CodeLength { get; }

        /// <summary>
        /// Trains the quantizer (learns any required statistics or codebooks) from a set of vectors.
        /// </summary>
        /// <param name="vectors">The training vectors. All must share the same dimensionality.</param>
        void Train(IEnumerable<Vector<T>> vectors);

        /// <summary>
        /// Encodes a vector into its compact byte representation.
        /// </summary>
        /// <param name="vector">The vector to encode.</param>
        /// <returns>The quantized code.</returns>
        byte[] Encode(Vector<T> vector);

        /// <summary>
        /// Approximately reconstructs a vector from its compact byte representation.
        /// </summary>
        /// <param name="code">The quantized code produced by <see cref="Encode"/>.</param>
        /// <returns>The reconstructed (approximate) vector.</returns>
        Vector<T> Decode(byte[] code);
    }
}
