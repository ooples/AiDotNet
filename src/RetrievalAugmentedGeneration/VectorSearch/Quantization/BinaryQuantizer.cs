using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    /// <summary>
    /// Sign-based binary quantizer that packs each dimension into a single bit
    /// (1 if the component is non-negative, 0 otherwise). Provides roughly 32x memory
    /// reduction versus 32-bit floats (64x versus 64-bit doubles) and enables extremely
    /// fast Hamming-distance ranking.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Binary quantization is the backbone of "binary embeddings" used by Qdrant and
    /// modern retrieval stacks for a cheap first-pass filter before an exact rerank.
    /// Distances between binary codes are computed with the Hamming distance (the number
    /// of differing bits), which maps to a popcount of the XOR of the two codes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Instead of remembering the exact value of each dimension we
    /// only remember whether it was positive or negative. That single yes/no fact takes
    /// one bit, so 8 dimensions fit in one byte. Comparing two vectors then just counts
    /// how many of these yes/no answers disagree.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class BinaryQuantizer<T> : IVectorQuantizer<T>
    {
        private readonly INumericOperations<T> _numOps;
        private int _dimension;
        private bool _trained;

        /// <inheritdoc/>
        public bool IsTrained => _trained;

        /// <inheritdoc/>
        public int Dimension => _dimension;

        /// <inheritdoc/>
        public int CodeLength => _trained ? ByteCount(_dimension) : 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryQuantizer{T}"/> class.
        /// </summary>
        public BinaryQuantizer()
        {
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Records the dimensionality of the vectors. Binary quantization is parameter-free
        /// (sign based), so training only needs to observe the vector length.
        /// </summary>
        /// <param name="vectors">The training vectors; only the dimensionality is used.</param>
        public void Train(IEnumerable<Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            int dim = -1;
            foreach (var vector in vectors)
            {
                if (vector == null)
                    throw new ArgumentException("Training set contains a null vector.", nameof(vectors));

                if (dim < 0)
                    dim = vector.Length;
                else if (vector.Length != dim)
                    throw new ArgumentException("All training vectors must have the same dimensionality.", nameof(vectors));
            }

            if (dim < 0)
                throw new ArgumentException("Training set must contain at least one vector.", nameof(vectors));

            _dimension = dim;
            _trained = true;
        }

        /// <inheritdoc/>
        public byte[] Encode(Vector<T> vector)
        {
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));

            var arr = vector.ToArray();

            // Allow encoding to infer/lock the dimensionality if not explicitly trained.
            if (!_trained)
            {
                _dimension = arr.Length;
                _trained = true;
            }
            else if (arr.Length != _dimension)
            {
                throw new ArgumentException("Vector dimensionality does not match the trained dimensionality.", nameof(vector));
            }

            var code = new byte[ByteCount(_dimension)];
            for (int i = 0; i < _dimension; i++)
            {
                double v = Convert.ToDouble(arr[i]);
                if (v >= 0.0)
                {
                    code[i >> 3] |= (byte)(1 << (i & 7));
                }
            }

            return code;
        }

        /// <inheritdoc/>
        public Vector<T> Decode(byte[] code)
        {
            if (code == null)
                throw new ArgumentNullException(nameof(code));
            if (!_trained)
                throw new InvalidOperationException("BinaryQuantizer must be trained (or have encoded at least once) before decoding.");
            if (code.Length != ByteCount(_dimension))
                throw new ArgumentException("Code length does not match the trained dimensionality.", nameof(code));

            var one = _numOps.FromDouble(1.0);
            var negOne = _numOps.FromDouble(-1.0);
            var values = new T[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                bool bit = (code[i >> 3] & (1 << (i & 7))) != 0;
                values[i] = bit ? one : negOne;
            }

            return new Vector<T>(values);
        }

        /// <summary>
        /// Computes the Hamming distance (number of differing bits) between two binary codes.
        /// </summary>
        /// <param name="a">The first code.</param>
        /// <param name="b">The second code.</param>
        /// <returns>The number of bits that differ between the two codes.</returns>
        public static int HammingDistance(byte[] a, byte[] b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length)
                throw new ArgumentException("Codes must have the same length.", nameof(b));

            int distance = 0;
            for (int i = 0; i < a.Length; i++)
            {
                distance += PopCount((byte)(a[i] ^ b[i]));
            }

            return distance;
        }

        private static int ByteCount(int dimension)
        {
            return (dimension + 7) / 8;
        }

        private static int PopCount(byte value)
        {
            int v = value;
            int count = 0;
            while (v != 0)
            {
                v &= v - 1; // clear the lowest set bit
                count++;
            }

            return count;
        }
    }
}
