using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    /// <summary>
    /// Per-dimension scalar quantizer that maps each floating-point component to a single
    /// unsigned byte (uint8) using learned min/max ranges. Provides roughly 4x memory
    /// reduction versus 32-bit floats (8x versus 64-bit doubles).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each dimension is linearly mapped from its observed [min, max] range onto the
    /// integer range [0, 255]. Reconstruction reverses the mapping using the bin center.
    /// This is the same idea used by FAISS <c>ScalarQuantizer</c> (SQ8) and pgvector's
    /// halfvec/quantization paths.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine each dimension has values between, say, -3 and +5.
    /// We slice that range into 256 equal buckets and store just the bucket number
    /// (0-255, one byte) instead of the full number. To decode we return the middle of
    /// the bucket. Close, but far smaller.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ScalarQuantizer<T> : IVectorQuantizer<T>
    {
        private const int Levels = 256;

        private readonly INumericOperations<T> _numOps;
        private double[]? _min;
        private double[]? _scale;   // (max - min) / 255 per dimension
        private int _dimension;

        /// <inheritdoc/>
        public bool IsTrained => _min != null;

        /// <inheritdoc/>
        public int Dimension => _dimension;

        /// <inheritdoc/>
        public int CodeLength => IsTrained ? _dimension : 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="ScalarQuantizer{T}"/> class.
        /// </summary>
        public ScalarQuantizer()
        {
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <inheritdoc/>
        public void Train(IEnumerable<Vector<T>> vectors)
        {
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            double[]? min = null;
            double[]? max = null;
            int dim = 0;
            int count = 0;

            foreach (var vector in vectors)
            {
                if (vector == null)
                    throw new ArgumentException("Training set contains a null vector.", nameof(vectors));

                var arr = vector.ToArray();
                if (count == 0)
                {
                    dim = arr.Length;
                    min = new double[dim];
                    max = new double[dim];
                    for (int i = 0; i < dim; i++)
                    {
                        double v = Convert.ToDouble(arr[i]);
                        min[i] = v;
                        max[i] = v;
                    }
                }
                else
                {
                    if (arr.Length != dim)
                        throw new ArgumentException("All training vectors must have the same dimensionality.", nameof(vectors));

                    for (int i = 0; i < dim; i++)
                    {
                        double v = Convert.ToDouble(arr[i]);
                        if (v < min![i]) min[i] = v;
                        if (v > max![i]) max[i] = v;
                    }
                }

                count++;
            }

            if (count == 0)
                throw new ArgumentException("Training set must contain at least one vector.", nameof(vectors));

            _dimension = dim;
            _min = min;
            _scale = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                double range = max![i] - min![i];
                // Guard against zero-range (constant) dimensions to avoid division by zero.
                _scale[i] = range > 0.0 ? range / (Levels - 1) : 0.0;
            }
        }

        /// <inheritdoc/>
        public byte[] Encode(Vector<T> vector)
        {
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));
            if (!IsTrained)
                throw new InvalidOperationException("ScalarQuantizer must be trained before encoding.");

            var arr = vector.ToArray();
            if (arr.Length != _dimension)
                throw new ArgumentException("Vector dimensionality does not match the trained dimensionality.", nameof(vector));

            var code = new byte[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                double v = Convert.ToDouble(arr[i]);
                int level;
                if (_scale![i] <= 0.0)
                {
                    level = 0;
                }
                else
                {
                    double normalized = (v - _min![i]) / _scale[i];
                    level = (int)Math.Round(normalized);
                    if (level < 0) level = 0;
                    if (level > Levels - 1) level = Levels - 1;
                }

                code[i] = (byte)level;
            }

            return code;
        }

        /// <inheritdoc/>
        public Vector<T> Decode(byte[] code)
        {
            if (code == null)
                throw new ArgumentNullException(nameof(code));
            if (!IsTrained)
                throw new InvalidOperationException("ScalarQuantizer must be trained before decoding.");
            if (code.Length != _dimension)
                throw new ArgumentException("Code length does not match the trained dimensionality.", nameof(code));

            var values = new T[_dimension];
            for (int i = 0; i < _dimension; i++)
            {
                double reconstructed = _min![i] + code[i] * _scale![i];
                values[i] = _numOps.FromDouble(reconstructed);
            }

            return new Vector<T>(values);
        }
    }
}
