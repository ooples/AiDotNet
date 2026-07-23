using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    public class BinaryQuantizerTests
    {
        [Fact]
        public void Encode_PacksOneBitPerDimension()
        {
            int dim = 20; // not a multiple of 8 -> ceil(20/8) = 3 bytes
            var q = new BinaryQuantizer<double>();
            q.Train(new List<Vector<double>> { new Vector<double>(new double[dim]) });

            var code = q.Encode(new Vector<double>(new double[dim]));

            Assert.Equal(3, code.Length);
            Assert.Equal(3, q.CodeLength);
            Assert.Equal(dim, q.Dimension);
        }

        [Fact]
        public void EncodeDecode_RoundTrip_PreservesSign()
        {
            var values = new double[] { 1.5, -2.0, 0.0, -0.1, 3.3, -7.0, 8.8, -0.5 };
            var q = new BinaryQuantizer<double>();
            var code = q.Encode(new Vector<double>(values));
            var decoded = q.Decode(code).ToArray();

            for (int i = 0; i < values.Length; i++)
            {
                // Non-negative (>= 0) maps to +1, negative maps to -1.
                double expectedSign = values[i] >= 0.0 ? 1.0 : -1.0;
                Assert.Equal(expectedSign, decoded[i], 9);
            }
        }

        [Fact]
        public void Memory_IsAtLeast32xSmallerThanRawFloat()
        {
            int dim = 128;
            var q = new BinaryQuantizer<double>();
            var code = q.Encode(new Vector<double>(new double[dim]));

            int rawFloatBytes = dim * sizeof(float);  // 4 bytes per component
            // 1 bit per dim -> ceil(dim/8) bytes; must be >= 32x smaller than float32.
            Assert.True(code.Length * 32 <= rawFloatBytes, $"codeBytes={code.Length}, rawFloatBytes={rawFloatBytes}");
        }

        [Fact]
        public void HammingDistance_CountsDifferingBits()
        {
            var q = new BinaryQuantizer<double>();
            // signs:  + - + - + - + -
            var a = q.Encode(new Vector<double>(new double[] { 1, -1, 1, -1, 1, -1, 1, -1 }));
            // signs:  + + + + - - - -   (differs in positions 1,3,4,6 -> 4 bits)
            var b = q.Encode(new Vector<double>(new double[] { 1, 1, 1, 1, -1, -1, -1, -1 }));

            Assert.Equal(4, BinaryQuantizer<double>.HammingDistance(a, b));
            Assert.Equal(0, BinaryQuantizer<double>.HammingDistance(a, a));
        }

        [Fact]
        public void HammingDistance_WithMismatchedLengths_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
                BinaryQuantizer<double>.HammingDistance(new byte[2], new byte[3]));
        }
    }
}
