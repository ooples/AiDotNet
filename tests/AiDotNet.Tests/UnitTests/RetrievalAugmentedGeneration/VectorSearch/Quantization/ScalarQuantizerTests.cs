using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Quantization;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.VectorSearch.Quantization
{
    public class ScalarQuantizerTests
    {
        private static List<Vector<double>> BuildTrainingSet(int count, int dim, double low, double high, int seed)
        {
            var rng = RandomHelper.CreateSeededRandom(seed);
            var result = new List<Vector<double>>();
            for (int n = 0; n < count; n++)
            {
                var arr = new double[dim];
                for (int i = 0; i < dim; i++)
                    arr[i] = low + rng.NextDouble() * (high - low);
                result.Add(new Vector<double>(arr));
            }

            return result;
        }

        [Fact]
        public void Train_WithEmptySet_Throws()
        {
            var q = new ScalarQuantizer<double>();
            Assert.Throws<ArgumentException>(() => q.Train(new List<Vector<double>>()));
        }

        [Fact]
        public void Encode_BeforeTraining_Throws()
        {
            var q = new ScalarQuantizer<double>();
            Assert.Throws<InvalidOperationException>(() => q.Encode(new Vector<double>(new double[] { 1, 2, 3 })));
        }

        [Fact]
        public void Encode_ProducesOneBytePerDimension()
        {
            int dim = 16;
            var data = BuildTrainingSet(200, dim, -5.0, 5.0, 1);
            var q = new ScalarQuantizer<double>();
            q.Train(data);

            var code = q.Encode(data[0]);

            Assert.Equal(dim, code.Length);
            Assert.Equal(dim, q.CodeLength);
            Assert.Equal(dim, q.Dimension);
        }

        [Fact]
        public void EncodeDecode_RoundTrip_WithinBucketError()
        {
            int dim = 32;
            double low = 0.0, high = 10.0;
            var data = BuildTrainingSet(500, dim, low, high, 2);
            var q = new ScalarQuantizer<double>();
            q.Train(data);

            double bucketWidth = (high - low) / 255.0;

            foreach (var vector in data.Take(20))
            {
                var decoded = q.Decode(q.Encode(vector));
                var original = vector.ToArray();
                var restored = decoded.ToArray();
                for (int i = 0; i < dim; i++)
                {
                    // Reconstruction error must be within one quantization bucket.
                    Assert.True(Math.Abs(original[i] - restored[i]) <= bucketWidth + 1e-9,
                        $"dim {i}: |{original[i]} - {restored[i]}| exceeded bucket width {bucketWidth}");
                }
            }
        }

        [Fact]
        public void Memory_IsAtLeastFourXSmallerThanRaw()
        {
            int dim = 64;
            var data = BuildTrainingSet(100, dim, -1.0, 1.0, 3);
            var q = new ScalarQuantizer<double>();
            q.Train(data);

            int rawBytes = dim * sizeof(double);   // 8 bytes per component
            int codeBytes = q.Encode(data[0]).Length;

            // uint8 codes are >= 4x smaller than 32-bit floats (>= 8x vs doubles).
            Assert.True(codeBytes * 4 <= rawBytes, $"codeBytes={codeBytes}, rawBytes={rawBytes}");
        }

        [Fact]
        public void ConstantDimension_DoesNotDivideByZero()
        {
            // All vectors share the same value in dim 0 (zero range).
            var data = new List<Vector<double>>();
            for (int n = 0; n < 10; n++)
                data.Add(new Vector<double>(new double[] { 7.0, n * 0.1 }));

            var q = new ScalarQuantizer<double>();
            q.Train(data);

            var decoded = q.Decode(q.Encode(data[5])).ToArray();
            Assert.Equal(7.0, decoded[0], 6);
        }
    }
}
