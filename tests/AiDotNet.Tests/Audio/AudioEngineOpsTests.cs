using System;
using System.Threading.Tasks;
using AiDotNet.Audio;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio
{
    public class AudioEngineOpsTests
    {
        /// <summary>
        /// Wav2Vec2 normalization of a Half clip longer than Half.MaxValue (65,504) samples: the count
        /// and the raw sums do not fit in Half, so the mean and std must be formed without them. The
        /// result must match a double reference computed from the same Half-rounded samples.
        /// </summary>
        [Fact(Timeout = 120000)]
        public async Task Standardize_HalfClipLongerThanHalfMax_MatchesDoubleReference()
        {
            await Task.Yield();
            const int length = 70000;
            var half = MathHelper.GetNumericOperations<Half>();
            var samples = new Half[length];
            var reference = new double[length];
            for (int i = 0; i < length; i++)
            {
                samples[i] = half.FromDouble(0.5 * Math.Sin(i * 0.01) + 0.25);
                reference[i] = half.ToDouble(samples[i]);
            }

            double mean = 0;
            for (int i = 0; i < length; i++) mean += reference[i];
            mean /= length;
            double sumSq = 0;
            for (int i = 0; i < length; i++) sumSq += (reference[i] - mean) * (reference[i] - mean);
            double std = Math.Max(Math.Sqrt(sumSq / length), 1e-7);

            var result = AudioEngineOps.Standardize(
                AiDotNetEngine.Current, new Tensor<Half>(samples, [length]), 1e-7).ToArray();

            Assert.Equal(length, result.Length);
            for (int i = 0; i < length; i++)
            {
                double expected = (reference[i] - mean) / std;
                double actual = half.ToDouble(result[i]);
                Assert.False(double.IsNaN(actual) || double.IsInfinity(actual), $"sample {i} is {actual}");
                // Half has a 2^-11 unit roundoff; the mean, the deviation, the std and the division each
                // round in Half, so outputs of magnitude up to ~2.2 carry a few 1e-3 of error.
                Assert.True(Math.Abs(actual - expected) <= 2e-2, $"sample {i}: expected {expected}, got {actual}");
            }
        }
    }
}
