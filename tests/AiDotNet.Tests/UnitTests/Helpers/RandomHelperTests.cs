using System;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class RandomHelperTests
    {
        [Fact]
        public void CreateSeededRandom_CoversLockedRandomOverrides()
        {
            var rng = RandomHelper.CreateSeededRandom(123);

            _ = rng.Next();
            _ = rng.Next(10);
            _ = rng.Next(1, 10);
            _ = rng.NextDouble();

            var bytes = Enumerable.Repeat((byte)0xA5, 32).ToArray();
            rng.NextBytes(bytes);
            Assert.True(bytes.Any(b => b != 0xA5), "NextBytes should fill buffer with random data.");

#if NET8_0_OR_GREATER
            _ = rng.NextInt64();
            _ = rng.NextInt64(10);
            _ = rng.NextInt64(1, 10);
            _ = rng.NextSingle();

            Span<byte> span = stackalloc byte[16];
            span.Fill(0xA5);
            rng.NextBytes(span);
            Assert.True(span.ToArray().Any(b => b != 0xA5), "NextBytes should fill span with random data.");
#endif
        }

        [Fact]
        public void CreateSeededRandom_WithSameSeed_IsDeterministic()
        {
            var a = RandomHelper.CreateSeededRandom(42);
            var b = RandomHelper.CreateSeededRandom(42);

            var expected = new[] { a.Next(), a.Next(), a.Next() };
            var actual = new[] { b.Next(), b.Next(), b.Next() };

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void CreateSecureRandom_ReturnsUsableGenerator()
        {
            var rng = RandomHelper.CreateSecureRandom();

            _ = rng.Next();
            _ = rng.NextDouble();
        }
    }
}
