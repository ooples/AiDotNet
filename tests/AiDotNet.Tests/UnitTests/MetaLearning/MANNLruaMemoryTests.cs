using System;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// MANN's external memory against a scalar transcription of the least-recently-used access module of Santoro et al.
/// 2016 (section 3.2): least-used weights from the previous usage, the gated write weights, zeroing of the least-used
/// slot before the write, additive writes, cosine-softmax reads and the decayed usage update.
/// </summary>
public class MANNLruaMemoryTests
{
    private const int Slots = 4;
    private const int KeyWidth = 3;
    private const int ValueWidth = 2;
    private const double Gamma = 0.9;
    private const double Gate = 0.3;

    /// <summary>The paper's equations, one slot and one entry at a time.</summary>
    private sealed class Reference
    {
        public Reference(double[][] keys)
        {
            Keys = keys;
            Values = Enumerable.Range(0, Slots).Select(_ => new double[ValueWidth]).ToArray();
        }

        public double[][] Keys { get; }

        public double[][] Values { get; }

        public double[] Usage { get; private set; } = new double[Slots];

        public double[] PreviousRead { get; private set; } = new double[Slots];

        /// <summary>Slots ordered by ascending usage, the lower index first among equal usages.</summary>
        private int[] LeastUsedOrder() => Enumerable.Range(0, Slots).OrderBy(i => Usage[i]).ThenBy(i => i).ToArray();

        public double[] Step(double[] key, double[] value)
        {
            int leastUsed = LeastUsedOrder()[0];
            var write = new double[Slots];
            for (int i = 0; i < Slots; i++)
            {
                double lu = i == leastUsed ? 1.0 : 0.0;
                write[i] = Gate * PreviousRead[i] + (1.0 - Gate) * lu;
            }

            Array.Clear(Keys[leastUsed], 0, KeyWidth);
            Array.Clear(Values[leastUsed], 0, ValueWidth);
            for (int i = 0; i < Slots; i++)
            {
                for (int j = 0; j < KeyWidth; j++) Keys[i][j] += write[i] * key[j];
                for (int j = 0; j < ValueWidth; j++) Values[i][j] += write[i] * value[j];
            }

            var read = ReadWeights(key);
            Usage = Enumerable.Range(0, Slots).Select(i => Gamma * Usage[i] + read[i] + write[i]).ToArray();
            PreviousRead = read;
            return Content(read);
        }

        public double[] ReadOnly(double[] key)
        {
            var read = ReadWeights(key);
            Usage = Enumerable.Range(0, Slots).Select(i => Gamma * Usage[i] + read[i]).ToArray();
            PreviousRead = read;
            return Content(read);
        }

        private double[] ReadWeights(double[] key)
        {
            var similarity = new double[Slots];
            for (int i = 0; i < Slots; i++)
            {
                double dot = 0, keyNorm = 0, slotNorm = 0;
                for (int j = 0; j < KeyWidth; j++)
                {
                    dot += key[j] * Keys[i][j];
                    keyNorm += key[j] * key[j];
                    slotNorm += Keys[i][j] * Keys[i][j];
                }

                double denominator = Math.Sqrt(keyNorm * slotNorm);
                similarity[i] = denominator > 1e-10 ? Math.Max(-1.0, Math.Min(1.0, dot / denominator)) : 0.0;
            }

            double max = similarity.Max();
            var exp = similarity.Select(x => Math.Exp(x - max)).ToArray();
            double sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        private double[] Content(double[] read)
        {
            var content = new double[ValueWidth];
            for (int i = 0; i < Slots; i++)
            {
                for (int j = 0; j < ValueWidth; j++) content[j] += read[i] * Values[i][j];
            }

            return content;
        }
    }

    private static ExternalMemory<double> Memory(int numReads = 1)
        => new ExternalMemory<double>(Slots, KeyWidth, ValueWidth, MathHelper.GetNumericOperations<double>(),
            Gamma, Gate, numReads);

    private static void AssertClose(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-10,
                $"{what}[{i}]: expected {expected[i]:G12}, got {actual[i]:G12}.");
        }
    }

    [Fact]
    public void WritesAndReads_FollowTheLeastRecentlyUsedAccessEquations()
    {
        var memory = Memory();
        var reference = new Reference(Enumerable.Range(0, Slots).Select(i => memory.GetKey(i).ToArray()).ToArray());

        var steps = new[]
        {
            (Key: new[] { 1.0, 0.5, -0.2 }, Value: new[] { 1.0, 0.0 }),
            (Key: new[] { -0.3, 0.8, 0.1 }, Value: new[] { 0.0, 1.0 }),
            (Key: new[] { 0.9, 0.4, -0.1 }, Value: new[] { 1.0, 0.0 }),
            (Key: new[] { 0.2, -0.7, 0.5 }, Value: new[] { 0.0, 1.0 }),
            (Key: new[] { -0.6, 0.1, 0.9 }, Value: new[] { 1.0, 0.0 }),
        };

        for (int t = 0; t < steps.Length; t++)
        {
            var expected = reference.Step(steps[t].Key, steps[t].Value);
            var actual = memory.WriteAndRead(new Vector<double>(steps[t].Key), new Vector<double>(steps[t].Value));

            AssertClose(expected, actual.ToArray(), $"step {t} read");
            AssertClose(reference.Usage, memory.GetUsage().ToArray(), $"step {t} usage");
            for (int i = 0; i < Slots; i++)
            {
                AssertClose(reference.Keys[i], memory.GetKey(i).ToArray(), $"step {t} slot {i} key");
            }
        }

        var query = new[] { 0.95, 0.45, -0.15 };
        AssertClose(reference.ReadOnly(query), memory.ReadAndUpdate(new Vector<double>(query)).ToArray(), "query read");
        AssertClose(reference.Usage, memory.GetUsage().ToArray(), "query usage");
    }

    [Fact]
    public void LeastUsedWeights_MarkOneSlotPerRead_TheLowerIndexAmongEqualUsages()
    {
        // A fresh memory has equal (zero) usage everywhere: the n least used are the first n slots.
        Assert.Equal(new[] { 1.0, 1.0, 0.0, 0.0 }, Memory(numReads: 2).LeastUsedWeights().ToArray());

        var memory = Memory(numReads: 1);
        Assert.Equal(0, memory.FindLeastRecentlyUsedSlot());

        // The first write lands on slot 0 (and every read adds usage), so slot 0 is no longer the least used.
        memory.WriteAndRead(new Vector<double>(new[] { 1.0, 0.0, 0.0 }), new Vector<double>(new[] { 1.0, 0.0 }));
        var usage = memory.GetUsage().ToArray();
        int expected = Enumerable.Range(0, Slots).OrderBy(i => usage[i]).ThenBy(i => i).First();
        Assert.NotEqual(0, expected);
        Assert.Equal(expected, memory.FindLeastRecentlyUsedSlot());
        Assert.Equal(1.0, memory.LeastUsedWeights()[expected]);
    }
}
