using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.FederatedLearning.Unlearning;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Half-precision ranges of the federated aggregation, server optimizer, influence solve and compression, and the
/// unlearning noise source. Half's largest finite value is 65504 and its smallest positive one about 6e-8, so squares,
/// small weights and rounded thresholds formed in Half leave the range (or move) where the double reference does not.
/// </summary>
public class FederatedHalfRangeReviewTests
{
    private static readonly INumericOperations<Half> HalfOps = MathHelper.GetNumericOperations<Half>();

    private static Half H(double value) => HalfOps.FromDouble(value);

    private static double D(Half value) => HalfOps.ToDouble(value);

    [Fact]
    public void FedYogi_InHalf_FollowsTheDoubleReferenceWhenTheSquaredDeltaOverflows()
    {
        // lr 1, b1 0.9, b2 0.99, tau 1e-3, x0 = 0 and target 300: g^2 = 90000 is past Half's 65504. Held in Half,
        // v became infinite and every step was 0. Double reference: x1 = 30 / (sqrt(900.000001) + 1e-3) = 0.9999667,
        // which Half stores as 1; then g = 299, m = 56.9, v = 1794.01 and x2 = 1 + 56.9 / (sqrt(v) + 1e-3) = 2.343351.
        var optimizer = new FedYogiServerOptimizer<Half>(learningRate: 1.0, beta1: 0.9, beta2: 0.99, epsilon: 1e-3);
        var target = new Vector<Half>(new[] { H(300) });

        var first = optimizer.Step(new Vector<Half>(new[] { H(0) }), target);
        Assert.Equal(0.9999667, D(first[0]), 3);

        var second = optimizer.Step(first, target);
        Assert.True(Math.Abs(D(second[0]) - 2.343351) < 2e-3, $"Expected 2.343351, got {D(second[0])}.");
    }

    [Fact]
    public void AsyncFedED_InHalf_KeepsAClientWhoseDecayedWeightUnderflows()
    {
        // Both clients last took part in round 0 and are aggregated in round 10 with a staleness decay of 0.01, so each
        // weight is ln 2 * 1e-20: zero in Half. Weighting in Half summed zeros and divided by a zero total (NaN);
        // the weights normalized in double are 1/2 each, so the mean is 5.
        var trainer = new AsyncFedEDTrainer<Half>(stalenessDecay: 0.01);
        trainer.UpdateClientEntropy(0, new[] { 1.0, 1.0 }, currentRound: 0);
        trainer.UpdateClientEntropy(1, new[] { 1.0, 1.0 }, currentRound: 0);
        var updates = new Dictionary<int, Dictionary<string, Half[]>>
        {
            [0] = new Dictionary<string, Half[]> { ["w"] = new[] { H(4), H(-2) } },
            [1] = new Dictionary<string, Half[]> { ["w"] = new[] { H(6), H(2) } },
        };

        var merged = trainer.AggregateWithEntropyWeights(updates, currentRound: 10);

        Assert.Equal(5.0, D(merged["w"][0]), 6);
        Assert.Equal(0.0, D(merged["w"][1]), 6);
    }

    [Fact]
    public void InfluenceSolve_InHalf_KeepsTheInnerProductsInRange()
    {
        // H = 100 I and v = [300, 400]: r.r = 250000 and p.Hp = 2.5e7 overflow Half, and alpha = inf / inf was NaN.
        // With the inner products in double, alpha = 0.01 and one step solves H x = v: x = [3, 4].
        var engine = AiDotNetEngine.Current;
        Tensor<Half> Hessian(Tensor<Half> p) => engine.TensorMultiplyScalar(p, H(100));

        var x = InfluenceFunctionUnlearner<Half>.SolveDamped(Hessian, new Tensor<Half>(new[] { H(300), H(400) }, new[] { 2 }), 0.0, 10, 1e-3);

        Assert.True(Math.Abs(D(x[0]) - 3.0) < 1e-2, $"Expected 3, got {D(x[0])}.");
        Assert.True(Math.Abs(D(x[1]) - 4.0) < 1e-2, $"Expected 4, got {D(x[1])}.");
    }

    [Fact]
    public void ThresholdSparsify_InHalf_ComparesAgainstTheExactThreshold()
    {
        // Half holds 0.2 as 0.199951171875, below the threshold 0.2; against the threshold rounded to Half it was kept.
        // (0.3 would not do: Half rounds it up, to 0.300048828125.)
        Assert.Equal(0.199951171875, D(H(0.2)));
        var delta = new Vector<Half>(new[] { H(0.2), H(0.5), H(-0.1), H(-0.75) });

        var sparse = InMemoryFederatedTrainer<Half, Matrix<Half>, Vector<Half>>.ThresholdSparsify(delta, 0.2, out int kept);

        Assert.Equal(2, kept);
        Assert.Equal(new[] { 0.0, 0.5, 0.0, -0.75 }, sparse.ToArray().Select(D).ToArray());
    }

    private static byte[] Words(params int[] words)
    {
        var bytes = new byte[words.Length * 4];
        Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    // The mapping written out with integer arithmetic: 32 bits of the first word and the top 21 of the second as a
    // 53-bit integer n, u1 = (n + 1) / 2^53 and u2 = n / 2^53.
    private static double[] ScalarNormals(int[] words, int count)
    {
        // A word counts as its signed value plus 2^31: the unsigned word with its top bit flipped.
        uint Unsigned(int index) => unchecked((uint)words[index]) ^ 0x80000000u;
        ulong Integer53(int high, int low) => ((ulong)Unsigned(high) << 21) | (Unsigned(low) >> 11);
        var normals = new double[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = (Integer53(i, count + i) + 1) / 9007199254740992.0;
            double u2 = Integer53(2 * count + i, 3 * count + i) / 9007199254740992.0;
            normals[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        return normals;
    }

    private static int[] PseudoWords(int count, int seed)
    {
        var random = new Random(seed);
        var words = new int[count];
        for (int i = 0; i < count; i++) words[i] = unchecked((int)(((uint)random.Next(1 << 16) << 16) | (uint)random.Next(1 << 16)));
        return words;
    }

    [Fact]
    public void StandardNormalsFromBytes_IsTheBoxMullerTransformOfTheWords()
    {
        // Zero words count as 2^31 each, so u1 and u2 are just above 1/2 ((2^52 + 2^20 + 1) / 2^53 and
        // (2^52 + 2^20) / 2^53): sqrt(-2 ln u1) cos(2 pi u2) = -sqrt(2 ln 2).
        var normals = UnlearningEngineOps.StandardNormalsFromBytes<double>(new byte[16], 1);

        Assert.Equal(-Math.Sqrt(2.0 * Math.Log(2.0)), normals[0], 6);
    }

    [Fact]
    public void StandardNormalsFromBytes_ForAnOddCount_MatchesTheMappingInIntegerArithmetic()
    {
        const int count = 7;
        var words = PseudoWords(4 * count, seed: 3);

        var normals = UnlearningEngineOps.StandardNormalsFromBytes<double>(Words(words), count).ToArray();

        var expected = ScalarNormals(words, count);
        Assert.Equal(count, normals.Length);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(expected[i], normals[i], 12);
        }
    }

    [Fact]
    public void StandardNormalsFromBytes_ReachesTheTailOfA53BitUniformAndStaysFinite()
    {
        // The smallest u1 (both of its words counting 0) is 2^-53: with u2 = 0, z = 8.5716, past the 6.7637 where a
        // 32-bit uniform stops. The largest (both words all ones) is exactly 1, whose logarithm is 0: z = 0, not NaN.
        // Sample 2 has the smallest u1 at the angle pi (u2 = 1/2): z = -8.5716.
        const int zero = int.MinValue, ones = int.MaxValue, half = 0;
        var bytes = Words(
            zero, ones, zero,      // u1 high
            zero, ones, zero,      // u1 low
            zero, zero, half,      // u2 high
            zero, zero, zero);     // u2 low
        double limit = Math.Sqrt(106.0 * Math.Log(2.0));

        var full = UnlearningEngineOps.StandardNormalsFromBytes<double>(bytes, 3).ToArray();
        Assert.Equal(limit, full[0], 12);
        Assert.True(full[0] > 6.77, $"largest sample {full[0]}");
        Assert.Equal(0.0, full[1], 15);
        Assert.Equal(-limit, full[2], 12);

        var single = UnlearningEngineOps.StandardNormalsFromBytes<float>(bytes, 3).ToArray();
        var narrow = UnlearningEngineOps.StandardNormalsFromBytes<Half>(bytes, 3).ToArray();
        Assert.All(full, z => Assert.True(!double.IsNaN(z) && !double.IsInfinity(z)));
        Assert.All(single, z => Assert.True(!float.IsNaN(z) && !float.IsInfinity(z)));
        Assert.All(narrow, z => Assert.True(!Half.IsNaN(z) && !Half.IsInfinity(z)));
        Assert.True(Math.Abs(D(narrow[0]) - limit) <= 0.01, $"Half tail sample {narrow[0]}");
    }

    [Fact]
    public void StandardNormals_DrawnInChunks_AskForEachChunksBytesAndJoinTheChunks()
    {
        // 11 samples, 4 at a time: chunks of 4, 4 and 3 samples, 64, 64 and 48 bytes. The source hands out a running
        // byte sequence, so the result is the three chunks' transforms one after the other.
        const int count = 11, perChunk = 4;
        var stream = Words(PseudoWords(4 * count, seed: 9));
        int position = 0;
        var requests = new List<int>();
        void Fill(byte[] buffer)
        {
            requests.Add(buffer.Length);
            Array.Copy(stream, position, buffer, 0, buffer.Length);
            position += buffer.Length;
        }

        var normals = UnlearningEngineOps.StandardNormals<double>(count, Fill, perChunk).ToArray();

        Assert.Equal(new[] { 64, 64, 48 }, requests);
        Assert.Equal(stream.Length, position);
        var expected = new List<double>();
        int offset = 0;
        foreach (int take in new[] { 4, 4, 3 })
        {
            var chunk = new byte[take * 16];
            Array.Copy(stream, offset, chunk, 0, chunk.Length);
            offset += chunk.Length;
            expected.AddRange(UnlearningEngineOps.StandardNormalsFromBytes<double>(chunk, take).ToArray());
        }

        Assert.Equal(expected.ToArray(), normals);
        Assert.Equal(count, normals.Distinct().Count());
    }

    [Fact]
    public void StandardNormals_RejectBadArgumentsAndReturnNothingForZero()
    {
        Assert.Equal(0, UnlearningEngineOps.SecureStandardNormals<float>(0).Length);
        Assert.Equal(0, UnlearningEngineOps.StandardNormalsFromBytes<float>(Array.Empty<byte>(), 0).Length);
        Assert.Throws<ArgumentOutOfRangeException>(() => UnlearningEngineOps.SecureStandardNormals<float>(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnlearningEngineOps.StandardNormals<float>(4, _ => { }, 0));
        Assert.Throws<ArgumentNullException>(() => UnlearningEngineOps.StandardNormals<float>(4, null!, 4));
        Assert.Throws<ArgumentException>(() => UnlearningEngineOps.StandardNormalsFromBytes<float>(new byte[31], 2));
    }

    [Fact]
    public void RandomBytesFor_CountsInLongPastTheIntRange()
    {
        // 300 million parameters need 4.8e9 random bytes: past int.MaxValue (an int product wraps), and never
        // allocated at once. Only a count whose bytes are past long's range is refused.
        Assert.Equal(16, UnlearningEngineOps.RandomBytesPerNormal);
        Assert.Equal(0L, UnlearningEngineOps.RandomBytesFor(0));
        Assert.Equal(4_800_000_000L, UnlearningEngineOps.RandomBytesFor(300_000_000));
        Assert.True(UnlearningEngineOps.RandomBytesFor(300_000_000) > int.MaxValue);
        Assert.Equal(16L * int.MaxValue, UnlearningEngineOps.RandomBytesFor(int.MaxValue));
        Assert.True(UnlearningEngineOps.RandomBytesFor(UnlearningEngineOps.NormalsPerChunk) <= int.MaxValue);
        Assert.Throws<OverflowException>(() => UnlearningEngineOps.RandomBytesFor(long.MaxValue / 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnlearningEngineOps.RandomBytesFor(-1));
    }

    [Fact]
    public void SecureStandardNormals_AcrossSeveralChunks_HaveUnitMomentsAndDoNotRepeat()
    {
        // An odd count over three full chunks and a partial one.
        const int count = 3 * 65536 + 4321;
        Assert.True(count > UnlearningEngineOps.NormalsPerChunk && count % 2 == 1);
        var first = UnlearningEngineOps.SecureStandardNormals<double>(count).ToArray();
        var second = UnlearningEngineOps.SecureStandardNormals<double>(count).ToArray();

        Assert.Equal(count, first.Length);
        Assert.All(first, z => Assert.True(!double.IsNaN(z) && !double.IsInfinity(z)));
        double mean = first.Average();
        double variance = first.Select(z => (z - mean) * (z - mean)).Average();
        double fourth = first.Select(z => Math.Pow(z - mean, 4)).Average() / (variance * variance);

        // Standard errors at n = 200929: 0.0022 for the mean, 0.0032 for the variance, 0.011 for the kurtosis (3).
        Assert.InRange(mean, -0.02, 0.02);
        Assert.InRange(variance, 0.97, 1.03);
        Assert.InRange(fourth, 2.9, 3.1);
        Assert.False(first.SequenceEqual(second));
        // No chunk repeats the one before it.
        Assert.False(first.Take(65536).SequenceEqual(first.Skip(65536).Take(65536)));
        Assert.Equal(count, first.Distinct().Count());

        var narrow = UnlearningEngineOps.SecureStandardNormals<Half>(70001).ToArray();
        Assert.Equal(70001, narrow.Length);
        Assert.All(narrow, z => Assert.True(!Half.IsNaN(z) && !Half.IsInfinity(z)));
    }
}
