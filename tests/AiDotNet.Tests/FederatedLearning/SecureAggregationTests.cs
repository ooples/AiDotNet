using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Bonawitz et al. (2017), "Practical Secure Aggregation for Privacy-Preserving Machine Learning" (CCS 2017):
/// the masked inputs must sum, modulo R = 2^b, to exactly the sum of the survivors' fixed-point encodings, with or
/// without dropouts, and the protocol must abort below the threshold.
/// </summary>
public class SecureAggregationTests
{
    /// <summary>Which round each dropped client leaves after.</summary>
    private sealed class Dropouts
    {
        public HashSet<int> AfterAdvertiseKeys { get; } = new();
        public HashSet<int> AfterShareKeys { get; } = new();
        public HashSet<int> AfterMaskedInput { get; } = new();
    }

    private static double[][] SampleInputs(int clients, int length, int seed, double magnitude)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var inputs = new double[clients][];
        for (int u = 0; u < clients; u++)
        {
            inputs[u] = new double[length];
            for (int i = 0; i < length; i++)
            {
                inputs[u][i] = (random.NextDouble() * 2.0 - 1.0) * magnitude;
            }
        }

        return inputs;
    }

    /// <summary>The reference: sum of the fixed-point encodings of the given clients, modulo R.</summary>
    private static ulong[] ExpectedRingSum(FixedPointRingEncoding encoding, double[][] inputs, IEnumerable<int> clients)
    {
        var sum = new ulong[inputs[0].Length];
        foreach (var u in clients)
        {
            for (int i = 0; i < sum.Length; i++)
            {
                sum[i] = encoding.Add(sum[i], encoding.Encode(inputs[u][i]));
            }
        }

        return sum;
    }

    /// <summary>Runs rounds 0, 1, 2 and 4 with client ids 10, 20, 30, ... and the given dropouts.</summary>
    private static ulong[] RunProtocol(
        FixedPointRingEncoding encoding,
        int threshold,
        double[][] inputs,
        Dropouts dropouts,
        Func<IReadOnlyDictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>>, IReadOnlyDictionary<int, IReadOnlyList<SecureAggregationEncryptedShare>>>? tamper = null)
    {
        int n = inputs.Length;
        int m = inputs[0].Length;
        var ids = Enumerable.Range(0, n).Select(u => 10 * (u + 1)).ToArray();
        var clients = ids.ToDictionary(id => id, id => new SecureAggregationClient<double>(id, m, threshold, encoding));
        var server = new SecureAggregationServer<double>(m, threshold, encoding);

        var u1 = server.CollectAdvertisedKeys(ids.Select(id => clients[id].AdvertiseKeys()).ToArray());

        var alive1 = ids.Where(id => !dropouts.AfterAdvertiseKeys.Contains(id / 10 - 1)).ToArray();
        var inboxes = server.CollectShares(alive1.SelectMany(id => clients[id].ShareKeys(u1)).ToArray());
        if (tamper is not null)
        {
            inboxes = tamper(inboxes);
        }

        var alive2 = alive1.Where(id => !dropouts.AfterShareKeys.Contains(id / 10 - 1)).ToArray();
        var masked = alive2
            .Select(id => clients[id].MaskInput(new Vector<double>(inputs[id / 10 - 1]), inboxes[id]))
            .ToArray();
        var u3 = server.CollectMaskedInputs(masked);

        var alive3 = u3.Where(id => !dropouts.AfterMaskedInput.Contains(id / 10 - 1)).ToArray();
        var responses = alive3.Select(id => clients[id].Unmask(u3)).ToArray();
        return server.AggregateRingSum(responses);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskedSum_DecodesToTheExactFixedPointSum_WithoutDropout()
    {
        await Task.Yield();
        var encoding = new FixedPointRingEncoding();
        var inputs = SampleInputs(clients: 5, length: 257, seed: 2017, magnitude: 3.0);

        var ring = RunProtocol(encoding, threshold: 4, inputs, new Dropouts());

        var expected = ExpectedRingSum(encoding, inputs, Enumerable.Range(0, 5));
        Assert.Equal(expected, ring);
        for (int i = 0; i < ring.Length; i++)
        {
            long exactFixedPoint = Enumerable.Range(0, 5).Sum(u => encoding.ToSignedInteger(encoding.Encode(inputs[u][i])));
            Assert.Equal(exactFixedPoint / encoding.Scale, encoding.Decode(ring[i]));
        }
    }

    [Theory(Timeout = 120000)]
    [InlineData(new int[0], new int[0], new[] { 2, 5 })]
    [InlineData(new int[0], new[] { 0, 6 }, new int[0])]
    [InlineData(new[] { 1, 3 }, new int[0], new int[0])]
    [InlineData(new int[0], new[] { 4 }, new[] { 0 })]
    [InlineData(new[] { 6 }, new[] { 2 }, new int[0])]
    [InlineData(new[] { 0 }, new int[0], new[] { 3 })]
    public async Task MaskedSum_IsBitExact_WithDropoutsUpToNMinusT(int[] afterAdvertise, int[] afterShareKeys, int[] afterMaskedInput)
    {
        await Task.Yield();
        // n = 7, t = 5: up to n - t = 2 clients may drop, at any round.
        var encoding = new FixedPointRingEncoding();
        var inputs = SampleInputs(clients: 7, length: 64, seed: 7, magnitude: 100.0);
        var dropouts = new Dropouts();
        dropouts.AfterAdvertiseKeys.UnionWith(afterAdvertise);
        dropouts.AfterShareKeys.UnionWith(afterShareKeys);
        dropouts.AfterMaskedInput.UnionWith(afterMaskedInput);

        var ring = RunProtocol(encoding, threshold: 5, inputs, dropouts);

        // The aggregate is over U_3: every client that sent a masked input, including those that then dropped.
        var u3 = Enumerable.Range(0, 7).Where(u => !afterAdvertise.Contains(u) && !afterShareKeys.Contains(u));
        Assert.Equal(ExpectedRingSum(encoding, inputs, u3), ring);
    }

    [Theory(Timeout = 120000)]
    [InlineData(SecureAggregationRound.AdvertiseKeys)]
    [InlineData(SecureAggregationRound.MaskedInputCollection)]
    [InlineData(SecureAggregationRound.Unmasking)]
    public async Task FewerThanThresholdSurvivors_AbortsCleanly(SecureAggregationRound round)
    {
        await Task.Yield();
        // n = 7, t = 5: three dropouts leave four clients, one short of the threshold.
        var encoding = new FixedPointRingEncoding();
        var inputs = SampleInputs(clients: 7, length: 8, seed: 3, magnitude: 1.0);
        int m = inputs[0].Length;
        var ids = Enumerable.Range(1, 7).ToArray();
        var clients = ids.ToDictionary(id => id, id => new SecureAggregationClient<double>(id, m, 5, encoding));
        var server = new SecureAggregationServer<double>(m, 5, encoding);
        var dropped = new HashSet<int> { 2, 4, 6 };

        var advertised = ids
            .Where(id => round != SecureAggregationRound.AdvertiseKeys || !dropped.Contains(id))
            .Select(id => clients[id].AdvertiseKeys())
            .ToArray();
        if (round == SecureAggregationRound.AdvertiseKeys)
        {
            var abort = Assert.Throws<SecureAggregationAbortedException>(() => server.CollectAdvertisedKeys(advertised));
            Assert.Equal(SecureAggregationRound.AdvertiseKeys, abort.Round);
            Assert.Equal(SecureAggregationRound.Completed, server.Round);
            return;
        }

        var u1 = server.CollectAdvertisedKeys(advertised);
        var inboxes = server.CollectShares(ids.SelectMany(id => clients[id].ShareKeys(u1)).ToArray());
        var masked = ids
            .Where(id => round != SecureAggregationRound.MaskedInputCollection || !dropped.Contains(id))
            .Select(id => clients[id].MaskInput(new Vector<double>(inputs[id - 1]), inboxes[id]))
            .ToArray();
        if (round == SecureAggregationRound.MaskedInputCollection)
        {
            var abort = Assert.Throws<SecureAggregationAbortedException>(() => server.CollectMaskedInputs(masked));
            Assert.Equal(SecureAggregationRound.MaskedInputCollection, abort.Round);
            return;
        }

        var u3 = server.CollectMaskedInputs(masked);
        var responses = u3.Where(id => !dropped.Contains(id)).Select(id => clients[id].Unmask(u3)).ToArray();
        var unmaskAbort = Assert.Throws<SecureAggregationAbortedException>(() => server.AggregateRingSum(responses));
        Assert.Equal(SecureAggregationRound.Unmasking, unmaskAbort.Round);
        Assert.Equal(SecureAggregationRound.Completed, server.Round);
    }

    [Fact(Timeout = 120000)]
    public async Task FullParticipationMode_AbortsWhenAnyClientDrops()
    {
        await Task.Yield();
        using var aggregation = new SecureAggregation<double>(3, new SecureAggregationOptions { Mode = SecureAggregationMode.FullParticipation });
        aggregation.InitializeRound(new[] { 1, 2, 3 });
        Assert.Equal(3, aggregation.Threshold);

        var masked = new[]
        {
            aggregation.MaskUpdate(1, new Vector<double>(new[] { 1.0, 2.0, 3.0 })),
            aggregation.MaskUpdate(2, new Vector<double>(new[] { 4.0, 5.0, 6.0 })),
        };

        var abort = Assert.Throws<SecureAggregationAbortedException>(() => aggregation.AggregateSum(masked));
        Assert.Equal(SecureAggregationRound.MaskedInputCollection, abort.Round);
    }

    [Fact(Timeout = 120000)]
    public async Task Shamir_SplitThenCombine_RoundTripsForEveryThresholdSubset()
    {
        await Task.Yield();
        var random = RandomHelper.CreateSeededRandom(521);
        var secret = new byte[32];
        random.NextBytes(secret);
        var x = Enumerable.Range(0, 6).ToDictionary(id => 100 + id, id => id + 1);

        var shares = ShamirSecretSharing.SplitSecret(secret, x, threshold: 4, deterministicSeed: null, info: "test");
        Assert.Equal(6, shares.Count);

        var ids = x.Keys.ToArray();
        int subsets = 0;
        for (int mask = 0; mask < (1 << ids.Length); mask++)
        {
            var subset = ids.Where((_, index) => (mask & (1 << index)) != 0).ToArray();
            if (subset.Length < 4)
            {
                continue;
            }

            var provided = subset.ToDictionary(id => id, id => shares[id]);
            Assert.Equal(secret, ShamirSecretSharing.CombineShares(provided, x, threshold: 4, secretLength: 32));
            subsets++;
        }

        Assert.Equal(15 + 6 + 1, subsets);

        var tooFew = ids.Take(3).ToDictionary(id => id, id => shares[id]);
        Assert.Throws<ArgumentException>(() => ShamirSecretSharing.CombineShares(tooFew, x, threshold: 4, secretLength: 32));
    }

    [Theory(Timeout = 120000)]
    [InlineData(64)]
    [InlineData(13)]
    public async Task PairwiseMasks_CancelExactly(int bitWidth)
    {
        await Task.Yield();
        var encoding = new FixedPointRingEncoding(bitWidth, fractionalBits: 0, clipBound: 1.0);
        var prg = new AesCtrSecureAggregationPrg();
        const int m = 1000;
        var ids = new[] { 3, 8, 11, 42, 97 };
        var keys = ids.ToDictionary(id => id, _ => P256KeyAgreement.Generate());

        var total = new ulong[m];
        var scratch = new ulong[m];
        foreach (var u in ids)
        {
            var own = new ulong[m];
            foreach (var v in ids.Where(v => v != u))
            {
                var seed = SecureAggregationProtocol.DerivePairwiseSeed(
                    keys[u].PrivateScalar, keys[u].PublicKey, keys[v].PublicKey, u, v);
                SecureAggregationProtocol.ApplyMask(prg, encoding, seed, own, scratch, subtract: u > v);
            }

            Assert.Contains(own, value => value != 0UL);
            Assert.All(own, value => Assert.Equal(value, value & encoding.ModulusMask));
            encoding.AddInPlace(total, own);
        }

        Assert.All(total, value => Assert.Equal(0UL, value));
    }

    [Fact(Timeout = 120000)]
    public async Task HalfInputs_DecodeToTheExactSumRoundedOnce()
    {
        await Task.Yield();
        var half = MathHelper.GetNumericOperations<Half>();
        var random = RandomHelper.CreateSeededRandom(16);
        const int n = 4;
        const int m = 32;
        var inputs = new Half[n][];
        for (int u = 0; u < n; u++)
        {
            inputs[u] = new Half[m];
            for (int i = 0; i < m; i++)
            {
                inputs[u][i] = half.FromDouble((random.NextDouble() * 2.0 - 1.0) * 1000.0);
            }
        }

        // Every half-precision value is a multiple of 2^-24, so the default encoding is exact and the
        // aggregate must equal the exact real sum of the inputs, rounded once to Half.
        using var aggregation = new SecureAggregation<Half>(m);
        aggregation.InitializeRound(Enumerable.Range(0, n));
        var masked = Enumerable.Range(0, n).Select(u => aggregation.MaskUpdate(u, new Vector<Half>(inputs[u]))).ToArray();
        var sum = aggregation.AggregateSum(masked);

        for (int i = 0; i < m; i++)
        {
            double exact = Enumerable.Range(0, n).Sum(u => half.ToDouble(inputs[u][i]));
            Assert.Equal(half.ToDouble(half.FromDouble(exact)), half.ToDouble(sum[i]));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FloatInputs_DecodeToTheExactSumRoundedOnce()
    {
        await Task.Yield();
        var random = RandomHelper.CreateSeededRandom(32);
        const int n = 5;
        const int m = 40;
        var inputs = new float[n][];
        for (int u = 0; u < n; u++)
        {
            inputs[u] = new float[m];
            for (int i = 0; i < m; i++)
            {
                // Multiples of 2^-20 within +-8: exactly representable in float and in 24 fractional bits.
                inputs[u][i] = (float)(random.Next(-(1 << 23), 1 << 23) / (double)(1 << 20));
            }
        }

        using var aggregation = new SecureAggregation<float>(m, new SecureAggregationOptions { Mode = SecureAggregationMode.ThresholdDropoutResilient });
        aggregation.InitializeRound(Enumerable.Range(0, n));
        Assert.Equal(4, aggregation.Threshold);
        var masked = Enumerable.Range(0, n).Select(u => aggregation.MaskUpdate(u, new Vector<float>(inputs[u]))).ToArray();
        var sum = aggregation.AggregateSum(masked);

        for (int i = 0; i < m; i++)
        {
            double exact = Enumerable.Range(0, n).Sum(u => (double)inputs[u][i]);
            Assert.Equal((float)exact, sum[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task WeightedAverage_DividesTheExactWeightedSumOnce()
    {
        await Task.Yield();
        using var aggregation = new SecureAggregation<double>(2);
        aggregation.InitializeRound(new[] { 10, 20 });
        var weights = new Dictionary<int, double> { [10] = 0.25, [20] = 0.75 };
        var masked = new[]
        {
            aggregation.MaskUpdate(10, new Vector<double>(new[] { 1.0, -2.0 }), weights[10]),
            aggregation.MaskUpdate(20, new Vector<double>(new[] { 3.0, 4.0 }), weights[20]),
        };

        var average = aggregation.AggregateWeightedAverage(masked, weights);

        Assert.Equal(0.25 * 1.0 + 0.75 * 3.0, average[0]);
        Assert.Equal(0.25 * -2.0 + 0.75 * 4.0, average[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task RingEncoding_WrapsAroundCorrectlyAtTheBoundary()
    {
        await Task.Yield();
        var encoding = new FixedPointRingEncoding(bitWidth: 16, fractionalBits: 0, clipBound: 10000.0);
        Assert.Equal(0xFFFFUL, encoding.Encode(-1.0));
        Assert.Equal(0UL, encoding.Add(0xFFFFUL, 1UL));
        Assert.Equal(0xFFFFUL, encoding.Subtract(0UL, 1UL));
        Assert.Equal(-32768L, encoding.ToSignedInteger(0x8000UL));
        Assert.Equal(32767L, encoding.ToSignedInteger(0x7FFFUL));
        Assert.Equal(10000UL, encoding.Encode(1e9));
        Assert.Equal((ulong)(65536 - 10000), encoding.Encode(double.NegativeInfinity));

        // Inputs at the clip bound: each masked element wraps around Z_{2^16} many times, the sum
        // -15000 lives just above the ring boundary at 2^16 - 15000.
        var inputs = new[]
        {
            new[] { -10000.0, 10000.0, -1.0 },
            new[] { -10000.0, 10000.0, -1.0 },
            new[] { 5000.0, 10000.0, 0.0 },
        };
        var ring = RunProtocol(encoding, threshold: 2, inputs, new Dropouts());
        Assert.Equal(new[] { (ulong)(65536 - 15000), 30000UL, (ulong)(65536 - 2) }, ring);
        Assert.Equal(-15000.0, encoding.Decode(ring[0]));
        Assert.Equal(30000.0, encoding.Decode(ring[1]));
        Assert.Equal(-2.0, encoding.Decode(ring[2]));

        // The same at b = 64, where the ring arithmetic is the natural ulong overflow.
        var wide = new FixedPointRingEncoding(bitWidth: 64, fractionalBits: 0, clipBound: Math.Pow(2, 61));
        var wideInputs = new[]
        {
            new[] { Math.Pow(2, 61), -Math.Pow(2, 61) },
            new[] { Math.Pow(2, 61), -Math.Pow(2, 61) },
            new[] { Math.Pow(2, 61), -Math.Pow(2, 61) },
        };
        var wideRing = RunProtocol(wide, threshold: 3, wideInputs, new Dropouts());
        Assert.Equal(new[] { 3UL << 61, unchecked((ulong)(-(3L << 61))) }, wideRing);
        Assert.Equal(3.0 * Math.Pow(2, 61), wide.Decode(wideRing[0]));
        Assert.Equal(-3.0 * Math.Pow(2, 61), wide.Decode(wideRing[1]));
    }

    [Fact(Timeout = 120000)]
    public async Task RingEncoding_RejectsConfigurationsWhoseSumCanOverflow()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(() => new FixedPointRingEncoding(bitWidth: 16, fractionalBits: 0, clipBound: 40000.0));
        var encoding = new FixedPointRingEncoding(bitWidth: 16, fractionalBits: 0, clipBound: 10000.0);
        encoding.EnsureSumFits(3);
        Assert.Throws<ArgumentException>(() => encoding.EnsureSumFits(4));
        Assert.Throws<ArgumentException>(() => encoding.Encode(double.NaN));
    }

    [Fact(Timeout = 120000)]
    public async Task TamperedShare_AbortsUnmasking()
    {
        await Task.Yield();
        var encoding = new FixedPointRingEncoding();
        var inputs = SampleInputs(clients: 3, length: 4, seed: 11, magnitude: 1.0);

        var abort = Assert.Throws<SecureAggregationAbortedException>(() => RunProtocol(encoding, 2, inputs, new Dropouts(), inboxes =>
        {
            var copy = inboxes.ToDictionary(pair => pair.Key, pair => pair.Value);
            var first = copy[10][0];
            var bytes = (byte[])first.Ciphertext.Clone();
            bytes[20] ^= 0x01;
            copy[10] = new[] { new SecureAggregationEncryptedShare(first.SenderId, first.RecipientId, bytes) }
                .Concat(copy[10].Skip(1))
                .ToArray();
            return copy;
        }));
        Assert.Equal(SecureAggregationRound.Unmasking, abort.Round);
    }

    [Fact(Timeout = 120000)]
    public async Task Client_RefusesRoundsOutOfOrder()
    {
        await Task.Yield();
        using var client = new SecureAggregationClient<double>(1, 2, 2, new FixedPointRingEncoding());
        Assert.Throws<InvalidOperationException>(() => client.Unmask(new[] { 1, 2 }));
        client.AdvertiseKeys();
        Assert.Throws<InvalidOperationException>(() => client.AdvertiseKeys());
    }

    [Theory(Timeout = 120000)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    [InlineData(4, 3)]
    [InlineData(7, 5)]
    [InlineData(10, 7)]
    public async Task DefaultThreshold_IsTwoThirdsPlusOne(int clients, int expected)
    {
        await Task.Yield();
        Assert.Equal(expected, SecureAggregationProtocol.ResolveThreshold(clients, SecureAggregationMode.ThresholdDropoutResilient, 0));
        Assert.Equal(clients, SecureAggregationProtocol.ResolveThreshold(clients, SecureAggregationMode.FullParticipation, 0));
    }

    [Fact(Timeout = 120000)]
    public async Task AesCtr_MatchesNistSp80038aCtrAes256Vector()
    {
        await Task.Yield();
        // NIST SP 800-38A, F.5.5 CTR-AES256.Encrypt, blocks 1 and 2.
        var key = Hex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
        var counter = Hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
        var data = Hex("6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51");

        AesCtr.Transform(key, counter, data);

        Assert.Equal(Hex("601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c5"), data);
    }

    [Fact(Timeout = 120000)]
    public async Task P256Agreement_MatchesTheEcdhDeriveKeyMaterialOfThisPlatform()
    {
        await Task.Yield();
        // .NET Framework 4.6.2 and .NET Standard 2.0 agree on BouncyCastle; every other build uses
        // ECDiffieHellman.DeriveKeyMaterial. Both must produce the same bytes, in both directions.
        var alice = P256KeyAgreement.Generate();
        var bob = P256KeyAgreement.Generate();

        var platform = P256KeyAgreement.Agree(alice.PrivateScalar, alice.PublicKey, bob.PublicKey);
        var reverse = P256KeyAgreement.Agree(bob.PrivateScalar, bob.PublicKey, alice.PublicKey);
        var bouncy = P256Sha256Agreement.Derive(alice.PrivateScalar, bob.PublicKey);

        Assert.Equal(32, platform.Length);
        Assert.Equal(platform, reverse);
        Assert.Equal(platform, bouncy);
        Assert.True(P256Sha256Agreement.IsKeyPair(alice.PrivateScalar, alice.PublicKey));
        Assert.False(P256Sha256Agreement.IsKeyPair(alice.PrivateScalar, bob.PublicKey));

        var offCurve = (byte[])alice.PublicKey.Clone();
        offCurve[63] ^= 0x01;
        Assert.Throws<ArgumentException>(() => P256KeyAgreement.ValidatePublicKey(offCurve, "key"));
    }

    private static byte[] Hex(string hex)
    {
        var bytes = new byte[hex.Length / 2];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = Convert.ToByte(hex.Substring(2 * i, 2), 16);
        }

        return bytes;
    }
}
