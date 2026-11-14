using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.FederatedLearning.Privacy;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// The privacy argument of Bonawitz et al. (2017), section 6, rests on one rule of round 4: for any client v the
/// server may learn shares of the masking key s_v^SK (v dropped before sending a masked input) or of the self-mask
/// seed b_v (v's masked input is in the sum), never both. With both, it could strip every mask from y_v and read
/// x_v. These tests pin that rule on the client, which is the party that enforces it, and on the server's intake.
/// </summary>
public class SecureAggregationPrivacyInvariantTests
{
    private const int Length = 9;

    private sealed class Round
    {
        public Round(
            Dictionary<int, SecureAggregationClient<double>> clients,
            SecureAggregationServer<double> server,
            IReadOnlyList<int> u3,
            int[] droppedAfterShareKeys,
            FixedPointRingEncoding encoding,
            Dictionary<int, double[]> inputs)
        {
            Clients = clients;
            Server = server;
            U3 = u3;
            DroppedAfterShareKeys = droppedAfterShareKeys;
            Encoding = encoding;
            Inputs = inputs;
        }

        public Dictionary<int, SecureAggregationClient<double>> Clients { get; }
        public SecureAggregationServer<double> Server { get; }
        public IReadOnlyList<int> U3 { get; }
        public int[] DroppedAfterShareKeys { get; }
        public FixedPointRingEncoding Encoding { get; }
        public Dictionary<int, double[]> Inputs { get; }
    }

    /// <summary>Rounds 0 to 2 for <paramref name="ids"/>; the clients in <paramref name="dropAfterShareKeys"/> share
    /// keys and then never send a masked input.</summary>
    private static Round RunToUnmasking(int[] ids, int threshold, int[] dropAfterShareKeys, int bitWidth = 64, int fractionalBits = 24, double clipBound = 65536.0)
    {
        var encoding = new FixedPointRingEncoding(bitWidth, fractionalBits, clipBound);
        var clients = ids.ToDictionary(id => id, id => new SecureAggregationClient<double>(id, Length, threshold, encoding));
        var server = new SecureAggregationServer<double>(Length, threshold, encoding);
        var inputs = ids.ToDictionary(id => id, id => Enumerable.Range(0, Length).Select(i => Math.Sin(id * 0.7 + i) * 3.0).ToArray());

        var u1 = server.CollectAdvertisedKeys(ids.Select(id => clients[id].AdvertiseKeys()).ToArray());
        var inboxes = server.CollectShares(ids.SelectMany(id => clients[id].ShareKeys(u1)).ToArray());
        var masked = ids.Where(id => !dropAfterShareKeys.Contains(id))
            .Select(id => clients[id].MaskInput(new Vector<double>(inputs[id]), inboxes[id]))
            .ToArray();
        var u3 = server.CollectMaskedInputs(masked);
        return new Round(clients, server, u3, dropAfterShareKeys, encoding, inputs);
    }

    [Fact(Timeout = 120000)]
    public async Task Unmask_ReleasesOneKindOfShareForEachClient_NeverBoth()
    {
        await Task.Yield();
        var ids = new[] { 3, 8, 11, 20, 21, 40 };
        var round = RunToUnmasking(ids, threshold: 4, dropAfterShareKeys: new[] { 8, 21 });
        var survivors = round.U3.ToArray();
        Assert.Equal(new[] { 3, 11, 20, 40 }, survivors);

        foreach (var id in survivors)
        {
            var response = round.Clients[id].Unmask(survivors);

            // s_v^SK only for the dropped clients, b_v only for the survivors: the two sets partition U_2.
            Assert.Equal(round.DroppedAfterShareKeys.OrderBy(v => v), response.MaskingKeyShares.Keys.OrderBy(v => v));
            Assert.Equal(survivors.OrderBy(v => v), response.SelfMaskShares.Keys.OrderBy(v => v));
            Assert.Empty(response.MaskingKeyShares.Keys.Intersect(response.SelfMaskShares.Keys));
            // A client never gives up a share of its own masking key.
            Assert.DoesNotContain(id, response.MaskingKeyShares.Keys);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Client_AnswersTheUnmaskingRoundOnce_SoASecondSurvivorListCannotCollectTheOtherShares()
    {
        await Task.Yield();
        var ids = new[] { 1, 2, 3, 4, 5 };
        var round = RunToUnmasking(ids, threshold: 3, dropAfterShareKeys: Array.Empty<int>());
        var client = round.Clients[2];

        var first = client.Unmask(round.U3.ToArray());
        Assert.Empty(first.MaskingKeyShares);

        // Naming client 5 as dropped in a second request would release s_5^SK shares after b_5 shares were released.
        Assert.Throws<InvalidOperationException>(() => client.Unmask(new[] { 1, 2, 3, 4 }));
        Assert.Equal(SecureAggregationRound.Completed, client.Round);
    }

    [Fact(Timeout = 120000)]
    public async Task Client_RefusesASurvivorListThatIsNotASubsetOfU2ContainingItself()
    {
        await Task.Yield();
        var ids = new[] { 1, 2, 3, 4, 5 };

        var outside = RunToUnmasking(ids, threshold: 3, dropAfterShareKeys: Array.Empty<int>());
        var abort = Assert.Throws<SecureAggregationAbortedException>(() => outside.Clients[1].Unmask(new[] { 1, 2, 3, 99 }));
        Assert.Equal(SecureAggregationRound.Unmasking, abort.Round);

        var withoutSelf = RunToUnmasking(ids, threshold: 3, dropAfterShareKeys: Array.Empty<int>());
        Assert.Throws<SecureAggregationAbortedException>(() => withoutSelf.Clients[1].Unmask(new[] { 2, 3, 4, 5 }));

        var belowThreshold = RunToUnmasking(ids, threshold: 3, dropAfterShareKeys: Array.Empty<int>());
        Assert.Throws<SecureAggregationAbortedException>(() => belowThreshold.Clients[1].Unmask(new[] { 1, 2 }));

        var duplicated = RunToUnmasking(ids, threshold: 3, dropAfterShareKeys: Array.Empty<int>());
        Assert.Throws<SecureAggregationAbortedException>(() => duplicated.Clients[1].Unmask(new[] { 1, 2, 3, 3 }));
    }

    [Fact(Timeout = 120000)]
    public async Task Server_RejectsAResponseCarryingTheWrongKindOfShare()
    {
        await Task.Yield();
        var ids = new[] { 1, 2, 3, 4, 5 };
        var round = RunToUnmasking(ids, threshold: 3, dropAfterShareKeys: new[] { 4 });
        var survivors = round.U3.ToArray();
        var honest = survivors.Select(id => round.Clients[id].Unmask(survivors)).ToArray();

        // A masking-key share of a survivor (client 2's input is in the sum, its self mask is being removed).
        var forSurvivor = new SecureAggregationUnmaskingResponse(
            honest[0].ClientId,
            new Dictionary<int, byte[]> { [4] = honest[0].MaskingKeyShares[4], [2] = honest[0].MaskingKeyShares[4] },
            honest[0].SelfMaskShares);
        Assert.Throws<ArgumentException>(() => round.Server.AggregateRingSum(new[] { forSurvivor }.Concat(honest.Skip(1))));

        // A self-mask share of a dropped client (its masking key is being reconstructed).
        var forDropped = new SecureAggregationUnmaskingResponse(
            honest[0].ClientId,
            honest[0].MaskingKeyShares,
            honest[0].SelfMaskShares.Concat(new[] { new KeyValuePair<int, byte[]>(4, honest[0].SelfMaskShares[1]) })
                .ToDictionary(pair => pair.Key, pair => pair.Value));
        Assert.Throws<ArgumentException>(() => round.Server.AggregateRingSum(new[] { forDropped }.Concat(honest.Skip(1))));

        // The honest responses still unmask the exact sum of the survivors.
        var ring = round.Server.AggregateRingSum(honest);
        var expected = new ulong[Length];
        foreach (var id in survivors)
        {
            for (int i = 0; i < Length; i++)
            {
                expected[i] = round.Encoding.Add(expected[i], round.Encoding.Encode(round.Inputs[id][i]));
            }
        }

        Assert.Equal(expected, ring);
    }

    [Theory(Timeout = 120000)]
    [InlineData(64, 24, 65536.0)]
    [InlineData(40, 16, 64.0)]
    [InlineData(17, 6, 16.0)]
    public async Task ExactSum_WithNegativeAndNonContiguousClientIds_AndDropouts(int bitWidth, int fractionalBits, double clipBound)
    {
        await Task.Yield();
        var ids = new[] { -7, -1, 0, 5, 1000, int.MaxValue };
        var round = RunToUnmasking(ids, threshold: 4, dropAfterShareKeys: new[] { -1, 1000 }, bitWidth, fractionalBits, clipBound);
        var survivors = round.U3.ToArray();

        // One survivor drops after sending its masked input: its input stays in the sum.
        var responders = survivors.Where(id => id != 0).ToArray();
        Assert.True(responders.Length >= 3);
        // t = 4 responders are required; all four survivors minus one is three, so this must abort ...
        Assert.Throws<SecureAggregationAbortedException>(
            () => round.Server.AggregateRingSum(responders.Select(id => round.Clients[id].Unmask(survivors)).ToArray()));

        // ... and with every survivor answering, the sum is exact in the chosen ring.
        var again = RunToUnmasking(ids, threshold: 4, dropAfterShareKeys: new[] { -1, 1000 }, bitWidth, fractionalBits, clipBound);
        var u3 = again.U3.ToArray();
        var ring = again.Server.AggregateRingSum(u3.Select(id => again.Clients[id].Unmask(u3)).ToArray());
        var expected = new ulong[Length];
        foreach (var id in u3)
        {
            for (int i = 0; i < Length; i++)
            {
                expected[i] = again.Encoding.Add(expected[i], again.Encoding.Encode(again.Inputs[id][i]));
            }
        }

        Assert.Equal(expected, ring);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskedInput_OfOneClient_IsNotItsEncodingAndDiffersBetweenRuns()
    {
        await Task.Yield();
        var ids = new[] { 1, 2, 3 };
        ulong[] MaskedOfClientOne()
        {
            var encoding = new FixedPointRingEncoding();
            var clients = ids.ToDictionary(id => id, id => new SecureAggregationClient<double>(id, Length, 2, encoding));
            var server = new SecureAggregationServer<double>(Length, 2, encoding);
            var u1 = server.CollectAdvertisedKeys(ids.Select(id => clients[id].AdvertiseKeys()).ToArray());
            var inboxes = server.CollectShares(ids.SelectMany(id => clients[id].ShareKeys(u1)).ToArray());
            return clients[1].MaskInput(new Vector<double>(Enumerable.Repeat(0.5, Length).ToArray()), inboxes[1]).Values;
        }

        var plain = new FixedPointRingEncoding().Encode(0.5);
        var first = MaskedOfClientOne();
        var second = MaskedOfClientOne();

        // Fresh keys and a fresh self-mask seed every aggregation: the same input never produces the same message,
        // and no element is left unmasked (each coincides with its encoding with probability 2^-64).
        Assert.DoesNotContain(plain, first);
        Assert.NotEqual(first, second);
        Assert.Equal(Length, first.Distinct().Count());
    }
}
