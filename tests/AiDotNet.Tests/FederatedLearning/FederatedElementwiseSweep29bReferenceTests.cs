using AiDotNet.FederatedLearning.Adapters;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Reference tests for the paper fixes made while moving the remaining federated-learning code (plan 29b)
/// onto engine ops. Every expected value is hand-computed from the cited paper's algorithm.
/// </summary>
public class FederatedElementwiseSweep29bReferenceTests
{
    private const int Precision = 10;

    [Fact(Timeout = 60000)]
    public async Task DPFedLoRA_AddsDataIndependentNoiseSigmaClipOverN()
    {
        await Task.Yield();

        // DP-FedAvg Gaussian mechanism: clip each whole update to C, average, add N(0, (sigma * C / n)^2).
        // Client 0 = [3, 4, 0, 0] (norm 5) clips to [0.6, 0.8, 0, 0]; client 1 = [0, 0, 0.3, 0.4] (norm 0.5)
        // is kept. The average is [0.3, 0.4, 0.15, 0.2] and every coordinate gets std 1 * 1 / 2 = 0.5.
        // (Splitting the clip budget by the clients' own per-layer norms gave stds 1/3 and 1/6.)
        var strategy = new DPFedLoRA<double>(
            modelDim: 100, rank: 1, noiseMultiplier: 1.0, clipNorm: 1.0,
            numAdaptedLayers: 2, layerInputDim: 1, layerOutputDim: 1, seed: 42);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 3.0, 4.0, 0.0, 0.0 }),
            [1] = new Vector<double>(new[] { 0.0, 0.0, 0.3, 0.4 }),
        };

        var aggregated = strategy.AggregateAdapters(clients, null);

        var average = new[] { 0.3, 0.4, 0.15, 0.2 };
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < average.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            Assert.Equal(average[i] + 0.5 * z, aggregated[i], Precision);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LoRAMerge_WritesStoredFactorsUnscaled()
    {
        await Task.Yield();

        // LoRA (Hu et al. 2021) applies alpha / r to B*A in the forward pass. The stored factors are merged
        // back as they are; scaling both by alpha / r = 2 would scale the update by 4 and compound per round.
        var full = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var adapters = new Vector<double>(new[] { 7.0, 8.0 });
        var expected = new[] { 1.0, 2.0, 3.0, 7.0, 8.0 };

        IFederatedAdapterStrategy<double>[] strategies =
        {
            new DPFedLoRA<double>(modelDim: 5, rank: 8, alpha: 16.0),
            new FLoRA<double>(modelDim: 5, rank: 8, alpha: 16.0),
        };

        foreach (var strategy in strategies)
        {
            var merged = strategy.MergeAdapterParameters(full, adapters);
            Assert.Equal(expected, merged.ToArray());
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FLoRA_ReturnsBestRankRApproximationOfStackedUpdate()
    {
        await Task.Yield();

        // FLoRA (Wang et al. 2024) stacking gives dW = sum_k p_k B_k A_k. Client 0: B = [1, 0], A = [2, 0];
        // client 1: B = [0, 1], A = [0, 1]; equal weights give dW = diag(1, 0.5). Its best rank-1
        // approximation (Eckart-Young) is diag(1, 0), so the returned factors must satisfy B*A = diag(1, 0).
        var strategy = new FLoRA<double>(modelDim: 100, rank: 1, numAdaptedLayers: 1, layerInputDim: 2, layerOutputDim: 2);
        var clients = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 0.0, 2.0, 0.0 }),
            [1] = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 }),
        };

        var aggregated = strategy.AggregateAdapters(clients, null);

        Assert.Equal(1.0, aggregated[0] * aggregated[2], Precision);
        Assert.Equal(0.0, aggregated[0] * aggregated[3], Precision);
        Assert.Equal(0.0, aggregated[1] * aggregated[2], Precision);
        Assert.Equal(0.0, aggregated[1] * aggregated[3], Precision);
    }
}
