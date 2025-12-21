using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningInvalidConfigurationTests
{
    [Fact]
    public async Task BuildAsync_WithPersonalizationAndMetaLearning_Throws()
    {
        var options = CreateBaseOptions();
        options.Personalization = new FederatedPersonalizationOptions { Enabled = true, Strategy = "Ditto" };
        options.MetaLearning = new FederatedMetaLearningOptions { Enabled = true, Strategy = "Reptile" };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Personalization and federated meta-learning", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithAsyncModeAndSecureAggregation_Throws()
    {
        var options = CreateBaseOptions();
        options.UseSecureAggregation = true;
        options.AsyncFederatedLearning = new AsyncFederatedLearningOptions { Mode = FederatedAsyncMode.FedAsync, SimulatedMaxClientDelaySteps = 1 };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Secure aggregation", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("asynchronous", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithAsyncModeAndHeOnly_Throws()
    {
        var options = CreateBaseOptions();
        options.AsyncFederatedLearning = new AsyncFederatedLearningOptions { Mode = FederatedAsyncMode.FedAsync, SimulatedMaxClientDelaySteps = 1 };
        options.HomomorphicEncryption = new HomomorphicEncryptionOptions
        {
            Enabled = true,
            Scheme = HomomorphicEncryptionScheme.Ckks,
            Mode = HomomorphicEncryptionMode.HeOnly,
            PolyModulusDegree = 4096
        };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("HE-only", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("asynchronous", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithSecureAggregationAndUnsupportedAggregator_Throws()
    {
        var options = CreateBaseOptions();
        options.UseSecureAggregation = true;
        options.AggregationStrategy = FederatedAggregationStrategy.FedBN;

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Secure aggregation", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("FedAvg", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithUnknownAsyncMode_Throws()
    {
        var options = CreateBaseOptions();
        options.AsyncFederatedLearning = new AsyncFederatedLearningOptions
        {
            Mode = (FederatedAsyncMode)999,
            SimulatedMaxClientDelaySteps = 1
        };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Unknown async federated learning mode", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithUnknownServerOptimizer_Throws()
    {
        var options = CreateBaseOptions();
        options.ServerOptimizer = new FederatedServerOptimizerOptions { Optimizer = (FederatedServerOptimizer)999 };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Unknown server optimizer", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithUnknownHeterogeneityCorrection_Throws()
    {
        var options = CreateBaseOptions();
        options.HeterogeneityCorrection = new FederatedHeterogeneityCorrectionOptions { Algorithm = (FederatedHeterogeneityCorrection)999 };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Unknown heterogeneity correction", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithUnknownPrivacyAccountant_Throws()
    {
        var options = CreateBaseOptions();
        options.UseDifferentialPrivacy = true;
        options.DifferentialPrivacyMode = DifferentialPrivacyMode.Central;
        options.PrivacyAccountant = (FederatedPrivacyAccountant)999;

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Unknown privacy accountant", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithUnknownClientSelectionStrategy_Throws()
    {
        var options = CreateBaseOptions();
        options.ClientSelection = new ClientSelectionOptions { Strategy = (FederatedClientSelectionStrategy)999 };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options));
        Assert.Contains("Unknown client selection strategy", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task BuildAsync_WithUnknownCompressionStrategy_Throws()
    {
        var options = CreateBaseOptions();
        options.Compression = new FederatedCompressionOptions { Strategy = "Bogus", Ratio = 0.1, UseErrorFeedback = false };

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => BuildAsync(options, useDeterministicDeltaOptimizer: true));
        Assert.Contains("Unknown compression strategy", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static FederatedLearningOptions CreateBaseOptions()
    {
        return new FederatedLearningOptions
        {
            NumberOfClients = 3,
            MaxRounds = 1,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0
        };
    }

    private static async Task BuildAsync(FederatedLearningOptions options, bool useDeterministicDeltaOptimizer = false)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(8), parameterCount: 8);
        AiDotNet.Interfaces.IOptimizer<double, Matrix<double>, Vector<double>> optimizer = useDeterministicDeltaOptimizer
            ? new FederatedDeterministicDeltaOptimizer(model)
            : new FederatedNoOpOptimizer(model);

        await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(options)
            .BuildAsync();
    }

    private static (Matrix<double> x, Vector<double> y) CreateToyData()
    {
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i * 2;
            y[i] = i;
        }

        return (x, y);
    }
}
