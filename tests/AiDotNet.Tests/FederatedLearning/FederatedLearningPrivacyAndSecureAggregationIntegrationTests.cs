using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningPrivacyAndSecureAggregationIntegrationTests
{
    [Fact]
    public async Task BuildAsync_WithCentralDpAndBasicAccounting_TracksPrivacySpend()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 4,
            MaxRounds = 3,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            UseDifferentialPrivacy = true,
            DifferentialPrivacyMode = DifferentialPrivacyMode.Central,
            DifferentialPrivacyClipNorm = 1.0,
            PrivacyAccountant = FederatedPrivacyAccountant.Basic,
            PrivacyEpsilon = 0.5,
            PrivacyDelta = 1e-5
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var flMetadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(flMetadata);
        Assert.True(flMetadata!.DifferentialPrivacyEnabled);
        Assert.Equal("Basic", flMetadata.PrivacyAccountantUsed);
        Assert.Equal(3, flMetadata.RoundsCompleted);
        Assert.Equal(1.5, flMetadata.TotalPrivacyBudgetConsumed, 6);
        Assert.Equal(3 * flOptions.PrivacyDelta, flMetadata.TotalPrivacyDeltaConsumed, 12);
        Assert.Equal(flOptions.PrivacyDelta, flMetadata.ReportedDelta, 12);
        Assert.Equal(flMetadata.TotalPrivacyBudgetConsumed, flMetadata.ReportedEpsilonAtDelta, 6);
        Assert.All(flMetadata.RoundMetrics, r => Assert.True(r.PrivacyBudgetConsumed > 0.0));
    }

    [Fact]
    public async Task BuildAsync_WithSecureAggregation_SetsSecureAggregationEnabled()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 4,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 7,
            UseSecureAggregation = true
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var flMetadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(flMetadata);
        Assert.True(flMetadata!.SecureAggregationEnabled);
        Assert.Equal(2, flMetadata.RoundsCompleted);
    }

    [Fact]
    public async Task BuildAsync_WithLocalAndCentralDpAndRdpAccounting_ReportsEpsilonAtDelta()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 4,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 99,
            UseDifferentialPrivacy = true,
            DifferentialPrivacyMode = DifferentialPrivacyMode.LocalAndCentral,
            DifferentialPrivacyClipNorm = 1.0,
            PrivacyAccountant = FederatedPrivacyAccountant.Rdp,
            PrivacyEpsilon = 1.0,
            PrivacyDelta = 1e-5
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var flMetadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(flMetadata);
        Assert.True(flMetadata!.DifferentialPrivacyEnabled);
        Assert.Equal("RDP", flMetadata.PrivacyAccountantUsed);
        Assert.Equal(flOptions.PrivacyDelta, flMetadata.ReportedDelta, 12);
        Assert.True(flMetadata.ReportedEpsilonAtDelta > 0.0);
        Assert.Equal(4 * flOptions.PrivacyDelta, flMetadata.TotalPrivacyDeltaConsumed, 12);
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
