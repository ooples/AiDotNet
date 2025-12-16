using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningClientSelectionIntegrationTests
{
    [Theory]
    [InlineData("UniformRandom")]
    [InlineData("WeightedRandom")]
    [InlineData("Stratified")]
    [InlineData("AvailabilityAware")]
    [InlineData("PerformanceAware")]
    [InlineData("Clustered")]
    public async Task BuildAsync_WithClientSelectionStrategy_CompletesAndReportsClients(string strategy)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new NoOpOptimizer(model);

        var selectionOptions = new ClientSelectionOptions
        {
            Strategy = strategy,
            ExplorationRate = 0.2,
            ClusterCount = 2,
            KMeansIterations = 3,
            AvailabilityThreshold = 0.0,
            ClientGroupKeys = CreateGroupKeysForSixClients(),
            ClientAvailabilityProbabilities = CreateAvailabilityForSixClients()
        };

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 6,
            MaxRounds = 2,
            ClientSelectionFraction = 0.5,
            ClientSelection = selectionOptions,
            LocalEpochs = 1,
            AggregationStrategy = "FedAvg",
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0
        };

        var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(2, metadata!.RoundsCompleted);
        Assert.All(metadata.RoundMetrics, r =>
        {
            Assert.True(r.SelectedClientIds.Count >= 1);
            Assert.True(r.SelectedClientIds.SequenceEqual(r.SelectedClientIds.OrderBy(id => id)));
        });
        Assert.True(metadata.TotalCommunicationMB > 0.0);
    }

    private static Dictionary<int, string> CreateGroupKeysForSixClients()
    {
        return new Dictionary<int, string>
        {
            [0] = "A",
            [1] = "A",
            [2] = "A",
            [3] = "B",
            [4] = "B",
            [5] = "B"
        };
    }

    private static Dictionary<int, double> CreateAvailabilityForSixClients()
    {
        return new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 0.9,
            [2] = 0.8,
            [3] = 0.7,
            [4] = 0.6,
            [5] = 0.5
        };
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

    private sealed class NoOpOptimizer : IOptimizer<double, Matrix<double>, Vector<double>>
    {
        private readonly OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>> _options;
        private readonly IFullModel<double, Matrix<double>, Vector<double>> _model;

        public NoOpOptimizer(
            IFullModel<double, Matrix<double>, Vector<double>> model,
            OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>? options = null)
        {
            _model = model;
            _options = options ?? new OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>();
        }

        public OptimizationResult<double, Matrix<double>, Vector<double>> Optimize(OptimizationInputData<double, Matrix<double>, Vector<double>> inputData)
        {
            var best = inputData.InitialSolution ?? _model;
            return new OptimizationResult<double, Matrix<double>, Vector<double>>
            {
                BestSolution = best.WithParameters(best.GetParameters()),
                Iterations = _options.MaxIterations
            };
        }

        public bool ShouldEarlyStop() => false;

        public OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>> GetOptions() => _options;

        public void Reset()
        {
        }

        public byte[] Serialize() => Array.Empty<byte>();

        public void Deserialize(byte[] data)
        {
        }

        public void SaveModel(string filePath)
        {
        }

        public void LoadModel(string filePath)
        {
        }
    }
}
