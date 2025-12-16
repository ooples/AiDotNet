using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningRobustAggregationIntegrationTests
{
    [Theory]
    [InlineData("Median", 5)]
    [InlineData("TrimmedMean", 5)]
    [InlineData("Krum", 5)]
    [InlineData("MultiKrum", 7)]
    [InlineData("Bulyan", 7)]
    public async Task BuildAsync_WithRobustAggregationStrategy_CompletesAndReportsStrategy(string aggregationStrategy, int numberOfClients)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new NoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = numberOfClients,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = aggregationStrategy,
            RobustAggregation = new RobustAggregationOptions
            {
                TrimFraction = 0.2,
                ByzantineClientCount = 1,
                MultiKrumSelectionCount = 0,
                UseClientWeightsWhenAveragingSelectedUpdates = false
            },
            RandomSeed = 42,
            MinRoundsBeforeConvergence = 1,
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
        Assert.Equal(aggregationStrategy, metadata!.AggregationStrategyUsed);
        Assert.True(metadata.RoundsCompleted >= 1);
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

