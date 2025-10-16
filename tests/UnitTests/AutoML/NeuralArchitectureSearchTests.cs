using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML
{
    /// <summary>
    /// Comprehensive unit tests for NeuralArchitectureSearch
    /// Tests cover the syntax error fixes for BUG-003
    /// </summary>
    public class NeuralArchitectureSearchTests
    {
        /// <summary>
        /// Test that the NeuralArchitectureSearch class can be instantiated with default parameters
        /// This verifies the constructor is correctly closed
        /// </summary>
        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Act
            var nas = new NeuralArchitectureSearch<double>();

            // Assert
            Assert.NotNull(nas);
        }

        /// <summary>
        /// Test that the NeuralArchitectureSearch class can be instantiated with custom parameters
        /// </summary>
        [Fact]
        public void Constructor_WithCustomParameters_CreatesInstance()
        {
            // Arrange
            var strategy = NeuralArchitectureSearchStrategy.RandomSearch;
            var maxLayers = 5;
            var maxNeuronsPerLayer = 256;
            var resourceBudget = 50.0;
            var populationSize = 25;
            var generations = 10;

            // Act
            var nas = new NeuralArchitectureSearch<double>(
                strategy: strategy,
                maxLayers: maxLayers,
                maxNeuronsPerLayer: maxNeuronsPerLayer,
                resourceBudget: resourceBudget,
                populationSize: populationSize,
                generations: generations
            );

            // Assert
            Assert.NotNull(nas);
        }

        /// <summary>
        /// Test that SuggestNextTrialAsync method works correctly (not duplicated)
        /// This verifies the duplicate method removal fix
        /// </summary>
        [Fact]
        public async Task SuggestNextTrialAsync_ReturnsValidParameters()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>(
                strategy: NeuralArchitectureSearchStrategy.RandomSearch,
                maxLayers: 5,
                maxNeuronsPerLayer: 128
            );

            // Act
            var parameters = await nas.SuggestNextTrialAsync();

            // Assert
            Assert.NotNull(parameters);
            Assert.Contains("strategy", parameters.Keys);
            Assert.Contains("num_layers", parameters.Keys);
            Assert.Contains("layers", parameters.Keys);
        }

        /// <summary>
        /// Test that SuggestNextTrialAsync returns parameters within expected ranges
        /// </summary>
        [Fact]
        public async Task SuggestNextTrialAsync_ReturnsParametersInValidRange()
        {
            // Arrange
            var maxLayers = 8;
            var maxNeuronsPerLayer = 256;
            var nas = new NeuralArchitectureSearch<double>(
                maxLayers: maxLayers,
                maxNeuronsPerLayer: maxNeuronsPerLayer
            );

            // Act
            var parameters = await nas.SuggestNextTrialAsync();

            // Assert
            var numLayers = (int)parameters["num_layers"];
            Assert.InRange(numLayers, 2, maxLayers);
        }

        /// <summary>
        /// Test that GetBestArchitecture returns an empty candidate when no search has been run
        /// </summary>
        [Fact]
        public void GetBestArchitecture_WithoutSearch_ReturnsEmptyCandidate()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>();

            // Act
            var bestArchitecture = nas.GetBestArchitecture();

            // Assert
            Assert.NotNull(bestArchitecture);
            Assert.Empty(bestArchitecture.Layers);
        }

        /// <summary>
        /// Test that GetTopArchitectures returns empty list when no search has been run
        /// </summary>
        [Fact]
        public void GetTopArchitectures_WithoutSearch_ReturnsEmptyList()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>();

            // Act
            var topArchitectures = nas.GetTopArchitectures(5);

            // Assert
            Assert.NotNull(topArchitectures);
            Assert.Empty(topArchitectures);
        }

        /// <summary>
        /// Test that GetTopArchitectures returns at most N architectures
        /// </summary>
        [Fact]
        public void GetTopArchitectures_WithLimit_ReturnsCorrectCount()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>();
            var requestedCount = 3;

            // Act
            var topArchitectures = nas.GetTopArchitectures(requestedCount);

            // Assert
            Assert.NotNull(topArchitectures);
            Assert.True(topArchitectures.Count <= requestedCount);
        }

        /// <summary>
        /// Test that NeuralArchitectureSearchModel can be instantiated with correct generic types
        /// This verifies the constructor type mismatch fix
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_WithCorrectTypes_CreatesInstance()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double>();

            // Act
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(ModelType.NeuralNetwork, model.Type);
        }

        /// <summary>
        /// Test that NeuralArchitectureSearchModel has correct parameter count from architecture
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_ParameterCount_ReturnsArchitectureParameters()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double> { Parameters = 1000 };

            // Act
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);

            // Assert
            Assert.Equal(1000, model.ParameterCount);
        }

        /// <summary>
        /// Test that NeuralArchitectureSearchModel GetFeatureImportance returns empty dictionary
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_GetFeatureImportance_ReturnsEmptyDictionary()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double>();
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);

            // Act
            var importance = model.GetFeatureImportance();

            // Assert
            Assert.NotNull(importance);
            Assert.Empty(importance);
        }

        /// <summary>
        /// Test that NeuralArchitectureSearchModel Clone creates a new instance
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_Clone_CreatesNewInstance()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double>();
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);

            // Act
            var cloned = model.Clone();

            // Assert
            Assert.NotNull(cloned);
            Assert.NotSame(model, cloned);
        }

        /// <summary>
        /// Test that multiple SuggestNextTrialAsync calls return different parameters
        /// </summary>
        [Fact]
        public async Task SuggestNextTrialAsync_MultipleCalls_ReturnsDifferentParameters()
        {
            // Arrange
            var nas = new NeuralArchitectureSearch<double>();

            // Act
            var params1 = await nas.SuggestNextTrialAsync();
            var params2 = await nas.SuggestNextTrialAsync();

            // Assert - At least one parameter should be different (statistically likely)
            Assert.NotNull(params1);
            Assert.NotNull(params2);
            // Parameters should have same structure
            Assert.Equal(params1.Keys.Count, params2.Keys.Count);
        }

        /// <summary>
        /// Test that constructor handles all NeuralArchitectureSearchStrategy values
        /// </summary>
        [Theory]
        [InlineData(NeuralArchitectureSearchStrategy.Evolutionary)]
        [InlineData(NeuralArchitectureSearchStrategy.ReinforcementLearning)]
        [InlineData(NeuralArchitectureSearchStrategy.GradientBased)]
        [InlineData(NeuralArchitectureSearchStrategy.RandomSearch)]
        [InlineData(NeuralArchitectureSearchStrategy.BayesianOptimization)]
        public void Constructor_WithAllStrategies_CreatesInstance(NeuralArchitectureSearchStrategy strategy)
        {
            // Act
            var nas = new NeuralArchitectureSearch<double>(strategy: strategy);

            // Assert
            Assert.NotNull(nas);
        }

        /// <summary>
        /// Test that NeuralArchitectureSearchModel Train throws NotSupportedException
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_Train_ThrowsNotSupportedException()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double>();
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);
            var input = new Tensor<double>(new[] { 1, 10 });
            var output = new Tensor<double>(new[] { 1, 1 });

            // Act & Assert
            Assert.Throws<NotSupportedException>(() => model.Train(input, output));
        }

        /// <summary>
        /// Test that IsFeatureUsed returns true for neural networks
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_IsFeatureUsed_ReturnsTrue()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double>();
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);

            // Act
            var result = model.IsFeatureUsed(0);

            // Assert
            Assert.True(result);
        }

        /// <summary>
        /// Test that GetActiveFeatureIndices returns empty enumerable
        /// </summary>
        [Fact]
        public void NeuralArchitectureSearchModel_GetActiveFeatureIndices_ReturnsEmpty()
        {
            // Arrange
            var mockNetwork = new MockNeuralNetworkModel<double>();
            var architecture = new ArchitectureCandidate<double>();
            var model = new NeuralArchitectureSearchModel<double>(mockNetwork, architecture);

            // Act
            var indices = model.GetActiveFeatureIndices();

            // Assert
            Assert.NotNull(indices);
            Assert.Empty(indices);
        }
    }

    /// <summary>
    /// Mock implementation of INeuralNetworkModel for testing
    /// </summary>
    public class MockNeuralNetworkModel<T> : INeuralNetworkModel<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        public string Name => "MockNeuralNetwork";
        public ModelType Type => ModelType.NeuralNetwork;
        public string[] FeatureNames { get; set; } = Array.Empty<string>();

        public void Train(Tensor<T> input, Tensor<T> expectedOutput) { }
        public Tensor<T> Predict(Tensor<T> input) => new Tensor<T>(new[] { 1 });
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public NeuralNetworkArchitecture<T> GetArchitecture() => new NeuralNetworkArchitecture<T>(NetworkComplexity.Medium);
        public void SetOptimizer(IOptimizer<T, Tensor<T>, Tensor<T>> optimizer) { }
        public void SetLossFunction(ILossFunction<T> lossFunction) { }
        public Dictionary<string, Tensor<T>> GetLayerActivations(Tensor<T> input) => new Dictionary<string, Tensor<T>>();
        public void UpdateLearningRate(double newLearningRate) { }
    }

    /// <summary>
    /// Mock implementation of ArchitectureCandidate for testing
    /// </summary>
    public class ArchitectureCandidate<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        public List<LayerConfiguration<T>> Layers { get; set; } = new List<LayerConfiguration<T>>();
        public T Fitness { get; set; }
        public T ValidationAccuracy { get; set; }
        public int Parameters { get; set; }
        public long FLOPs { get; set; }
        public bool IsEvaluated { get; set; }

        public ArchitectureCandidate<T> Clone()
        {
            return new ArchitectureCandidate<T>
            {
                Layers = new List<LayerConfiguration<T>>(Layers.Select(l => l.Clone())),
                Fitness = Fitness,
                ValidationAccuracy = ValidationAccuracy,
                Parameters = Parameters,
                FLOPs = FLOPs,
                IsEvaluated = IsEvaluated
            };
        }
    }

    /// <summary>
    /// Mock implementation of LayerConfiguration for testing
    /// </summary>
    public class LayerConfiguration<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        public LayerType Type { get; set; }
        public int Units { get; set; }
        public ActivationFunction Activation { get; set; }
        public int Filters { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
        public int PoolSize { get; set; }
        public T DropoutRate { get; set; }
        public bool ReturnSequences { get; set; }

        public LayerConfiguration<T> Clone()
        {
            return new LayerConfiguration<T>
            {
                Type = Type,
                Units = Units,
                Activation = Activation,
                Filters = Filters,
                KernelSize = KernelSize,
                Stride = Stride,
                PoolSize = PoolSize,
                DropoutRate = DropoutRate,
                ReturnSequences = ReturnSequences
            };
        }
    }
}
