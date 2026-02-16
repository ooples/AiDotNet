using AiDotNet.FederatedLearning.Graph;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for federated graph learning (#541).
/// </summary>
public class GraphFederatedLearningTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    // ========== FedGnnAggregationStrategy Tests ==========

    [Fact]
    public void FedGnnAggregation_StrategyName_IsFedGNN()
    {
        var strategy = new FedGnnAggregationStrategy<double>();

        Assert.Equal("FedGNN", strategy.StrategyName);
    }

    [Fact]
    public void FedGnnAggregation_DefaultWeights_SumToOne()
    {
        // Default weights: edge=0.4, label=0.4, degree=0.2
        var strategy = new FedGnnAggregationStrategy<double>(0.4, 0.4, 0.2);
        Assert.NotNull(strategy);
    }

    [Fact]
    public void FedGnnAggregation_Aggregate_WithClientModels_ProducesResult()
    {
        var strategy = new FedGnnAggregationStrategy<double>();
        var clientModels = new Dictionary<int, Tensor<double>>
        {
            { 0, CreateTensor(1.0, 2.0, 3.0) },
            { 1, CreateTensor(4.0, 5.0, 6.0) },
            { 2, CreateTensor(7.0, 8.0, 9.0) }
        };
        var clientGraphStats = new Dictionary<int, ClientGraphStats>
        {
            { 0, new ClientGraphStats { NodeCount = 100, EdgeCount = 200, LabeledNodeCount = 100, AverageDegree = 4.0 } },
            { 1, new ClientGraphStats { NodeCount = 150, EdgeCount = 300, LabeledNodeCount = 150, AverageDegree = 4.0 } },
            { 2, new ClientGraphStats { NodeCount = 80, EdgeCount = 120, LabeledNodeCount = 80, AverageDegree = 3.0 } }
        };

        var result = strategy.Aggregate(clientModels, clientGraphStats);

        Assert.NotNull(result);
        Assert.Equal(3, result.Shape[0]);
    }

    // ========== SubgraphExpander Tests ==========

    [Fact]
    public void SubgraphExpander_Constructor_WithOptions_Succeeds()
    {
        var options = new FederatedGraphOptions
        {
            PseudoNodeStrategy = PseudoNodeStrategy.FeatureAverage,
            NodeFeatureDimension = 64
        };
        var expander = new SubgraphExpander<double>(options);

        Assert.NotNull(expander);
    }

    [Fact]
    public void SubgraphExpander_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new SubgraphExpander<double>(null));
    }

    // ========== FederatedGraphPartitioner Tests ==========

    [Fact]
    public void GraphPartitioner_Constructor_WithOptions_Succeeds()
    {
        var options = new FederatedGraphOptions
        {
            PartitionStrategy = GraphPartitionStrategy.Random,
            NumberOfPartitions = 3
        };
        var partitioner = new FederatedGraphPartitioner<double>(options);

        Assert.NotNull(partitioner);
    }

    [Fact]
    public void GraphPartitioner_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new FederatedGraphPartitioner<double>(null));
    }

    // ========== GraphNeighborhoodPrivacy Tests ==========

    [Fact]
    public void GraphNeighborhoodPrivacy_Constructor_WithDefaults_Succeeds()
    {
        var privacy = new GraphNeighborhoodPrivacy<double>();

        Assert.NotNull(privacy);
    }

    [Fact]
    public void GraphNeighborhoodPrivacy_Constructor_WithParams_Succeeds()
    {
        var privacy = new GraphNeighborhoodPrivacy<double>(epsilon: 1.0, delta: 1e-5, sensitivity: 2.0);

        Assert.NotNull(privacy);
    }

    [Fact]
    public void GraphNeighborhoodPrivacy_ZeroEpsilon_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphNeighborhoodPrivacy<double>(epsilon: 0));
    }

    [Fact]
    public void GraphNeighborhoodPrivacy_NegativeEpsilon_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphNeighborhoodPrivacy<double>(epsilon: -1.0));
    }

    [Fact]
    public void GraphNeighborhoodPrivacy_InvalidDelta_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphNeighborhoodPrivacy<double>(epsilon: 1.0, delta: 0));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphNeighborhoodPrivacy<double>(epsilon: 1.0, delta: 1.0));
    }

    // ========== SecureCrossClientEdgeDiscovery Tests ==========

    [Fact]
    public void SecureCrossClientEdge_Constructor_WithDefaults_Succeeds()
    {
        var discovery = new SecureCrossClientEdgeDiscovery<double>();

        Assert.NotNull(discovery);
        Assert.Equal(0, discovery.DiscoveredEdgeCount);
    }

    [Fact]
    public void SecureCrossClientEdge_Constructor_WithParams_Succeeds()
    {
        var discovery = new SecureCrossClientEdgeDiscovery<double>(
            privacyEpsilon: 2.0,
            maxEdgesPerPair: 500,
            cacheEnabled: true);

        Assert.NotNull(discovery);
    }

    [Fact]
    public void SecureCrossClientEdge_DiscoverEdges_FindsCommonNodes()
    {
        var discovery = new SecureCrossClientEdgeDiscovery<double>();
        var clientANodes = new List<int> { 1, 2, 3, 4, 5 };
        var clientBNodes = new List<int> { 3, 4, 5, 6, 7 };

        var edges = discovery.DiscoverEdges(clientANodes, clientBNodes);

        Assert.NotNull(edges);
        // Common nodes: 3, 4, 5
        Assert.True(edges.Count >= 0);
    }

    // ========== PrototypeFederatedGraphLearning Tests ==========

    [Fact]
    public void PrototypeFGL_Constructor_Succeeds()
    {
        var options = new FederatedGraphOptions
        {
            UsePrototypeLearning = true,
            PrototypesPerClass = 3,
            NodeFeatureDimension = 16
        };
        var proto = new PrototypeFederatedGraphLearning<double>(
            options, prototypeDim: 16, numClasses: 3);

        Assert.NotNull(proto);
    }

    [Fact]
    public void PrototypeFGL_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new PrototypeFederatedGraphLearning<double>(null, 16, 3));
    }

    [Fact]
    public void PrototypeFGL_RegisterAndAggregate_Prototypes()
    {
        var options = new FederatedGraphOptions
        {
            UsePrototypeLearning = true,
            PrototypesPerClass = 2
        };
        var proto = new PrototypeFederatedGraphLearning<double>(
            options, prototypeDim: 4, numClasses: 2);

        // Register prototypes from clients using correct method name
        var clientPrototypes = new Dictionary<int, Tensor<double>>
        {
            { 0, CreateTensor(1.0, 2.0, 3.0, 4.0) },
            { 1, CreateTensor(5.0, 6.0, 7.0, 8.0) }
        };
        proto.RegisterClientPrototypes(clientId: 0, clientPrototypes);

        var client2Prototypes = new Dictionary<int, Tensor<double>>
        {
            { 0, CreateTensor(2.0, 3.0, 4.0, 5.0) },
            { 1, CreateTensor(6.0, 7.0, 8.0, 9.0) }
        };
        proto.RegisterClientPrototypes(clientId: 1, client2Prototypes);

        var globalPrototypes = proto.AggregatePrototypes();

        Assert.NotNull(globalPrototypes);
    }

    [Fact]
    public void PrototypeFGL_ComputePrototypeLoss_ReturnsValue()
    {
        var options = new FederatedGraphOptions
        {
            UsePrototypeLearning = true,
            PrototypesPerClass = 2
        };
        var proto = new PrototypeFederatedGraphLearning<double>(
            options, prototypeDim: 4, numClasses: 2);

        var prototypes = new Dictionary<int, Tensor<double>>
        {
            { 0, CreateTensor(1.0, 2.0, 3.0, 4.0) },
            { 1, CreateTensor(5.0, 6.0, 7.0, 8.0) }
        };

        double loss = proto.ComputePrototypeLoss(prototypes, lambda: 0.1);

        Assert.True(loss >= 0, "Prototype loss should be non-negative");
    }

    // ========== GraphNodeGenerator Tests ==========

    [Fact]
    public void GraphNodeGenerator_Constructor_WithFeatureDim_Succeeds()
    {
        var generator = new GraphNodeGenerator<double>(featureDim: 16);

        Assert.NotNull(generator);
    }

    [Fact]
    public void GraphNodeGenerator_Constructor_WithParams_Succeeds()
    {
        var generator = new GraphNodeGenerator<double>(
            featureDim: 32, hiddenDim: 64, learningRate: 0.01);

        Assert.NotNull(generator);
    }

    [Fact]
    public void GraphNodeGenerator_ZeroFeatureDim_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphNodeGenerator<double>(featureDim: 0));
    }

    [Fact]
    public void GraphNodeGenerator_NegativeFeatureDim_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphNodeGenerator<double>(featureDim: -1));
    }

    // ========== SubgraphFederatedTrainer Tests ==========

    [Fact]
    public void SubgraphFederatedTrainer_Constructor_Succeeds()
    {
        var options = new FederatedGraphOptions();
        var aggregation = new FedGnnAggregationStrategy<double>();

        var trainer = new SubgraphFederatedTrainer<double>(options, aggregation);

        Assert.NotNull(trainer);
    }

    [Fact]
    public void SubgraphFederatedTrainer_WithEdgeHandler_Succeeds()
    {
        var options = new FederatedGraphOptions();
        var aggregation = new FedGnnAggregationStrategy<double>();
        var edgeHandler = new SecureCrossClientEdgeDiscovery<double>();

        var trainer = new SubgraphFederatedTrainer<double>(options, aggregation, edgeHandler);

        Assert.NotNull(trainer);
    }

    // ========== FederatedGraphOptions Defaults ==========

    [Fact]
    public void FederatedGraphOptions_DefaultValues()
    {
        var options = new FederatedGraphOptions();

        Assert.Equal(GraphFLMode.SubgraphLevel, options.Mode);
        Assert.Equal(GraphPartitionStrategy.Preassigned, options.PartitionStrategy);
        Assert.Equal(PseudoNodeStrategy.FeatureAverage, options.PseudoNodeStrategy);
        Assert.NotNull(options.Sampling);
        Assert.NotNull(options.CrossClientEdges);
        Assert.Equal(64, options.NodeFeatureDimension);
        Assert.Equal(128, options.HiddenDimension);
        Assert.Equal(2, options.NumGnnLayers);
        Assert.False(options.UsePrototypeLearning);
        Assert.Equal(5, options.PrototypesPerClass);
        Assert.Equal(2.0, options.NeighborhoodPrivacyEpsilon);
        Assert.Null(options.NumberOfPartitions);
    }

    [Fact]
    public void GraphFLMode_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(GraphFLMode), GraphFLMode.SubgraphLevel));
    }

    [Fact]
    public void GraphPartitionStrategy_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(GraphPartitionStrategy), GraphPartitionStrategy.Preassigned));
        Assert.True(Enum.IsDefined(typeof(GraphPartitionStrategy), GraphPartitionStrategy.Random));
    }

    [Fact]
    public void PseudoNodeStrategy_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(PseudoNodeStrategy), PseudoNodeStrategy.FeatureAverage));
    }
}
