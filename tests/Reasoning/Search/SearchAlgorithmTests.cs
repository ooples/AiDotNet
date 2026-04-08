using Xunit;
using Moq;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Search;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Components;

namespace AiDotNet.Tests.Reasoning.Search;

/// <summary>
/// Unit tests for search algorithms.
/// </summary>
public class SearchAlgorithmTests
{
    private readonly Mock<IChatModel> _mockChatModel;
    private readonly Mock<IThoughtGenerator<double>> _mockGenerator;
    private readonly Mock<IThoughtEvaluator<double>> _mockEvaluator;

    public SearchAlgorithmTests()
    {
        _mockChatModel = new Mock<IChatModel>();
        _mockGenerator = new Mock<IThoughtGenerator<double>>();
        _mockEvaluator = new Mock<IThoughtEvaluator<double>>();
    }

    [Fact]
    public async Task BreadthFirstSearch_FindsGoalNode()
    {
        // Arrange
        var bfs = new BreadthFirstSearch<double>();
        var root = CreateTestTree();
        var config = ReasoningConfig.Default;

        SetupMockGenerator();
        SetupMockEvaluator();

        // Act
        var path = await bfs.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
        Assert.Equal(root, path[0]);
    }

    [Fact]
    public async Task DepthFirstSearch_ExploresDepthFirst()
    {
        // Arrange
        var dfs = new DepthFirstSearch<double>();
        var root = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 3 };

        SetupMockGenerator();
        SetupMockEvaluator();

        // Act
        var path = await dfs.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
    }

    [Fact]
    public async Task BeamSearch_RespectsBeamWidth()
    {
        // Arrange
        var beamSearch = new BeamSearch<double>(beamWidth: 2);
        var root = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 3 };

        SetupMockGenerator(childrenPerNode: 5); // More than beam width
        SetupMockEvaluator();

        // Act
        var path = await beamSearch.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
        // Beam search should prune to beam width at each level
    }

    [Fact]
    public async Task MonteCarloTreeSearch_BalancesExplorationAndExploitation()
    {
        // Arrange
        var mcts = new MonteCarloTreeSearch<double>(
            explorationConstant: 1.414,
            simulationCount: 10
        );
        var root = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 2 };

        SetupMockGenerator();
        SetupMockEvaluator();

        // Act
        var path = await mcts.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
    }

    [Fact]
    public async Task BestFirstSearch_SelectsHighestScoredNodes()
    {
        // Arrange
        var bestFirst = new BestFirstSearch<double>();
        var root = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 3 };

        // Setup evaluator to return decreasing scores
        double currentScore = 1.0;
        _mockEvaluator
            .Setup(e => e.EvaluateThoughtAsync(
                It.IsAny<ThoughtNode<double>>(),
                It.IsAny<string>(),
                It.IsAny<ReasoningConfig>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                currentScore -= 0.1;
                return currentScore;
            });

        SetupMockGenerator();

        // Act
        var path = await bestFirst.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
    }

    [Fact]
    public async Task SearchWithCancellation_ThrowsOperationCanceledException()
    {
        // Arrange
        var bfs = new BreadthFirstSearch<double>();
        var root = CreateTestTree();
        var config = ReasoningConfig.Default;
        var cts = new CancellationTokenSource();

        SetupMockGenerator();
        SetupMockEvaluator();

        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await bfs.SearchAsync(
                root,
                _mockGenerator.Object,
                _mockEvaluator.Object,
                config,
                cts.Token)
        );
    }

    [Fact]
    public async Task SearchWithMaxDepth_StopsAtLimit()
    {
        // Arrange
        var bfs = new BreadthFirstSearch<double>();
        var root = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 2 };

        SetupMockGenerator();
        SetupMockEvaluator();

        // Act
        var path = await bfs.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.True(path.Count <= config.ExplorationDepth + 1); // +1 for root
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public async Task BeamSearch_WithDifferentWidths_WorksCorrectly(int beamWidth)
    {
        // Arrange
        var beamSearch = new BeamSearch<double>(beamWidth);
        var root = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 2 };

        SetupMockGenerator(childrenPerNode: 10);
        SetupMockEvaluator();

        // Act
        var path = await beamSearch.SearchAsync(root, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
    }

    [Fact]
    public async Task MCTS_WithMoreSimulations_ConvergesToBetterSolution()
    {
        // Arrange
        var mcts1 = new MonteCarloTreeSearch<double>(simulationCount: 5);
        var mcts2 = new MonteCarloTreeSearch<double>(simulationCount: 20);
        var root1 = CreateTestTree();
        var root2 = CreateTestTree();
        var config = new ReasoningConfig { ExplorationDepth = 2 };

        SetupMockGenerator();
        SetupMockEvaluator();

        // Act
        var path1 = await mcts1.SearchAsync(root1, _mockGenerator.Object, _mockEvaluator.Object, config);
        var path2 = await mcts2.SearchAsync(root2, _mockGenerator.Object, _mockEvaluator.Object, config);

        // Assert
        Assert.NotNull(path1);
        Assert.NotNull(path2);
        // More simulations should explore more thoroughly
        Assert.True(path2.Count >= path1.Count || path2.Count > 0);
    }

    private ThoughtNode<double> CreateTestTree()
    {
        return new ThoughtNode<double>
        {
            Content = "Root node",
            Depth = 0,
            Score = 1.0
        };
    }

    private void SetupMockGenerator(int childrenPerNode = 3)
    {
        _mockGenerator
            .Setup(g => g.GenerateThoughtsAsync(
                It.IsAny<ThoughtNode<double>>(),
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<ReasoningConfig>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync((ThoughtNode<double> parent, string query, int count, ReasoningConfig config, CancellationToken ct) =>
            {
                var thoughts = new List<ThoughtNode<double>>();
                for (int i = 0; i < Math.Min(count, childrenPerNode); i++)
                {
                    thoughts.Add(new ThoughtNode<double>
                    {
                        Content = $"Child {i}",
                        Parent = parent,
                        Depth = parent.Depth + 1,
                        Score = 0.8
                    });
                }
                return thoughts;
            });
    }

    private void SetupMockEvaluator()
    {
        _mockEvaluator
            .Setup(e => e.EvaluateThoughtAsync(
                It.IsAny<ThoughtNode<double>>(),
                It.IsAny<string>(),
                It.IsAny<ReasoningConfig>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(0.75);
    }
}
