using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the DecisionTreeNode<T> class covering node construction,
/// tree structure, and prediction logic.
/// </summary>
public class DecisionTreeNodeIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Node Construction

    [Fact]
    public void DecisionTreeNode_DefaultConstructor_CreatesLeafNode()
    {
        var node = new DecisionTreeNode<double>();

        Assert.True(node.IsLeaf);
        Assert.Null(node.Left);
        Assert.Null(node.Right);
        Assert.Equal(0.0, node.Prediction);
        Assert.Equal(0.0, node.SplitValue);
        Assert.Equal(0.0, node.Threshold);
    }

    [Fact]
    public void DecisionTreeNode_SplitConstructor_CreatesInternalNode()
    {
        var node = new DecisionTreeNode<double>(featureIndex: 2, splitValue: 5.0);

        Assert.False(node.IsLeaf);
        Assert.Equal(2, node.FeatureIndex);
        Assert.Equal(5.0, node.SplitValue);
    }

    [Fact]
    public void DecisionTreeNode_PredictionConstructor_CreatesLeafWithValue()
    {
        var node = new DecisionTreeNode<double>(prediction: 42.5);

        Assert.True(node.IsLeaf);
        Assert.Equal(42.5, node.Prediction);
    }

    #endregion

    #region Tree Structure

    [Fact]
    public void DecisionTreeNode_CanBuildSimpleTree()
    {
        // Create a simple decision tree:
        //        [Feature 0 <= 5]
        //         /           \
        //   [Pred: 10]    [Pred: 20]

        var root = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            Left = new DecisionTreeNode<double>(prediction: 10.0),
            Right = new DecisionTreeNode<double>(prediction: 20.0)
        };

        Assert.False(root.IsLeaf);
        Assert.NotNull(root.Left);
        Assert.NotNull(root.Right);
        Assert.True(root.Left.IsLeaf);
        Assert.True(root.Right.IsLeaf);
        Assert.Equal(10.0, root.Left.Prediction);
        Assert.Equal(20.0, root.Right.Prediction);
    }

    [Fact]
    public void DecisionTreeNode_CanBuildMultiLevelTree()
    {
        // Create a deeper tree:
        //            [Feature 0 <= 5]
        //             /           \
        //      [Feature 1 <= 3]  [Pred: 30]
        //       /           \
        //   [Pred: 10]    [Pred: 20]

        var root = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            Left = new DecisionTreeNode<double>(featureIndex: 1, splitValue: 3.0)
            {
                Left = new DecisionTreeNode<double>(prediction: 10.0),
                Right = new DecisionTreeNode<double>(prediction: 20.0)
            },
            Right = new DecisionTreeNode<double>(prediction: 30.0)
        };

        Assert.False(root.IsLeaf);
        Assert.False(root.Left?.IsLeaf);
        Assert.True(root.Right?.IsLeaf);
        Assert.True(root.Left?.Left?.IsLeaf);
        Assert.True(root.Left?.Right?.IsLeaf);
    }

    #endregion

    #region Sample Management

    [Fact]
    public void DecisionTreeNode_Samples_DefaultsToEmptyList()
    {
        var node = new DecisionTreeNode<double>();

        Assert.NotNull(node.Samples);
        Assert.Empty(node.Samples);
    }

    [Fact]
    public void DecisionTreeNode_SampleValues_DefaultsToEmptyList()
    {
        var node = new DecisionTreeNode<double>();

        Assert.NotNull(node.SampleValues);
        Assert.Empty(node.SampleValues);
    }

    [Fact]
    public void DecisionTreeNode_SampleCounts_TrackLeftRightDistribution()
    {
        var node = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            LeftSampleCount = 30,
            RightSampleCount = 70
        };

        Assert.Equal(30, node.LeftSampleCount);
        Assert.Equal(70, node.RightSampleCount);
    }

    #endregion

    #region Node Statistics

    [Fact]
    public void DecisionTreeNode_UpdateStatistics_CalculatesSumSquaredError()
    {
        var node = new DecisionTreeNode<double>(prediction: 5.0);

        // Add samples with targets
        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 1.0 }), 4.0));
        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 2.0 }), 6.0));
        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 3.0 }), 5.0));

        node.UpdateNodeStatistics();

        // SSE = (4-5)^2 + (6-5)^2 + (5-5)^2 = 1 + 1 + 0 = 2
        Assert.Equal(2.0, node.SumSquaredError, Tolerance);
    }

    [Fact]
    public void DecisionTreeNode_UpdateStatistics_EmptySamples_ReturnsZero()
    {
        var node = new DecisionTreeNode<double>(prediction: 5.0);

        node.UpdateNodeStatistics();

        Assert.Equal(0.0, node.SumSquaredError);
    }

    [Fact]
    public void DecisionTreeNode_UpdateStatistics_PerfectPrediction_ReturnsZeroSSE()
    {
        var node = new DecisionTreeNode<double>(prediction: 10.0);

        // All samples have target equal to prediction
        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 1.0 }), 10.0));
        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 2.0 }), 10.0));

        node.UpdateNodeStatistics();

        Assert.Equal(0.0, node.SumSquaredError, Tolerance);
    }

    #endregion

    #region Properties

    [Fact]
    public void DecisionTreeNode_Threshold_CanBeSetAndRetrieved()
    {
        var node = new DecisionTreeNode<double>
        {
            Threshold = 7.5
        };

        Assert.Equal(7.5, node.Threshold);
    }

    [Fact]
    public void DecisionTreeNode_FeatureIndex_CanBeSetAndRetrieved()
    {
        var node = new DecisionTreeNode<double>
        {
            FeatureIndex = 3
        };

        Assert.Equal(3, node.FeatureIndex);
    }

    [Fact]
    public void DecisionTreeNode_Predictions_Vector_CanBeSetAndRetrieved()
    {
        var node = new DecisionTreeNode<double>();
        var predictions = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        node.Predictions = predictions;

        Assert.NotNull(node.Predictions);
        Assert.Equal(3, node.Predictions.Length);
        Assert.Equal(1.0, node.Predictions[0]);
    }

    [Fact]
    public void DecisionTreeNode_LinearModel_DefaultsToNull()
    {
        var node = new DecisionTreeNode<double>();

        Assert.Null(node.LinearModel);
    }

    #endregion

    #region Tree Traversal Simulation

    [Fact]
    public void DecisionTreeNode_TreeTraversal_ReachesCorrectLeaf()
    {
        // Build tree for: if x[0] <= 5 then 10 else 20
        var root = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            Left = new DecisionTreeNode<double>(prediction: 10.0),
            Right = new DecisionTreeNode<double>(prediction: 20.0)
        };

        // Simulate traversal for x[0] = 3 (should go left)
        var input1 = new Vector<double>(new[] { 3.0 });
        var result1 = TraverseTree(root, input1);
        Assert.Equal(10.0, result1);

        // Simulate traversal for x[0] = 7 (should go right)
        var input2 = new Vector<double>(new[] { 7.0 });
        var result2 = TraverseTree(root, input2);
        Assert.Equal(20.0, result2);
    }

    [Fact]
    public void DecisionTreeNode_DeepTreeTraversal_ReachesCorrectLeaf()
    {
        // Build tree for more complex decision
        var root = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            Left = new DecisionTreeNode<double>(featureIndex: 1, splitValue: 2.0)
            {
                Left = new DecisionTreeNode<double>(prediction: 100.0),
                Right = new DecisionTreeNode<double>(prediction: 200.0)
            },
            Right = new DecisionTreeNode<double>(featureIndex: 1, splitValue: 8.0)
            {
                Left = new DecisionTreeNode<double>(prediction: 300.0),
                Right = new DecisionTreeNode<double>(prediction: 400.0)
            }
        };

        // x[0]=3, x[1]=1 -> Left(5), Left(2) -> 100
        Assert.Equal(100.0, TraverseTree(root, new Vector<double>(new[] { 3.0, 1.0 })));

        // x[0]=3, x[1]=4 -> Left(5), Right(2) -> 200
        Assert.Equal(200.0, TraverseTree(root, new Vector<double>(new[] { 3.0, 4.0 })));

        // x[0]=7, x[1]=5 -> Right(5), Left(8) -> 300
        Assert.Equal(300.0, TraverseTree(root, new Vector<double>(new[] { 7.0, 5.0 })));

        // x[0]=7, x[1]=10 -> Right(5), Right(8) -> 400
        Assert.Equal(400.0, TraverseTree(root, new Vector<double>(new[] { 7.0, 10.0 })));
    }

    /// <summary>
    /// Helper method to simulate tree traversal
    /// </summary>
    private double TraverseTree(DecisionTreeNode<double> node, Vector<double> input)
    {
        if (node.IsLeaf)
        {
            return node.Prediction;
        }

        // Standard decision tree split: left if value <= splitValue
        if (input[node.FeatureIndex] <= node.SplitValue)
        {
            return TraverseTree(node.Left!, input);
        }
        else
        {
            return TraverseTree(node.Right!, input);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void DecisionTreeNode_LeafWithNoChildren_IsValidLeaf()
    {
        var leaf = new DecisionTreeNode<double>(prediction: 42.0);

        Assert.True(leaf.IsLeaf);
        Assert.Null(leaf.Left);
        Assert.Null(leaf.Right);
    }

    [Fact]
    public void DecisionTreeNode_InternalWithOnlyLeftChild_IsValid()
    {
        var node = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            Left = new DecisionTreeNode<double>(prediction: 10.0)
        };

        Assert.False(node.IsLeaf);
        Assert.NotNull(node.Left);
        Assert.Null(node.Right);
    }

    [Fact]
    public void DecisionTreeNode_InternalWithOnlyRightChild_IsValid()
    {
        var node = new DecisionTreeNode<double>(featureIndex: 0, splitValue: 5.0)
        {
            Right = new DecisionTreeNode<double>(prediction: 20.0)
        };

        Assert.False(node.IsLeaf);
        Assert.Null(node.Left);
        Assert.NotNull(node.Right);
    }

    #endregion
}
