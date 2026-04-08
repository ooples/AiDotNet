using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the ConditionalInferenceTreeNode<T> class covering
/// p-value handling, inheritance from DecisionTreeNode, and statistical testing context.
/// </summary>
public class ConditionalInferenceTreeNodeIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Construction and Initialization

    [Fact]
    public void ConditionalInferenceTreeNode_DefaultConstructor_CreatesPValueZero()
    {
        var node = new ConditionalInferenceTreeNode<double>();

        Assert.Equal(0.0, node.PValue);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_DefaultConstructor_IsLeafNode()
    {
        var node = new ConditionalInferenceTreeNode<double>();

        Assert.True(node.IsLeaf);
        Assert.Null(node.Left);
        Assert.Null(node.Right);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_InheritsFromDecisionTreeNode()
    {
        var node = new ConditionalInferenceTreeNode<double>();

        // Should have all DecisionTreeNode properties
        Assert.Equal(0.0, node.Prediction);
        Assert.Equal(0.0, node.SplitValue);
        Assert.Equal(0.0, node.Threshold);
        Assert.NotNull(node.Samples);
        Assert.NotNull(node.SampleValues);
    }

    #endregion

    #region PValue Property

    [Fact]
    public void ConditionalInferenceTreeNode_PValue_CanBeSetAndRetrieved()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            PValue = 0.05
        };

        Assert.Equal(0.05, node.PValue, Tolerance);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.001)]
    [InlineData(0.01)]
    [InlineData(0.05)]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void ConditionalInferenceTreeNode_PValue_AcceptsValidProbabilityValues(double pValue)
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            PValue = pValue
        };

        Assert.Equal(pValue, node.PValue, Tolerance);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_LowPValue_IndicatesSignificantSplit()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            PValue = 0.001 // Highly significant
        };

        Assert.True(node.PValue < 0.05);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_HighPValue_IndicatesNonSignificantSplit()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            PValue = 0.75 // Not significant
        };

        Assert.True(node.PValue > 0.05);
    }

    #endregion

    #region Inherited DecisionTreeNode Functionality

    [Fact]
    public void ConditionalInferenceTreeNode_CanSetFeatureIndex()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 3
        };

        Assert.Equal(3, node.FeatureIndex);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_CanSetSplitValue()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            SplitValue = 7.5
        };

        Assert.Equal(7.5, node.SplitValue, Tolerance);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_CanSetPrediction()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            Prediction = 42.0
        };

        Assert.Equal(42.0, node.Prediction, Tolerance);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_CanBuildTreeStructure()
    {
        // Create a conditional inference tree with p-values at each node
        var root = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 0,
            SplitValue = 5.0,
            PValue = 0.001, // Highly significant split
            IsLeaf = false, // Must explicitly set since using object initializer
            Left = new ConditionalInferenceTreeNode<double>
            {
                Prediction = 10.0,
                PValue = 0.0 // Leaf
            },
            Right = new ConditionalInferenceTreeNode<double>
            {
                Prediction = 20.0,
                PValue = 0.0 // Leaf
            }
        };

        Assert.False(root.IsLeaf);
        Assert.Equal(0.001, root.PValue, Tolerance);
        Assert.NotNull(root.Left);
        Assert.NotNull(root.Right);
    }

    #endregion

    #region Tree Traversal with P-Values

    [Fact]
    public void ConditionalInferenceTreeNode_TreeTraversal_PValuesPreserved()
    {
        // Build a multi-level tree with varying p-values
        var root = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 0,
            SplitValue = 5.0,
            PValue = 0.01,
            Left = new ConditionalInferenceTreeNode<double>
            {
                FeatureIndex = 1,
                SplitValue = 3.0,
                PValue = 0.03,
                Left = new ConditionalInferenceTreeNode<double> { Prediction = 100.0, PValue = 0.0 },
                Right = new ConditionalInferenceTreeNode<double> { Prediction = 200.0, PValue = 0.0 }
            },
            Right = new ConditionalInferenceTreeNode<double>
            {
                Prediction = 300.0,
                PValue = 0.0
            }
        };

        // Verify p-values at each level
        Assert.Equal(0.01, root.PValue, Tolerance);
        Assert.Equal(0.03, ((ConditionalInferenceTreeNode<double>)root.Left!).PValue, Tolerance);
        Assert.Equal(0.0, ((ConditionalInferenceTreeNode<double>)root.Right!).PValue, Tolerance);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_CanSimulateTraversal()
    {
        // Build tree similar to DecisionTreeNode tests
        var root = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 0,
            SplitValue = 5.0,
            PValue = 0.02,
            IsLeaf = false, // Must explicitly set since using object initializer
            Left = new ConditionalInferenceTreeNode<double>
            {
                Prediction = 10.0,
                PValue = 0.0
            },
            Right = new ConditionalInferenceTreeNode<double>
            {
                Prediction = 20.0,
                PValue = 0.0
            }
        };

        // Simulate traversal
        var input1 = new Vector<double>(new[] { 3.0 });
        var result1 = TraverseTree(root, input1);
        Assert.Equal(10.0, result1);

        var input2 = new Vector<double>(new[] { 7.0 });
        var result2 = TraverseTree(root, input2);
        Assert.Equal(20.0, result2);
    }

    private double TraverseTree(DecisionTreeNode<double> node, Vector<double> input)
    {
        if (node.IsLeaf)
        {
            return node.Prediction;
        }

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

    #region Float Type Support

    [Fact]
    public void ConditionalInferenceTreeNode_Float_WorksCorrectly()
    {
        var node = new ConditionalInferenceTreeNode<float>
        {
            PValue = 0.05f,
            FeatureIndex = 1,
            SplitValue = 3.5f,
            Prediction = 15.0f
        };

        Assert.Equal(0.05f, node.PValue, 1e-6f);
        Assert.Equal(1, node.FeatureIndex);
        Assert.Equal(3.5f, node.SplitValue, 1e-6f);
        Assert.Equal(15.0f, node.Prediction, 1e-6f);
    }

    #endregion

    #region Statistical Significance Testing Context

    [Fact]
    public void ConditionalInferenceTreeNode_AllSignificantSplits()
    {
        // A tree where all splits are statistically significant (p < 0.05)
        var root = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 0,
            SplitValue = 5.0,
            PValue = 0.001,
            Left = new ConditionalInferenceTreeNode<double>
            {
                FeatureIndex = 1,
                SplitValue = 2.5,
                PValue = 0.01,
                Left = new ConditionalInferenceTreeNode<double> { Prediction = 1.0 },
                Right = new ConditionalInferenceTreeNode<double> { Prediction = 2.0 }
            },
            Right = new ConditionalInferenceTreeNode<double>
            {
                FeatureIndex = 2,
                SplitValue = 7.5,
                PValue = 0.02,
                Left = new ConditionalInferenceTreeNode<double> { Prediction = 3.0 },
                Right = new ConditionalInferenceTreeNode<double> { Prediction = 4.0 }
            }
        };

        // Check that all internal nodes have significant p-values
        Assert.True(root.PValue < 0.05);
        Assert.True(((ConditionalInferenceTreeNode<double>)root.Left!).PValue < 0.05);
        Assert.True(((ConditionalInferenceTreeNode<double>)root.Right!).PValue < 0.05);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_MixedSignificance()
    {
        // A tree with mixed significance levels
        var root = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 0,
            SplitValue = 5.0,
            PValue = 0.001, // Very significant
            Left = new ConditionalInferenceTreeNode<double>
            {
                FeatureIndex = 1,
                SplitValue = 2.5,
                PValue = 0.15, // Not significant (might want to prune)
                Left = new ConditionalInferenceTreeNode<double> { Prediction = 1.0 },
                Right = new ConditionalInferenceTreeNode<double> { Prediction = 2.0 }
            },
            Right = new ConditionalInferenceTreeNode<double>
            {
                Prediction = 3.0,
                PValue = 0.0
            }
        };

        // Root is significant
        Assert.True(root.PValue < 0.05);
        // Left child split is NOT significant
        Assert.True(((ConditionalInferenceTreeNode<double>)root.Left!).PValue > 0.05);
    }

    #endregion

    #region Sample Count Management (Inherited)

    [Fact]
    public void ConditionalInferenceTreeNode_SampleCounts_WorksCorrectly()
    {
        var node = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 0,
            SplitValue = 5.0,
            PValue = 0.01,
            LeftSampleCount = 45,
            RightSampleCount = 55
        };

        Assert.Equal(45, node.LeftSampleCount);
        Assert.Equal(55, node.RightSampleCount);
    }

    [Fact]
    public void ConditionalInferenceTreeNode_Samples_CanBeAddedAndAccessed()
    {
        var node = new ConditionalInferenceTreeNode<double>();

        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 1.0, 2.0 }), 10.0));
        node.Samples.Add(new Sample<double>(new Vector<double>(new[] { 3.0, 4.0 }), 20.0));

        Assert.Equal(2, node.Samples.Count);
    }

    #endregion
}
