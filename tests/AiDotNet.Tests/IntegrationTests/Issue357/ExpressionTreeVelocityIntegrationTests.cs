using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the ExpressionTreeVelocity<T> class covering
/// velocity tracking for expression tree optimization.
/// </summary>
public class ExpressionTreeVelocityIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Construction and Initialization

    [Fact]
    public void ExpressionTreeVelocity_DefaultConstructor_CreatesEmptyCollections()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        Assert.NotNull(velocity.NodeValueChanges);
        Assert.NotNull(velocity.StructureChanges);
        Assert.Empty(velocity.NodeValueChanges);
        Assert.Empty(velocity.StructureChanges);
    }

    [Fact]
    public void ExpressionTreeVelocity_Float_WorksCorrectly()
    {
        var velocity = new ExpressionTreeVelocity<float>();

        Assert.NotNull(velocity.NodeValueChanges);
        Assert.NotNull(velocity.StructureChanges);
    }

    #endregion

    #region NodeValueChanges

    [Fact]
    public void ExpressionTreeVelocity_NodeValueChanges_CanAddEntries()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.NodeValueChanges[1] = 0.5;
        velocity.NodeValueChanges[2] = -0.3;

        Assert.Equal(2, velocity.NodeValueChanges.Count);
        Assert.Equal(0.5, velocity.NodeValueChanges[1], Tolerance);
        Assert.Equal(-0.3, velocity.NodeValueChanges[2], Tolerance);
    }

    [Fact]
    public void ExpressionTreeVelocity_NodeValueChanges_CanUpdateEntries()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.NodeValueChanges[1] = 0.5;
        velocity.NodeValueChanges[1] = 1.2; // Update

        Assert.Single(velocity.NodeValueChanges);
        Assert.Equal(1.2, velocity.NodeValueChanges[1], Tolerance);
    }

    [Fact]
    public void ExpressionTreeVelocity_NodeValueChanges_CanRemoveEntries()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.NodeValueChanges[1] = 0.5;
        velocity.NodeValueChanges[2] = -0.3;
        velocity.NodeValueChanges.Remove(1);

        Assert.Single(velocity.NodeValueChanges);
        Assert.False(velocity.NodeValueChanges.ContainsKey(1));
        Assert.True(velocity.NodeValueChanges.ContainsKey(2));
    }

    [Fact]
    public void ExpressionTreeVelocity_NodeValueChanges_SupportsNegativeValues()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.NodeValueChanges[1] = -5.5;
        velocity.NodeValueChanges[2] = -0.001;

        Assert.Equal(-5.5, velocity.NodeValueChanges[1], Tolerance);
        Assert.Equal(-0.001, velocity.NodeValueChanges[2], Tolerance);
    }

    [Fact]
    public void ExpressionTreeVelocity_NodeValueChanges_SupportsZeroValues()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.NodeValueChanges[1] = 0.0;

        Assert.True(velocity.NodeValueChanges.ContainsKey(1));
        Assert.Equal(0.0, velocity.NodeValueChanges[1], Tolerance);
    }

    [Fact]
    public void ExpressionTreeVelocity_NodeValueChanges_SupportsLargeNodeIds()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.NodeValueChanges[1000000] = 1.5;

        Assert.True(velocity.NodeValueChanges.ContainsKey(1000000));
        Assert.Equal(1.5, velocity.NodeValueChanges[1000000], Tolerance);
    }

    #endregion

    #region StructureChanges

    [Fact]
    public void ExpressionTreeVelocity_StructureChanges_CanAddModifications()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.StructureChanges.Add(new NodeModification
        {
            NodeId = 1,
            Type = ModificationType.ChangeNodeType,
            NewNodeType = ExpressionNodeType.Add
        });

        Assert.Single(velocity.StructureChanges);
    }

    [Fact]
    public void ExpressionTreeVelocity_StructureChanges_CanAddMultipleModifications()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.StructureChanges.Add(new NodeModification
        {
            NodeId = 1,
            Type = ModificationType.ChangeNodeType,
            NewNodeType = ExpressionNodeType.Multiply
        });

        velocity.StructureChanges.Add(new NodeModification
        {
            NodeId = 2,
            Type = ModificationType.RemoveNode
        });

        velocity.StructureChanges.Add(new NodeModification
        {
            NodeId = 3,
            Type = ModificationType.AddNode
        });

        Assert.Equal(3, velocity.StructureChanges.Count);
    }

    [Fact]
    public void ExpressionTreeVelocity_StructureChanges_PreservesOrder()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.StructureChanges.Add(new NodeModification { NodeId = 3 });
        velocity.StructureChanges.Add(new NodeModification { NodeId = 1 });
        velocity.StructureChanges.Add(new NodeModification { NodeId = 2 });

        Assert.Equal(3, velocity.StructureChanges[0].NodeId);
        Assert.Equal(1, velocity.StructureChanges[1].NodeId);
        Assert.Equal(2, velocity.StructureChanges[2].NodeId);
    }

    [Fact]
    public void ExpressionTreeVelocity_StructureChanges_CanClear()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        velocity.StructureChanges.Add(new NodeModification { NodeId = 1 });
        velocity.StructureChanges.Add(new NodeModification { NodeId = 2 });
        velocity.StructureChanges.Clear();

        Assert.Empty(velocity.StructureChanges);
    }

    #endregion

    #region Combined Velocity Information

    [Fact]
    public void ExpressionTreeVelocity_CanTrackBothValueAndStructureChanges()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        // Track value changes
        velocity.NodeValueChanges[1] = 0.5;
        velocity.NodeValueChanges[3] = -0.2;

        // Track structure changes
        velocity.StructureChanges.Add(new NodeModification
        {
            NodeId = 2,
            Type = ModificationType.ChangeNodeType,
            NewNodeType = ExpressionNodeType.Add
        });

        Assert.Equal(2, velocity.NodeValueChanges.Count);
        Assert.Single(velocity.StructureChanges);
    }

    [Fact]
    public void ExpressionTreeVelocity_CanResetVelocity()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        // Add some data
        velocity.NodeValueChanges[1] = 0.5;
        velocity.StructureChanges.Add(new NodeModification { NodeId = 1 });

        // Reset by clearing both
        velocity.NodeValueChanges.Clear();
        velocity.StructureChanges.Clear();

        Assert.Empty(velocity.NodeValueChanges);
        Assert.Empty(velocity.StructureChanges);
    }

    [Fact]
    public void ExpressionTreeVelocity_CanReplaceCollections()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        // Replace NodeValueChanges
        velocity.NodeValueChanges = new Dictionary<int, double>
        {
            { 1, 1.0 },
            { 2, 2.0 }
        };

        // Replace StructureChanges
        velocity.StructureChanges = new List<NodeModification>
        {
            new NodeModification { NodeId = 10 }
        };

        Assert.Equal(2, velocity.NodeValueChanges.Count);
        Assert.Single(velocity.StructureChanges);
        Assert.Equal(10, velocity.StructureChanges[0].NodeId);
    }

    #endregion

    #region Optimization Simulation

    [Fact]
    public void ExpressionTreeVelocity_SimulateOptimizationStep()
    {
        var velocity = new ExpressionTreeVelocity<double>();

        // Simulate calculating velocity for an expression tree optimization
        // Assume we have coefficients at nodes 1, 2, 3 and want to adjust them
        double learningRate = 0.1;
        double[] gradients = { -0.5, 0.3, -0.2 };
        int[] nodeIds = { 1, 2, 3 };

        for (int i = 0; i < nodeIds.Length; i++)
        {
            // velocity = -learning_rate * gradient
            velocity.NodeValueChanges[nodeIds[i]] = -learningRate * gradients[i];
        }

        Assert.Equal(0.05, velocity.NodeValueChanges[1], Tolerance);  // -0.1 * -0.5
        Assert.Equal(-0.03, velocity.NodeValueChanges[2], Tolerance); // -0.1 * 0.3
        Assert.Equal(0.02, velocity.NodeValueChanges[3], Tolerance);  // -0.1 * -0.2
    }

    [Fact]
    public void ExpressionTreeVelocity_SimulateMomentumUpdate()
    {
        var currentVelocity = new ExpressionTreeVelocity<double>();
        var previousVelocity = new ExpressionTreeVelocity<double>();

        // Previous velocity
        previousVelocity.NodeValueChanges[1] = 0.1;
        previousVelocity.NodeValueChanges[2] = -0.05;

        // New gradients suggest changes
        double learningRate = 0.1;
        double momentum = 0.9;
        double[] newGradients = { -0.3, 0.2 };

        // Update with momentum: v_new = momentum * v_old - learning_rate * gradient
        int[] nodeIds = { 1, 2 };
        for (int i = 0; i < nodeIds.Length; i++)
        {
            int nodeId = nodeIds[i];
            double prevVel = previousVelocity.NodeValueChanges.ContainsKey(nodeId)
                ? previousVelocity.NodeValueChanges[nodeId]
                : 0.0;
            currentVelocity.NodeValueChanges[nodeId] =
                momentum * prevVel - learningRate * newGradients[i];
        }

        // Node 1: 0.9 * 0.1 - 0.1 * -0.3 = 0.09 + 0.03 = 0.12
        Assert.Equal(0.12, currentVelocity.NodeValueChanges[1], Tolerance);

        // Node 2: 0.9 * -0.05 - 0.1 * 0.2 = -0.045 - 0.02 = -0.065
        Assert.Equal(-0.065, currentVelocity.NodeValueChanges[2], Tolerance);
    }

    #endregion

    #region Type Support

    [Fact]
    public void ExpressionTreeVelocity_Float_NodeValueChanges()
    {
        var velocity = new ExpressionTreeVelocity<float>();

        velocity.NodeValueChanges[1] = 0.5f;
        velocity.NodeValueChanges[2] = -0.3f;

        Assert.Equal(0.5f, velocity.NodeValueChanges[1], 1e-6f);
        Assert.Equal(-0.3f, velocity.NodeValueChanges[2], 1e-6f);
    }

    [Fact]
    public void ExpressionTreeVelocity_Int_NodeValueChanges()
    {
        var velocity = new ExpressionTreeVelocity<int>();

        velocity.NodeValueChanges[1] = 5;
        velocity.NodeValueChanges[2] = -3;

        Assert.Equal(5, velocity.NodeValueChanges[1]);
        Assert.Equal(-3, velocity.NodeValueChanges[2]);
    }

    #endregion
}
