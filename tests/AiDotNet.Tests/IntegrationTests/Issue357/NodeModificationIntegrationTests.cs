using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the NodeModification class covering
/// modification tracking for computational graph nodes.
/// </summary>
public class NodeModificationIntegrationTests
{
    #region Construction and Default Values

    [Fact]
    public void NodeModification_DefaultConstructor_CreatesWithDefaultValues()
    {
        var modification = new NodeModification();

        Assert.Equal(0, modification.NodeId);
        Assert.Equal(default(ModificationType), modification.Type);
        Assert.Null(modification.NewNodeType);
    }

    #endregion

    #region NodeId Property

    [Fact]
    public void NodeModification_NodeId_CanBeSetAndRetrieved()
    {
        var modification = new NodeModification
        {
            NodeId = 42
        };

        Assert.Equal(42, modification.NodeId);
    }

    [Fact]
    public void NodeModification_NodeId_AcceptsZero()
    {
        var modification = new NodeModification
        {
            NodeId = 0
        };

        Assert.Equal(0, modification.NodeId);
    }

    [Fact]
    public void NodeModification_NodeId_AcceptsNegativeValues()
    {
        // While unusual, negative IDs should be accepted by the data structure
        var modification = new NodeModification
        {
            NodeId = -1
        };

        Assert.Equal(-1, modification.NodeId);
    }

    [Fact]
    public void NodeModification_NodeId_AcceptsLargeValues()
    {
        var modification = new NodeModification
        {
            NodeId = int.MaxValue
        };

        Assert.Equal(int.MaxValue, modification.NodeId);
    }

    #endregion

    #region ModificationType Property

    [Fact]
    public void NodeModification_Type_AddNode()
    {
        var modification = new NodeModification
        {
            Type = ModificationType.AddNode
        };

        Assert.Equal(ModificationType.AddNode, modification.Type);
    }

    [Fact]
    public void NodeModification_Type_RemoveNode()
    {
        var modification = new NodeModification
        {
            Type = ModificationType.RemoveNode
        };

        Assert.Equal(ModificationType.RemoveNode, modification.Type);
    }

    [Fact]
    public void NodeModification_Type_ChangeNodeType()
    {
        var modification = new NodeModification
        {
            Type = ModificationType.ChangeNodeType
        };

        Assert.Equal(ModificationType.ChangeNodeType, modification.Type);
    }

    [Theory]
    [InlineData(ModificationType.AddNode)]
    [InlineData(ModificationType.RemoveNode)]
    [InlineData(ModificationType.ChangeNodeType)]
    public void NodeModification_Type_AllValidValues(ModificationType modType)
    {
        var modification = new NodeModification
        {
            Type = modType
        };

        Assert.Equal(modType, modification.Type);
    }

    #endregion

    #region NewNodeType Property

    [Fact]
    public void NodeModification_NewNodeType_DefaultsToNull()
    {
        var modification = new NodeModification();

        Assert.Null(modification.NewNodeType);
    }

    [Fact]
    public void NodeModification_NewNodeType_CanBeSetToConstant()
    {
        var modification = new NodeModification
        {
            NewNodeType = ExpressionNodeType.Constant
        };

        Assert.Equal(ExpressionNodeType.Constant, modification.NewNodeType);
    }

    [Fact]
    public void NodeModification_NewNodeType_CanBeSetToVariable()
    {
        var modification = new NodeModification
        {
            NewNodeType = ExpressionNodeType.Variable
        };

        Assert.Equal(ExpressionNodeType.Variable, modification.NewNodeType);
    }

    [Theory]
    [InlineData(ExpressionNodeType.Constant)]
    [InlineData(ExpressionNodeType.Variable)]
    [InlineData(ExpressionNodeType.Add)]
    [InlineData(ExpressionNodeType.Subtract)]
    [InlineData(ExpressionNodeType.Multiply)]
    [InlineData(ExpressionNodeType.Divide)]
    public void NodeModification_NewNodeType_AllExpressionNodeTypes(ExpressionNodeType nodeType)
    {
        var modification = new NodeModification
        {
            NewNodeType = nodeType
        };

        Assert.Equal(nodeType, modification.NewNodeType);
    }

    [Fact]
    public void NodeModification_NewNodeType_CanBeSetToNull()
    {
        var modification = new NodeModification
        {
            NewNodeType = ExpressionNodeType.Add
        };

        modification.NewNodeType = null;

        Assert.Null(modification.NewNodeType);
    }

    #endregion

    #region Combined Properties

    [Fact]
    public void NodeModification_AddNodeModification_AllPropertiesSet()
    {
        var modification = new NodeModification
        {
            NodeId = 5,
            Type = ModificationType.AddNode,
            NewNodeType = ExpressionNodeType.Multiply
        };

        Assert.Equal(5, modification.NodeId);
        Assert.Equal(ModificationType.AddNode, modification.Type);
        Assert.Equal(ExpressionNodeType.Multiply, modification.NewNodeType);
    }

    [Fact]
    public void NodeModification_RemoveNodeModification_NewNodeTypeNotRelevant()
    {
        var modification = new NodeModification
        {
            NodeId = 10,
            Type = ModificationType.RemoveNode,
            NewNodeType = null // Not relevant for removal
        };

        Assert.Equal(10, modification.NodeId);
        Assert.Equal(ModificationType.RemoveNode, modification.Type);
        Assert.Null(modification.NewNodeType);
    }

    [Fact]
    public void NodeModification_ChangeNodeType_RequiresNewNodeType()
    {
        var modification = new NodeModification
        {
            NodeId = 15,
            Type = ModificationType.ChangeNodeType,
            NewNodeType = ExpressionNodeType.Add // What we're changing to
        };

        Assert.Equal(15, modification.NodeId);
        Assert.Equal(ModificationType.ChangeNodeType, modification.Type);
        Assert.NotNull(modification.NewNodeType);
        Assert.Equal(ExpressionNodeType.Add, modification.NewNodeType);
    }

    #endregion

    #region Usage Scenarios

    [Fact]
    public void NodeModification_ChangeOperatorFromAddToMultiply()
    {
        // Scenario: Change a node from addition to multiplication
        var modification = new NodeModification
        {
            NodeId = 3, // Some addition node in the tree
            Type = ModificationType.ChangeNodeType,
            NewNodeType = ExpressionNodeType.Multiply
        };

        Assert.Equal(3, modification.NodeId);
        Assert.Equal(ModificationType.ChangeNodeType, modification.Type);
        Assert.Equal(ExpressionNodeType.Multiply, modification.NewNodeType);
    }

    [Fact]
    public void NodeModification_RemoveConstantNode()
    {
        // Scenario: Remove a constant node from the tree
        var modification = new NodeModification
        {
            NodeId = 7,
            Type = ModificationType.RemoveNode
        };

        Assert.Equal(7, modification.NodeId);
        Assert.Equal(ModificationType.RemoveNode, modification.Type);
    }

    [Fact]
    public void NodeModification_AddVariableNode()
    {
        // Scenario: Add a new variable node to the tree
        var modification = new NodeModification
        {
            NodeId = 0, // Parent node where we're adding
            Type = ModificationType.AddNode,
            NewNodeType = ExpressionNodeType.Variable
        };

        Assert.Equal(0, modification.NodeId);
        Assert.Equal(ModificationType.AddNode, modification.Type);
        Assert.Equal(ExpressionNodeType.Variable, modification.NewNodeType);
    }

    #endregion

    #region Collection Usage

    [Fact]
    public void NodeModification_CanBeStoredInList()
    {
        var modifications = new List<NodeModification>
        {
            new NodeModification { NodeId = 1, Type = ModificationType.AddNode },
            new NodeModification { NodeId = 2, Type = ModificationType.RemoveNode },
            new NodeModification { NodeId = 3, Type = ModificationType.ChangeNodeType, NewNodeType = ExpressionNodeType.Divide }
        };

        Assert.Equal(3, modifications.Count);
        Assert.Equal(ModificationType.AddNode, modifications[0].Type);
        Assert.Equal(ModificationType.RemoveNode, modifications[1].Type);
        Assert.Equal(ModificationType.ChangeNodeType, modifications[2].Type);
    }

    [Fact]
    public void NodeModification_CanBeFilteredByType()
    {
        var modifications = new List<NodeModification>
        {
            new NodeModification { NodeId = 1, Type = ModificationType.AddNode },
            new NodeModification { NodeId = 2, Type = ModificationType.RemoveNode },
            new NodeModification { NodeId = 3, Type = ModificationType.AddNode },
            new NodeModification { NodeId = 4, Type = ModificationType.ChangeNodeType }
        };

        var addNodeMods = modifications.Where(m => m.Type == ModificationType.AddNode).ToList();

        Assert.Equal(2, addNodeMods.Count);
        Assert.Equal(1, addNodeMods[0].NodeId);
        Assert.Equal(3, addNodeMods[1].NodeId);
    }

    [Fact]
    public void NodeModification_CanBeGroupedByNodeId()
    {
        var modifications = new List<NodeModification>
        {
            new NodeModification { NodeId = 1, Type = ModificationType.AddNode },
            new NodeModification { NodeId = 1, Type = ModificationType.ChangeNodeType },
            new NodeModification { NodeId = 2, Type = ModificationType.RemoveNode }
        };

        var groupedByNode = modifications.GroupBy(m => m.NodeId).ToList();

        Assert.Equal(2, groupedByNode.Count);
        Assert.Equal(2, groupedByNode.First(g => g.Key == 1).Count());
        Assert.Single(groupedByNode.First(g => g.Key == 2));
    }

    #endregion

    #region Integration with ExpressionTreeVelocity

    [Fact]
    public void NodeModification_WorksWithExpressionTreeVelocity()
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
            NodeId = 5,
            Type = ModificationType.RemoveNode
        });

        Assert.Equal(2, velocity.StructureChanges.Count);
        Assert.Equal(1, velocity.StructureChanges[0].NodeId);
        Assert.Equal(5, velocity.StructureChanges[1].NodeId);
    }

    #endregion
}
