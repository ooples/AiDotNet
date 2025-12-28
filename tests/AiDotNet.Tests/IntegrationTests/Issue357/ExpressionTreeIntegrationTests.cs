using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the ExpressionTree<T, TInput, TOutput> class covering
/// tree construction, evaluation, complexity, serialization, and genetic operations.
/// </summary>
public class ExpressionTreeIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Node Type Construction

    [Fact]
    public void ExpressionTree_ConstantNode_CreatesCorrectly()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        Assert.Equal(ExpressionNodeType.Constant, tree.Type);
        Assert.Equal(5.0, tree.Value, Tolerance);
    }

    [Fact]
    public void ExpressionTree_VariableNode_CreatesCorrectly()
    {
        // Variable x[2]
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);

        Assert.Equal(ExpressionNodeType.Variable, tree.Type);
        Assert.Equal(2.0, tree.Value, Tolerance);
    }

    [Fact]
    public void ExpressionTree_AddNode_WithChildren()
    {
        // Create: 3 + 5
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        Assert.Equal(ExpressionNodeType.Add, add.Type);
        Assert.NotNull(add.Left);
        Assert.NotNull(add.Right);
    }

    [Theory]
    [InlineData(ExpressionNodeType.Add)]
    [InlineData(ExpressionNodeType.Subtract)]
    [InlineData(ExpressionNodeType.Multiply)]
    [InlineData(ExpressionNodeType.Divide)]
    public void ExpressionTree_OperationNodes_CreateCorrectly(ExpressionNodeType nodeType)
    {
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var operation = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            nodeType, 0.0, left, right);

        Assert.Equal(nodeType, operation.Type);
        Assert.NotNull(operation.Left);
        Assert.NotNull(operation.Right);
    }

    #endregion

    #region Evaluation

    [Fact]
    public void ExpressionTree_Evaluate_ConstantReturnsValue()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 42.0);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        double result = tree.Evaluate(input);

        Assert.Equal(42.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_VariableReturnsInputValue()
    {
        // Variable x[1]
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 1.0);
        var input = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        double result = tree.Evaluate(input);

        Assert.Equal(20.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_Addition()
    {
        // Create: 3 + 5 = 8
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        var input = new Vector<double>(new[] { 1.0 });
        double result = add.Evaluate(input);

        Assert.Equal(8.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_Subtraction()
    {
        // Create: 10 - 3 = 7
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 10.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var subtract = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Subtract, 0.0, left, right);

        var input = new Vector<double>(new[] { 1.0 });
        double result = subtract.Evaluate(input);

        Assert.Equal(7.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_Multiplication()
    {
        // Create: 4 * 6 = 24
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 4.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 6.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, left, right);

        var input = new Vector<double>(new[] { 1.0 });
        double result = multiply.Evaluate(input);

        Assert.Equal(24.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_Division()
    {
        // Create: 12 / 4 = 3
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 12.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 4.0);
        var divide = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Divide, 0.0, left, right);

        var input = new Vector<double>(new[] { 1.0 });
        double result = divide.Evaluate(input);

        Assert.Equal(3.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_ComplexExpression()
    {
        // Create: (x[0] * 2) + x[1] = (3 * 2) + 4 = 10
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, x0, two);
        var x1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 1.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, multiply, x1);

        var input = new Vector<double>(new[] { 3.0, 4.0 });
        double result = add.Evaluate(input);

        Assert.Equal(10.0, result, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Evaluate_NestedExpression()
    {
        // Create: ((2 + 3) * (6 - 1)) = 5 * 5 = 25
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var three = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, two, three);

        var six = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 6.0);
        var one = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 1.0);
        var subtract = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Subtract, 0.0, six, one);

        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, add, subtract);

        var input = new Vector<double>(new[] { 1.0 });
        double result = multiply.Evaluate(input);

        Assert.Equal(25.0, result, Tolerance);
    }

    #endregion

    #region Complexity

    [Fact]
    public void ExpressionTree_Complexity_SingleNode()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        Assert.Equal(1, tree.Complexity);
    }

    [Fact]
    public void ExpressionTree_Complexity_TwoLevelTree()
    {
        // Tree: (a + b) has complexity 3
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        Assert.Equal(3, add.Complexity);
    }

    [Fact]
    public void ExpressionTree_Complexity_NestedTree()
    {
        // Tree: ((a + b) * c) has complexity 5
        var a = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 1.0);
        var b = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, a, b);
        var c = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, add, c);

        Assert.Equal(5, multiply.Complexity);
    }

    #endregion

    #region Feature Count

    [Fact]
    public void ExpressionTree_FeatureCount_ConstantOnly()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        Assert.Equal(0, tree.FeatureCount);
    }

    [Fact]
    public void ExpressionTree_FeatureCount_SingleVariable()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);

        Assert.Equal(1, tree.FeatureCount);
    }

    [Fact]
    public void ExpressionTree_FeatureCount_MultipleVariables()
    {
        // x[0] + x[2] (uses variables at index 0 and 2)
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x2);

        // Should count 2 unique features (x[0] and x[2])
        Assert.Equal(2, add.FeatureCount);
    }

    [Fact]
    public void ExpressionTree_FeatureCount_RepeatedVariable()
    {
        // x[0] + x[0] (same variable twice)
        var x0a = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x0b = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0a, x0b);

        // Should count only 1 unique feature
        Assert.Equal(1, add.FeatureCount);
    }

    [Fact]
    public void ExpressionTree_IsFeatureUsed_ReturnsTrue()
    {
        // x[0] + x[2]
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x2);

        Assert.True(add.IsFeatureUsed(0));
        Assert.True(add.IsFeatureUsed(2));
    }

    [Fact]
    public void ExpressionTree_IsFeatureUsed_ReturnsFalse()
    {
        // x[0] + x[2]
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x2);

        Assert.False(add.IsFeatureUsed(1)); // x[1] not used
        Assert.False(add.IsFeatureUsed(3)); // x[3] not used
    }

    #endregion

    #region ToString

    [Fact]
    public void ExpressionTree_ToString_Constant()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        Assert.Equal("5", tree.ToString());
    }

    [Fact]
    public void ExpressionTree_ToString_Variable()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);

        Assert.Equal("x[2]", tree.ToString());
    }

    [Fact]
    public void ExpressionTree_ToString_Addition()
    {
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        Assert.Equal("(3 + 5)", add.ToString());
    }

    [Fact]
    public void ExpressionTree_ToString_ComplexExpression()
    {
        // (x[0] * 2)
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, x0, two);

        Assert.Equal("(x[0] * 2)", multiply.ToString());
    }

    #endregion

    #region Get All Nodes

    [Fact]
    public void ExpressionTree_GetAllNodes_SingleNode()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        var nodes = tree.GetAllNodes();

        Assert.Single(nodes);
    }

    [Fact]
    public void ExpressionTree_GetAllNodes_TreeWithChildren()
    {
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        var nodes = add.GetAllNodes();

        Assert.Equal(3, nodes.Count);
    }

    #endregion

    #region Coefficients

    [Fact]
    public void ExpressionTree_Coefficients_SingleConstant()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        var coeffs = tree.Coefficients;

        Assert.Single(coeffs);
        Assert.Equal(5.0, coeffs[0], Tolerance);
    }

    [Fact]
    public void ExpressionTree_Coefficients_MultipleConstants()
    {
        // 3 + 5
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        var coeffs = add.Coefficients;

        Assert.Equal(2, coeffs.Length);
        Assert.Equal(3.0, coeffs[0], Tolerance);
        Assert.Equal(5.0, coeffs[1], Tolerance);
    }

    [Fact]
    public void ExpressionTree_ParameterCount_EqualsConstantCount()
    {
        // (2 * x[0]) + 5
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, two, x0);
        var five = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, multiply, five);

        Assert.Equal(2, add.ParameterCount);
    }

    #endregion

    #region SetType and SetValue

    [Fact]
    public void ExpressionTree_SetType_ChangesNodeType()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        tree.SetType(ExpressionNodeType.Variable);

        Assert.Equal(ExpressionNodeType.Variable, tree.Type);
    }

    [Fact]
    public void ExpressionTree_SetValue_ChangesValue()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        tree.SetValue(10.0);

        Assert.Equal(10.0, tree.Value, Tolerance);
    }

    #endregion

    #region SetLeft and SetRight

    [Fact]
    public void ExpressionTree_SetLeft_UpdatesParent()
    {
        var parent = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0);
        var child = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        parent.SetLeft(child);

        Assert.Same(child, parent.Left);
        Assert.Same(parent, child.Parent);
    }

    [Fact]
    public void ExpressionTree_SetRight_UpdatesParent()
    {
        var parent = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0);
        var child = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        parent.SetRight(child);

        Assert.Same(child, parent.Right);
        Assert.Same(parent, child.Parent);
    }

    #endregion

    #region Predict

    [Fact]
    public void ExpressionTree_Predict_SingleSample()
    {
        // y = 2 * x[0] + 3
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, two, x0);
        var three = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, multiply, three);

        var input = new Matrix<double>(new double[,] { { 5.0 } });
        var predictions = add.Predict(input);

        // 2 * 5 + 3 = 13
        Assert.Equal(13.0, predictions[0], Tolerance);
    }

    [Fact]
    public void ExpressionTree_Predict_MultipleSamples()
    {
        // y = x[0] + x[1]
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 1.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x1);

        var input = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var predictions = add.Predict(input);

        Assert.Equal(3, predictions.Length);
        Assert.Equal(3.0, predictions[0], Tolerance);  // 1 + 2
        Assert.Equal(7.0, predictions[1], Tolerance);  // 3 + 4
        Assert.Equal(11.0, predictions[2], Tolerance); // 5 + 6
    }

    [Fact]
    public void ExpressionTree_Predict_InsufficientFeatures_Throws()
    {
        // Tree uses x[2]
        var x2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);

        // Input only has 2 columns
        var input = new Matrix<double>(new double[,] { { 1.0, 2.0 } });

        Assert.Throws<ArgumentOutOfRangeException>(() => x2.Predict(input));
    }

    #endregion

    #region Copy and Clone

    [Fact]
    public void ExpressionTree_Copy_CreatesIndependentTree()
    {
        var original = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        var copy = (ExpressionTree<double, Matrix<double>, Vector<double>>)original.Copy();

        Assert.NotSame(original, copy);
        Assert.Equal(original.Type, copy.Type);
        Assert.Equal(original.Value, copy.Value, Tolerance);
    }

    [Fact]
    public void ExpressionTree_Clone_DeepCopiesChildren()
    {
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var original = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        var clone = (ExpressionTree<double, Matrix<double>, Vector<double>>)original.Clone();

        Assert.NotSame(original, clone);
        Assert.NotSame(original.Left, clone.Left);
        Assert.NotSame(original.Right, clone.Right);
    }

    #endregion

    #region Serialization

    [Fact]
    public void ExpressionTree_SerializeDeserialize_PreservesStructure()
    {
        // Create: 3 + 5
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var original = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        // Serialize
        byte[] data = original.Serialize();

        // Deserialize
        var deserialized = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 0.0);
        deserialized.Deserialize(data);

        // Verify structure
        Assert.Equal(original.Type, deserialized.Type);
        Assert.NotNull(deserialized.Left);
        Assert.NotNull(deserialized.Right);
        Assert.Equal(ExpressionNodeType.Constant, deserialized.Left.Type);
        Assert.Equal(ExpressionNodeType.Constant, deserialized.Right.Type);
        Assert.Equal(3.0, deserialized.Left.Value, Tolerance);
        Assert.Equal(5.0, deserialized.Right.Value, Tolerance);
    }

    [Fact]
    public void ExpressionTree_SerializeDeserialize_PreservesEvaluation()
    {
        // Create: 2 * x[0] + 3
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, two, x0);
        var three = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var original = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, multiply, three);

        // Serialize and deserialize
        byte[] data = original.Serialize();
        var deserialized = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 0.0);
        deserialized.Deserialize(data);

        // Both should evaluate to same result
        var input = new Vector<double>(new[] { 5.0 });
        double originalResult = original.Evaluate(input);
        double deserializedResult = deserialized.Evaluate(input);

        Assert.Equal(originalResult, deserializedResult, Tolerance);
    }

    #endregion

    #region Unique IDs

    [Fact]
    public void ExpressionTree_EachNode_HasUniqueId()
    {
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        var nodes = add.GetAllNodes();
        var ids = nodes.Select(n => n.Id).ToList();

        Assert.Equal(3, ids.Count);
        Assert.Equal(3, ids.Distinct().Count()); // All IDs should be unique
    }

    [Fact]
    public void ExpressionTree_FindNodeById_ReturnsCorrectNode()
    {
        var left = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var right = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, left, right);

        var foundNode = add.FindNodeById(right.Id);

        Assert.NotNull(foundNode);
        Assert.Equal(5.0, foundNode.Value, Tolerance);
    }

    [Fact]
    public void ExpressionTree_FindNodeById_NonExistent_ReturnsNull()
    {
        var tree = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 5.0);

        var foundNode = tree.FindNodeById(-999);

        Assert.Null(foundNode);
    }

    #endregion

    #region Feature Importance

    [Fact]
    public void ExpressionTree_GetFeatureImportance_SingleVariable()
    {
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);

        var importance = x0.GetFeatureImportance();

        Assert.Single(importance);
        Assert.True(importance.ContainsKey("x[0]"));
        Assert.Equal(1.0, importance["x[0]"], Tolerance);
    }

    [Fact]
    public void ExpressionTree_GetFeatureImportance_MultipleVariables()
    {
        // x[0] + x[1]
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 1.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x1);

        var importance = add.GetFeatureImportance();

        Assert.Equal(2, importance.Count);
        Assert.Equal(0.5, importance["x[0]"], Tolerance);
        Assert.Equal(0.5, importance["x[1]"], Tolerance);
    }

    [Fact]
    public void ExpressionTree_GetFeatureImportance_RepeatedVariable()
    {
        // x[0] + x[0] (x[0] appears twice, x[1] once)
        var x0a = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x0b = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 1.0);
        var add1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0a, x0b);
        var add2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, add1, x1);

        var importance = add2.GetFeatureImportance();

        Assert.Equal(2, importance.Count);
        Assert.Equal(2.0 / 3.0, importance["x[0]"], Tolerance); // 2 occurrences out of 3
        Assert.Equal(1.0 / 3.0, importance["x[1]"], Tolerance); // 1 occurrence out of 3
    }

    #endregion

    #region Active Feature Indices

    [Fact]
    public void ExpressionTree_GetActiveFeatureIndices_ReturnsUsedIndices()
    {
        // x[0] + x[2]
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x2);

        var indices = add.GetActiveFeatureIndices().ToList();

        Assert.Equal(2, indices.Count);
        Assert.Contains(0, indices);
        Assert.Contains(2, indices);
    }

    [Fact]
    public void ExpressionTree_SetActiveFeatureIndices_DeactivatesUnused()
    {
        // x[0] + x[1] + x[2]
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var x1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 1.0);
        var add1 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, x0, x1);
        var x2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 2.0);
        var add2 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, add1, x2);

        // Only keep x[0] and x[2] active
        add2.SetActiveFeatureIndices(new[] { 0, 2 });

        // x[1] should be converted to constant 0
        var nodes = add2.GetAllNodes();
        var x1Node = nodes.FirstOrDefault(n =>
            n.Type == ExpressionNodeType.Constant && n.Value == 0.0);

        Assert.NotNull(x1Node);

        // Active features should now be 0 and 2 only
        var activeIndices = add2.GetActiveFeatureIndices().ToList();
        Assert.Equal(2, activeIndices.Count);
        Assert.Contains(0, activeIndices);
        Assert.Contains(2, activeIndices);
        Assert.DoesNotContain(1, activeIndices);
    }

    #endregion

    #region Metadata

    [Fact]
    public void ExpressionTree_GetModelMetadata_ReturnsCorrectInfo()
    {
        // 2 * x[0] + 3
        var two = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 2.0);
        var x0 = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Variable, 0.0);
        var multiply = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Multiply, 0.0, two, x0);
        var three = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Constant, 3.0);
        var add = new ExpressionTree<double, Matrix<double>, Vector<double>>(
            ExpressionNodeType.Add, 0.0, multiply, three);

        var metadata = add.GetModelMetadata();

        Assert.Equal(ModelType.ExpressionTree, metadata.ModelType);
        Assert.Equal(1, metadata.FeatureCount);
        Assert.Equal(5, metadata.Complexity);
    }

    #endregion
}
