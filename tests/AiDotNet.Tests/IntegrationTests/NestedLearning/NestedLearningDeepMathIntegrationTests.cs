using AiDotNet.NestedLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NestedLearning;

/// <summary>
/// Deep math-correctness integration tests for NestedLearning classes:
/// AssociativeMemory (Hebbian learning, cosine similarity, retrieval blending)
/// and ContextFlow (EMA propagation, transpose gradient, outer product update).
/// </summary>
public class NestedLearningDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region AssociativeMemory - Hebbian Learning

    [Fact]
    public void AssociativeMemory_SingleAssociation_MatrixIsScaledOuterProduct()
    {
        // Hebbian rule: W += lr * target * input^T
        // With lr=0.01 (hardcoded in Associate), input=[1,0], target=[0,1]:
        // W += 0.01 * [0,1]^T * [1,0] = 0.01 * [[0*1, 0*0], [1*1, 1*0]]
        //    = [[0, 0], [0.01, 0]]
        var mem = new AssociativeMemory<double>(dimension: 2);
        var input = new Vector<double>(new double[] { 1.0, 0.0 });
        var target = new Vector<double>(new double[] { 0.0, 1.0 });

        mem.Associate(input, target);

        var W = mem.GetAssociationMatrix();
        Assert.Equal(0.0, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(0.01, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
    }

    [Fact]
    public void AssociativeMemory_TwoAssociations_MatrixAccumulatesOuterProducts()
    {
        // First: W += 0.01 * [0,1]^T * [1,0] = [[0,0],[0.01,0]]
        // Second: W += 0.01 * [1,0]^T * [0,1] = [[0,0.01],[0,0]]
        // Total: W = [[0, 0.01], [0.01, 0]]
        var mem = new AssociativeMemory<double>(dimension: 2);

        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));

        var W = mem.GetAssociationMatrix();
        Assert.Equal(0.0, W[0, 0], Tolerance);
        Assert.Equal(0.01, W[0, 1], Tolerance);
        Assert.Equal(0.01, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
    }

    [Fact]
    public void AssociativeMemory_SameInputTarget_DiagonalUpdate()
    {
        // Associate [1,0] with [1,0]: W += 0.01 * [1,0]^T * [1,0] = [[0.01, 0], [0, 0]]
        // Associate [0,1] with [0,1]: W += 0.01 * [0,1]^T * [0,1] = [[0, 0], [0, 0.01]]
        // Total: W = [[0.01, 0], [0, 0.01]] = 0.01 * I
        var mem = new AssociativeMemory<double>(dimension: 2);

        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var W = mem.GetAssociationMatrix();
        Assert.Equal(0.01, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(0.0, W[1, 0], Tolerance);
        Assert.Equal(0.01, W[1, 1], Tolerance);
    }

    #endregion

    #region AssociativeMemory - Retrieval

    [Fact]
    public void AssociativeMemory_Retrieve_MatrixMultiply_HandComputed()
    {
        // After associations building W = [[0, 0.01], [0.01, 0]]:
        // Retrieve([1, 0]) = W * [1, 0] = [0, 0.01]
        // Retrieve([0, 1]) = W * [0, 1] = [0.01, 0]
        // Note: cosine similarity of [1,0] with stored [1,0] = 1.0 > 0.8 → blending!
        // But for orthogonal query [0,1] vs stored [1,0], similarity = 0 → no blending
        var mem = new AssociativeMemory<double>(dimension: 2);

        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));

        // Query orthogonal to both stored inputs won't trigger blending
        // W * [1,1] = [0+0.01, 0.01+0] = [0.01, 0.01]
        // But [1,1] has cosine similarity with [1,0] = 1/sqrt(2) ≈ 0.707 < 0.8
        // and with [0,1] = 1/sqrt(2) ≈ 0.707 < 0.8
        // So pure matrix retrieval: [0.01, 0.01]
        var query = new Vector<double>(new double[] { 1.0, 1.0 });
        var result = mem.Retrieve(query);
        Assert.Equal(0.01, result[0], Tolerance);
        Assert.Equal(0.01, result[1], Tolerance);
    }

    [Fact]
    public void AssociativeMemory_Retrieve_ExactMatch_TriggersBlending()
    {
        // Associate [1,0] with [0,1], so memory stores input=[1,0], target=[0,1]
        // W = [[0,0],[0.01,0]]
        // Retrieve with query=[1,0] (same as stored input):
        //   cosine_sim([1,0], [1,0]) = 1.0 > 0.8 → blending triggered
        //   matrix_result = W * [1,0] = [0, 0.01]
        //   buffer_match = [0, 1] (the stored target)
        //   blended = 0.7 * [0, 0.01] + 0.3 * [0, 1] = [0, 0.007 + 0.3] = [0, 0.307]
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 1.0, 0.0 }));
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.307, result[1], Tolerance);
    }

    [Fact]
    public void AssociativeMemory_Retrieve_OrthogonalQuery_NoBlending()
    {
        // Associate [1,0] with [0,1]
        // W = [[0,0],[0.01,0]]
        // Retrieve with query=[0,1] (orthogonal to stored input [1,0]):
        //   cosine_sim([0,1], [1,0]) = 0.0 < 0.8 → no blending
        //   result = W * [0,1] = [0, 0]
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 0.0, 1.0 }));
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
    }

    #endregion

    #region AssociativeMemory - Update and Properties

    [Fact]
    public void AssociativeMemory_Update_CustomLearningRate_HandComputed()
    {
        // Update with lr=0.5: W += 0.5 * [1,0]^T * [1,0] = [[0.5, 0], [0, 0]]
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Update(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }),
            0.5);

        var W = mem.GetAssociationMatrix();
        Assert.Equal(0.5, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(0.0, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
    }

    [Fact]
    public void AssociativeMemory_CapacityLimit_FIFOEviction()
    {
        var mem = new AssociativeMemory<double>(dimension: 2, capacity: 2);
        Assert.Equal(2, mem.Capacity);

        mem.Associate(new Vector<double>(new double[] { 1.0, 0.0 }), new Vector<double>(new double[] { 1.0, 0.0 }));
        Assert.Equal(1, mem.MemoryCount);

        mem.Associate(new Vector<double>(new double[] { 0.0, 1.0 }), new Vector<double>(new double[] { 0.0, 1.0 }));
        Assert.Equal(2, mem.MemoryCount);

        mem.Associate(new Vector<double>(new double[] { 1.0, 1.0 }), new Vector<double>(new double[] { 1.0, 1.0 }));
        Assert.Equal(2, mem.MemoryCount); // Oldest evicted
    }

    [Fact]
    public void AssociativeMemory_Clear_ResetsMatrixToZero()
    {
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 1.0 }));

        var W = mem.GetAssociationMatrix();
        Assert.True(W[0, 0] > 0, "Matrix should be non-zero after association");

        mem.Clear();

        W = mem.GetAssociationMatrix();
        Assert.Equal(0.0, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(0.0, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
        Assert.Equal(0, mem.MemoryCount);
    }

    [Fact]
    public void AssociativeMemory_DimensionMismatch_Throws()
    {
        var mem = new AssociativeMemory<double>(dimension: 2);
        var wrong = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }); // dim=3
        var correct = new Vector<double>(new double[] { 1.0, 0.0 });

        Assert.Throws<ArgumentException>(() => mem.Associate(wrong, correct));
        Assert.Throws<ArgumentException>(() => mem.Associate(correct, wrong));
        Assert.Throws<ArgumentException>(() => mem.Retrieve(wrong));
    }

    [Fact]
    public void AssociativeMemory_IdentityRetrieval_AfterStandardBasisAssociations()
    {
        // Associate each standard basis vector with itself using Update(lr=1.0)
        // This creates W = I (identity matrix)
        // Retrieving any vector should return itself (W*v = v)
        var dim = 3;
        var mem = new AssociativeMemory<double>(dimension: dim);

        for (int i = 0; i < dim; i++)
        {
            var e = new Vector<double>(dim);
            e[i] = 1.0;
            mem.Update(e, e, 1.0);
        }

        var W = mem.GetAssociationMatrix();
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.Equal(expected, W[i, j], Tolerance);
            }
        }

        // Retrieve with arbitrary vector: W * v = I * v = v
        // Note: no memories in buffer (Update doesn't add to buffer), so no blending
        var query = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var result = mem.Retrieve(query);
        Assert.Equal(0.5, result[0], Tolerance);
        Assert.Equal(-0.3, result[1], Tolerance);
        Assert.Equal(0.8, result[2], Tolerance);
    }

    #endregion

    #region ContextFlow - EMA Propagation

    [Fact]
    public void ContextFlow_FirstPropagation_IsScaledTransformation()
    {
        // First propagation from zero context:
        // context = 0.9 * [0,...,0] + 0.1 * (W * input) = 0.1 * (W * input)
        var dim = 3;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 2);

        var input = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
        var matrices = flow.GetTransformationMatrices();

        // Compute expected: W_0 * input
        var transformed = matrices[0].Multiply(input);

        var result = flow.PropagateContext(input, currentLevel: 0);

        // result should be 0.1 * transformed
        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(0.1 * transformed[i], result[i], 1e-10);
        }
    }

    [Fact]
    public void ContextFlow_SecondPropagation_EMAUpdate()
    {
        // First propagation: ctx1 = 0.1 * W * input1
        // Second propagation: ctx2 = 0.9 * ctx1 + 0.1 * W * input2
        var dim = 2;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 1);
        var matrices = flow.GetTransformationMatrices();

        var input1 = new Vector<double>(new double[] { 1.0, 0.0 });
        var input2 = new Vector<double>(new double[] { 0.0, 1.0 });

        var ctx1 = flow.PropagateContext(input1, currentLevel: 0);

        var transformed2 = matrices[0].Multiply(input2);
        var ctx2 = flow.PropagateContext(input2, currentLevel: 0);

        // ctx2 = 0.9 * ctx1 + 0.1 * transformed2
        for (int i = 0; i < dim; i++)
        {
            double expected = 0.9 * ctx1[i] + 0.1 * transformed2[i];
            Assert.Equal(expected, ctx2[i], 1e-10);
        }
    }

    [Fact]
    public void ContextFlow_PropagateContext_DifferentLevels_Independent()
    {
        // Propagation at level 0 should not affect level 1
        var dim = 2;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 2);

        var input = new Vector<double>(new double[] { 1.0, 1.0 });

        flow.PropagateContext(input, currentLevel: 0);

        // Level 1 should still be zero
        var ctx1 = flow.GetContextState(1);
        Assert.Equal(0.0, ctx1[0], Tolerance);
        Assert.Equal(0.0, ctx1[1], Tolerance);
    }

    #endregion

    #region ContextFlow - Gradient Computation

    [Fact]
    public void ContextFlow_ComputeContextGradients_TransposeProduct()
    {
        // gradient = W^T * upstream_gradient
        var dim = 2;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 1);
        var matrices = flow.GetTransformationMatrices();
        var W = matrices[0];

        var upstream = new Vector<double>(new double[] { 1.0, 0.0 });
        var result = flow.ComputeContextGradients(upstream, level: 0);

        // expected = W^T * [1, 0] = first column of W^T = first row of W
        Assert.Equal(W[0, 0], result[0], 1e-10);
        Assert.Equal(W[0, 1], result[1], 1e-10);
    }

    [Fact]
    public void ContextFlow_ComputeContextGradients_LinearInUpstream()
    {
        // Gradient is linear: grad(a*v) = a * grad(v)
        var dim = 2;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 1);

        var v = new Vector<double>(new double[] { 1.0, -0.5 });
        var twoV = new Vector<double>(new double[] { 2.0, -1.0 });

        var grad1 = flow.ComputeContextGradients(v, level: 0);
        var grad2 = flow.ComputeContextGradients(twoV, level: 0);

        // grad2 should be 2 * grad1
        Assert.Equal(2.0 * grad1[0], grad2[0], 1e-10);
        Assert.Equal(2.0 * grad1[1], grad2[1], 1e-10);
    }

    #endregion

    #region ContextFlow - Reset and Validation

    [Fact]
    public void ContextFlow_Reset_ZerosAllContextStates()
    {
        var dim = 2;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 2);

        // Propagate to make non-zero
        flow.PropagateContext(new Vector<double>(new double[] { 1.0, 1.0 }), 0);
        flow.PropagateContext(new Vector<double>(new double[] { 1.0, 1.0 }), 1);

        flow.Reset();

        for (int level = 0; level < 2; level++)
        {
            var ctx = flow.GetContextState(level);
            Assert.Equal(0.0, ctx[0], Tolerance);
            Assert.Equal(0.0, ctx[1], Tolerance);
        }
    }

    [Fact]
    public void ContextFlow_InvalidLevel_Throws()
    {
        var flow = new ContextFlow<double>(contextDimension: 2, numLevels: 3);
        var v = new Vector<double>(new double[] { 1.0, 0.0 });

        Assert.Throws<ArgumentException>(() => flow.PropagateContext(v, -1));
        Assert.Throws<ArgumentException>(() => flow.PropagateContext(v, 3));
        Assert.Throws<ArgumentException>(() => flow.ComputeContextGradients(v, -1));
        Assert.Throws<ArgumentException>(() => flow.ComputeContextGradients(v, 3));
        Assert.Throws<ArgumentException>(() => flow.GetContextState(-1));
        Assert.Throws<ArgumentException>(() => flow.GetContextState(3));
    }

    [Fact]
    public void ContextFlow_NumberOfLevels_MatchesConstructor()
    {
        var flow = new ContextFlow<double>(contextDimension: 4, numLevels: 5);
        Assert.Equal(5, flow.NumberOfLevels);
    }

    [Fact]
    public void ContextFlow_UpdateFlow_ChangesTransformationMatrix()
    {
        // UpdateFlow uses outer product: W -= lr * grad @ context^T
        var dim = 2;
        var flow = new ContextFlow<double>(contextDimension: dim, numLevels: 1);

        // First propagate to set context state
        flow.PropagateContext(new Vector<double>(new double[] { 1.0, 0.0 }), 0);

        // Save original matrix
        var originalW = flow.GetTransformationMatrices()[0];
        double w00Before = originalW[0, 0];

        // Update with gradient
        var gradients = new Vector<double>[] { new Vector<double>(new double[] { 1.0, 0.0 }) };
        var learningRates = new double[] { 0.1 };

        flow.UpdateFlow(gradients, learningRates);

        // Matrix should have changed
        var newW = flow.GetTransformationMatrices()[0];
        bool changed = Math.Abs(newW[0, 0] - w00Before) > 1e-15;
        Assert.True(changed, "Transformation matrix should change after UpdateFlow");
    }

    #endregion
}
