using AiDotNet.NestedLearning;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_SingleAssociation_MatrixIsScaledOuterProduct()
    {
        // Modern continuous Hopfield (Ramsauer et al. 2021): Associate stores the
        // (key=input, value=target) pair, and GetAssociationMatrix returns the
        // outer-product sum W = Σ value ⊗ key. There is no Hebbian learning rate —
        // the association strength lives in the softmax retrieval temperature, not
        // the matrix — so the outer product is UNSCALED. With input=[1,0],
        // target=[0,1]:  W = [0,1]^T ⊗ [1,0] = [[0,0],[1,0]]
        var mem = new AssociativeMemory<double>(dimension: 2);
        var input = new Vector<double>(new double[] { 1.0, 0.0 });
        var target = new Vector<double>(new double[] { 0.0, 1.0 });

        mem.Associate(input, target);

        var W = mem.GetAssociationMatrix();
        Assert.Equal(0.0, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(1.0, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_TwoAssociations_MatrixAccumulatesOuterProducts()
    {
        // Outer-product sum W = Σ value ⊗ key over the stored pairs (unscaled):
        // First:  [0,1]^T ⊗ [1,0] = [[0,0],[1,0]]
        // Second: [1,0]^T ⊗ [0,1] = [[0,1],[0,0]]
        // Total:  W = [[0, 1], [1, 0]]
        var mem = new AssociativeMemory<double>(dimension: 2);

        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));

        var W = mem.GetAssociationMatrix();
        Assert.Equal(0.0, W[0, 0], Tolerance);
        Assert.Equal(1.0, W[0, 1], Tolerance);
        Assert.Equal(1.0, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_SameInputTarget_DiagonalUpdate()
    {
        // Associate [1,0]->[1,0]: [1,0]^T ⊗ [1,0] = [[1,0],[0,0]]
        // Associate [0,1]->[0,1]: [0,1]^T ⊗ [0,1] = [[0,0],[0,1]]
        // Total (unscaled outer-product sum): W = [[1,0],[0,1]] = I
        var mem = new AssociativeMemory<double>(dimension: 2);

        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var W = mem.GetAssociationMatrix();
        Assert.Equal(1.0, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(0.0, W[1, 0], Tolerance);
        Assert.Equal(1.0, W[1, 1], Tolerance);
    }

    #endregion

    #region AssociativeMemory - Retrieval

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_MatrixMultiply_HandComputed()
    {
        // Ramsauer retrieval: result = softmax(β · Kᵀq) · V over stored memories.
        // Stored: (key=[1,0], value=[0,1]) and (key=[0,1], value=[1,0]).
        var mem = new AssociativeMemory<double>(dimension: 2);

        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));

        // Query [1,1] scores both keys equally (k·q = 1 for both), so softmax is
        // [0.5, 0.5] regardless of the temperature β. The retrieval is the equal
        // blend of the two values: 0.5·[0,1] + 0.5·[1,0] = [0.5, 0.5].
        var query = new Vector<double>(new double[] { 1.0, 1.0 });
        var result = mem.Retrieve(query);
        Assert.Equal(0.5, result[0], Tolerance);
        Assert.Equal(0.5, result[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_ExactMatch_TriggersBlending()
    {
        // Single stored memory (key=[1,0], value=[0,1]). With one memory the
        // softmax over keys is [1] for ANY query, so Ramsauer retrieval returns
        // exactly the stored value [0,1] — querying with the exact key is the
        // attractor fixed point (Ramsauer et al. 2021, Theorem 4).
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 1.0, 0.0 }));
        // The softmax normalizes with /(sumExp + 1e-10) for numerical stability,
        // so a single memory's weight is 1/(1+1e-10) ≈ 1 - 1e-10 rather than
        // exactly 1; allow that stability epsilon (1e-9 covers it comfortably).
        const double SoftmaxEpsTolerance = 1e-9;
        Assert.Equal(0.0, result[0], SoftmaxEpsTolerance);
        Assert.Equal(1.0, result[1], SoftmaxEpsTolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_OrthogonalQuery_NoBlending()
    {
        // Single stored memory (key=[1,0], value=[0,1]). Unlike a linear Hebbian
        // matrix (where an orthogonal query would zero out), Ramsauer retrieval is
        // softmax over keys: with a single memory the softmax is [1] regardless of
        // the query, so even an orthogonal query [0,1] still retrieves the stored
        // value [0,1]. The temperature only matters once ≥2 memories compete.
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 0.0, 1.0 }));
        // Single-memory softmax weight is 1/(1+1e-10) ≈ 1 - 1e-10 (the stability
        // epsilon in the normalizer), so allow 1e-9 around the stored value.
        const double SoftmaxEpsTolerance = 1e-9;
        Assert.Equal(0.0, result[0], SoftmaxEpsTolerance);
        Assert.Equal(1.0, result[1], SoftmaxEpsTolerance);
    }

    #endregion

    #region AssociativeMemory - Update and Properties

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Update_CustomLearningRate_HandComputed()
    {
        // Update on a NEW key stores the full (unscaled) value — the learning rate
        // only modulates the blend when a near-duplicate key (cosine > 0.99)
        // already exists, so that retrieval returns actual stored values rather
        // than lr-shrunk ones. Update([1,0], [1,0], lr=0.5) therefore stores the
        // pair as-is, giving W = [1,0]^T ⊗ [1,0] = [[1,0],[0,0]] (lr is not applied
        // to a first/unique association).
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Update(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }),
            0.5);

        var W = mem.GetAssociationMatrix();
        Assert.Equal(1.0, W[0, 0], Tolerance);
        Assert.Equal(0.0, W[0, 1], Tolerance);
        Assert.Equal(0.0, W[1, 0], Tolerance);
        Assert.Equal(0.0, W[1, 1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_CapacityLimit_FIFOEviction()
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

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Clear_ResetsMatrixToZero()
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

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_DimensionMismatch_Throws()
    {
        var mem = new AssociativeMemory<double>(dimension: 2);
        var wrong = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }); // dim=3
        var correct = new Vector<double>(new double[] { 1.0, 0.0 });

        Assert.Throws<ArgumentException>(() => mem.Associate(wrong, correct));
        Assert.Throws<ArgumentException>(() => mem.Associate(correct, wrong));
        Assert.Throws<ArgumentException>(() => mem.Retrieve(wrong));
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_IdentityRetrieval_AfterStandardBasisAssociations()
    {
        // Associate each standard basis vector with itself using Update(lr=1.0).
        // The outer-product sum is W = Σ e_i ⊗ e_i = I (identity matrix).
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

        // Ramsauer retrieval over the 3 orthonormal memories is
        // result = softmax(β · [q·e0, q·e1, q·e2]) · [e0; e1; e2], which (because
        // the values ARE the basis) equals the softmax distribution itself — a
        // convex combination, not the raw query. So the result is a probability
        // distribution: every component in (0,1), components sum to 1, and the
        // ordering follows the scores (q2=0.8 > q0=0.5 > q1=-0.3 → result[2] >
        // result[0] > result[1]). The dominant basis is the one most aligned with
        // the query (e2), demonstrating content-addressable recall.
        var query = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var result = mem.Retrieve(query);

        double sum = result[0] + result[1] + result[2];
        Assert.Equal(1.0, sum, Tolerance);
        for (int i = 0; i < dim; i++)
        {
            Assert.True(result[i] > 0.0 && result[i] < 1.0,
                $"result[{i}]={result[i]} must be a softmax weight in (0,1)");
        }
        Assert.True(result[2] > result[0], "e2 (highest score) must dominate e0");
        Assert.True(result[0] > result[1], "e0 (higher score) must exceed e1");
    }

    #endregion

    #region ContextFlow - EMA Propagation

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_FirstPropagation_IsScaledTransformation()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_SecondPropagation_EMAUpdate()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_PropagateContext_DifferentLevels_Independent()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_ComputeContextGradients_TransposeProduct()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_ComputeContextGradients_LinearInUpstream()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_Reset_ZerosAllContextStates()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_InvalidLevel_Throws()
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

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_NumberOfLevels_MatchesConstructor()
    {
        var flow = new ContextFlow<double>(contextDimension: 4, numLevels: 5);
        Assert.Equal(5, flow.NumberOfLevels);
    }

    [Fact(Timeout = 120000)]
    public async Task ContextFlow_UpdateFlow_ChangesTransformationMatrix()
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
