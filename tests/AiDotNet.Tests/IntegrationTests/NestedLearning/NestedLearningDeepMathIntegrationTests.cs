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

    #region AssociativeMemory - Storage (GetAssociationMatrix = diagnostic outer-product sum)

    // The implementation is Modern Continuous Hopfield retrieval (Ramsauer et
    // al. 2021): Associate appends (input, target) to a memory bank; Retrieve
    // returns softmax(β · K^T q) · V, NOT W·q for any stored W. GetAssociationMatrix
    // is documented as "for diagnostics/testing only" and computes the *unscaled*
    // Hebbian outer-product sum Σ target_i ⊗ input_i^T (no learning-rate
    // multiplication anywhere in Associate / Update — the older Hebbian-style
    // 0.01-scaled-update was removed when this class migrated to Ramsauer).

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_SingleAssociation_DiagnosticMatrixIsOuterProduct()
    {
        // After Associate([1,0], [0,1]): W_diag = target ⊗ input = [[0,0],[1,0]].
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
    public async Task AssociativeMemory_TwoAssociations_DiagnosticMatrixAccumulatesOuterProducts()
    {
        // First Associate([1,0], [0,1]): W += [[0,0],[1,0]]
        // Second Associate([0,1], [1,0]): W += [[0,1],[0,0]]
        // Total: W_diag = [[0,1],[1,0]].
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
    public async Task AssociativeMemory_SameInputTarget_DiagonalMatrix()
    {
        // Associate([1,0], [1,0]): W += [[1,0],[0,0]]
        // Associate([0,1], [0,1]): W += [[0,0],[0,1]]
        // Total: W_diag = I.
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

    #region AssociativeMemory - Retrieve (Ramsauer 2021 softmax attention)

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_TwoMemories_EquidistantQuery_AveragesTargets()
    {
        // Stored: ([1,0], [0,1]) and ([0,1], [1,0]). Both inputs are equidistant
        // from query=[1,1] (dot product = 1 each), so softmax(β·[1,1]) = [0.5, 0.5]
        // and result = 0.5·[0,1] + 0.5·[1,0] = [0.5, 0.5].
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 1.0, 1.0 }));
        Assert.Equal(0.5, result[0], 1e-6);
        Assert.Equal(0.5, result[1], 1e-6);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_SingleMemory_ReturnsTargetExactly()
    {
        // Only one memory pair stored. Softmax over a single score is always 1.0
        // regardless of the score value, so the retrieved vector is the stored
        // target exactly — this is the pattern-completion property of Modern
        // Hopfield: with one stored pattern the network always returns it.
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 1.0, 0.0 }));
        // Softmax denominator has a +1e-10 numerical-stability epsilon, so the
        // single-memory weight is 1.0 - O(1e-10) rather than exactly 1.0.
        Assert.Equal(0.0, result[0], 1e-9);
        Assert.Equal(1.0, result[1], 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_SingleMemory_OrthogonalQueryStillReturnsTarget()
    {
        // Single memory, orthogonal query. Softmax over one score = 1.0 → still
        // returns the target. Unlike a Hebbian W·q (which would return zero on
        // an orthogonal query), Ramsauer retrieval is "pattern completion":
        // any query against a single stored pattern recalls that pattern.
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 0.0, 1.0 }));
        // Same softmax-epsilon caveat as the previous test.
        Assert.Equal(0.0, result[0], 1e-9);
        Assert.Equal(1.0, result[1], 1e-9);
    }

    #endregion

    #region AssociativeMemory - Update and Properties

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Update_NewKey_AppendsAsUnscaledMemory()
    {
        // Update against an empty memory bank: no existing key matches (cosine
        // > 0.99 threshold), so Update degenerates to Associate — appending an
        // unscaled (input, target) pair. The lr argument only modulates blend
        // strength when an existing matching key is found.
        // After Update([1,0], [1,0], lr=0.5): W_diag = [[1,0],[0,0]].
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
    public async Task AssociativeMemory_StandardBasisStorage_PatternCompletesToNearestBasis()
    {
        // Update each standard basis vector with itself. None of the keys match
        // any other (cosine of orthogonal basis = 0 < 0.99 threshold), so each
        // Update appends — giving us 3 memories with mutually orthogonal keys.
        // The diagnostic matrix is then Σ e_i ⊗ e_i = I.
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
            for (int j = 0; j < dim; j++)
                Assert.Equal((i == j) ? 1.0 : 0.0, W[i, j], Tolerance);

        // Retrieve via Ramsauer softmax-attention: with orthogonal keys
        // {e_0, e_1, e_2} and query [0.5, -0.3, 0.8], the per-key scores are
        // β · q_i = 8 · {0.5, -0.3, 0.8} = {4, -2.4, 6.4}. After softmax,
        // weight on e_2 dominates (~0.917) since 0.8 is the largest |q_i|.
        // The result is a soft-argmax mixture of the basis targets, NOT q itself
        // (W·q only equals q for a *linear* Hebbian retrieval — Ramsauer is
        // softmax-attention).
        var query = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var result = mem.Retrieve(query);

        // The component closest to the highest-scoring stored key (e_2) should
        // dominate. Soft-argmax never produces a one-hot, so use looser bounds.
        Assert.True(result[2] > 0.85,
            $"expected dominant component at index 2 (the e_2 basis), got {result[2]}");
        Assert.True(result[0] > 0 && result[0] < 0.15,
            $"expected e_0 weight in (0, 0.15), got {result[0]}");
        Assert.True(Math.Abs(result[1]) < 1e-3,
            $"expected near-zero e_1 weight (negative score → suppressed), got {result[1]}");
        // Sum of softmax weights == 1, so all components sum to 1 (basis targets are e_i).
        Assert.Equal(1.0, result[0] + result[1] + result[2], 1e-6);
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
