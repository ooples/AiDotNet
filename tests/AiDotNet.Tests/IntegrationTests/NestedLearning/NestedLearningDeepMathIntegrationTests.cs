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
<<<<<<< HEAD
        // After Associate([1,0], [0,1]): W_diag = target ⊗ input = [[0,0],[1,0]].
||||||| 0d65f659c
        // Hebbian rule: W += lr * target * input^T
        // With lr=0.01 (hardcoded in Associate), input=[1,0], target=[0,1]:
        // W += 0.01 * [0,1]^T * [1,0] = 0.01 * [[0*1, 0*0], [1*1, 1*0]]
        //    = [[0, 0], [0.01, 0]]
=======
        // Modern continuous Hopfield (Ramsauer et al. 2021): Associate stores the
        // (key=input, value=target) pair, and GetAssociationMatrix returns the
        // outer-product sum W = Σ value ⊗ key. There is no Hebbian learning rate —
        // the association strength lives in the softmax retrieval temperature, not
        // the matrix — so the outer product is UNSCALED. With input=[1,0],
        // target=[0,1]:  W = [0,1]^T ⊗ [1,0] = [[0,0],[1,0]]
>>>>>>> origin/master
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
<<<<<<< HEAD
        // First Associate([1,0], [0,1]): W += [[0,0],[1,0]]
        // Second Associate([0,1], [1,0]): W += [[0,1],[0,0]]
        // Total: W_diag = [[0,1],[1,0]].
||||||| 0d65f659c
        // First: W += 0.01 * [0,1]^T * [1,0] = [[0,0],[0.01,0]]
        // Second: W += 0.01 * [1,0]^T * [0,1] = [[0,0.01],[0,0]]
        // Total: W = [[0, 0.01], [0.01, 0]]
=======
        // Outer-product sum W = Σ value ⊗ key over the stored pairs (unscaled):
        // First:  [0,1]^T ⊗ [1,0] = [[0,0],[1,0]]
        // Second: [1,0]^T ⊗ [0,1] = [[0,1],[0,0]]
        // Total:  W = [[0, 1], [1, 0]]
>>>>>>> origin/master
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
<<<<<<< HEAD
        // Associate([1,0], [1,0]): W += [[1,0],[0,0]]
        // Associate([0,1], [0,1]): W += [[0,0],[0,1]]
        // Total: W_diag = I.
||||||| 0d65f659c
        // Associate [1,0] with [1,0]: W += 0.01 * [1,0]^T * [1,0] = [[0.01, 0], [0, 0]]
        // Associate [0,1] with [0,1]: W += 0.01 * [0,1]^T * [0,1] = [[0, 0], [0, 0.01]]
        // Total: W = [[0.01, 0], [0, 0.01]] = 0.01 * I
=======
        // Associate [1,0]->[1,0]: [1,0]^T ⊗ [1,0] = [[1,0],[0,0]]
        // Associate [0,1]->[0,1]: [0,1]^T ⊗ [0,1] = [[0,0],[0,1]]
        // Total (unscaled outer-product sum): W = [[1,0],[0,1]] = I
>>>>>>> origin/master
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
<<<<<<< HEAD
        // Stored: ([1,0], [0,1]) and ([0,1], [1,0]). Both inputs are equidistant
        // from query=[1,1] (dot product = 1 each), so softmax(β·[1,1]) = [0.5, 0.5]
        // and result = 0.5·[0,1] + 0.5·[1,0] = [0.5, 0.5].
||||||| 0d65f659c
        // After associations building W = [[0, 0.01], [0.01, 0]]:
        // Retrieve([1, 0]) = W * [1, 0] = [0, 0.01]
        // Retrieve([0, 1]) = W * [0, 1] = [0.01, 0]
        // Note: cosine similarity of [1,0] with stored [1,0] = 1.0 > 0.8 → blending!
        // But for orthogonal query [0,1] vs stored [1,0], similarity = 0 → no blending
=======
        // Ramsauer retrieval: result = softmax(β · Kᵀq) · V over stored memories.
        // Stored: (key=[1,0], value=[0,1]) and (key=[0,1], value=[1,0]).
>>>>>>> origin/master
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));
        mem.Associate(
            new Vector<double>(new double[] { 0.0, 1.0 }),
            new Vector<double>(new double[] { 1.0, 0.0 }));

<<<<<<< HEAD
        var result = mem.Retrieve(new Vector<double>(new double[] { 1.0, 1.0 }));
        Assert.Equal(0.5, result[0], 1e-6);
        Assert.Equal(0.5, result[1], 1e-6);
||||||| 0d65f659c
        // Query orthogonal to both stored inputs won't trigger blending
        // W * [1,1] = [0+0.01, 0.01+0] = [0.01, 0.01]
        // But [1,1] has cosine similarity with [1,0] = 1/sqrt(2) ≈ 0.707 < 0.8
        // and with [0,1] = 1/sqrt(2) ≈ 0.707 < 0.8
        // So pure matrix retrieval: [0.01, 0.01]
        var query = new Vector<double>(new double[] { 1.0, 1.0 });
        var result = mem.Retrieve(query);
        Assert.Equal(0.01, result[0], Tolerance);
        Assert.Equal(0.01, result[1], Tolerance);
=======
        // Query [1,1] scores both keys equally (k·q = 1 for both), so softmax is
        // [0.5, 0.5] regardless of the temperature β. The retrieval is the equal
        // blend of the two values: 0.5·[0,1] + 0.5·[1,0] = [0.5, 0.5].
        var query = new Vector<double>(new double[] { 1.0, 1.0 });
        var result = mem.Retrieve(query);
        Assert.Equal(0.5, result[0], Tolerance);
        Assert.Equal(0.5, result[1], Tolerance);
>>>>>>> origin/master
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_SingleMemory_ReturnsTargetExactly()
    {
<<<<<<< HEAD
        // Only one memory pair stored. Softmax over a single score is always 1.0
        // regardless of the score value, so the retrieved vector is the stored
        // target exactly — this is the pattern-completion property of Modern
        // Hopfield: with one stored pattern the network always returns it.
||||||| 0d65f659c
        // Associate [1,0] with [0,1], so memory stores input=[1,0], target=[0,1]
        // W = [[0,0],[0.01,0]]
        // Retrieve with query=[1,0] (same as stored input):
        //   cosine_sim([1,0], [1,0]) = 1.0 > 0.8 → blending triggered
        //   matrix_result = W * [1,0] = [0, 0.01]
        //   buffer_match = [0, 1] (the stored target)
        //   blended = 0.7 * [0, 0.01] + 0.3 * [0, 1] = [0, 0.007 + 0.3] = [0, 0.307]
=======
        // Single stored memory (key=[1,0], value=[0,1]). With one memory the
        // softmax over keys is [1] for ANY query, so Ramsauer retrieval returns
        // exactly the stored value [0,1] — querying with the exact key is the
        // attractor fixed point (Ramsauer et al. 2021, Theorem 4).
>>>>>>> origin/master
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 1.0, 0.0 }));
<<<<<<< HEAD
        // Softmax denominator has a +1e-10 numerical-stability epsilon, so the
        // single-memory weight is 1.0 - O(1e-10) rather than exactly 1.0.
        Assert.Equal(0.0, result[0], 1e-9);
        Assert.Equal(1.0, result[1], 1e-9);
||||||| 0d65f659c
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.307, result[1], Tolerance);
=======
        // The softmax normalizes with /(sumExp + 1e-10) for numerical stability,
        // so a single memory's weight is 1/(1+1e-10) ≈ 1 - 1e-10 rather than
        // exactly 1; allow that stability epsilon (1e-9 covers it comfortably).
        const double SoftmaxEpsTolerance = 1e-9;
        Assert.Equal(0.0, result[0], SoftmaxEpsTolerance);
        Assert.Equal(1.0, result[1], SoftmaxEpsTolerance);
>>>>>>> origin/master
    }

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Retrieve_SingleMemory_OrthogonalQueryStillReturnsTarget()
    {
<<<<<<< HEAD
        // Single memory, orthogonal query. Softmax over one score = 1.0 → still
        // returns the target. Unlike a Hebbian W·q (which would return zero on
        // an orthogonal query), Ramsauer retrieval is "pattern completion":
        // any query against a single stored pattern recalls that pattern.
||||||| 0d65f659c
        // Associate [1,0] with [0,1]
        // W = [[0,0],[0.01,0]]
        // Retrieve with query=[0,1] (orthogonal to stored input [1,0]):
        //   cosine_sim([0,1], [1,0]) = 0.0 < 0.8 → no blending
        //   result = W * [0,1] = [0, 0]
=======
        // Single stored memory (key=[1,0], value=[0,1]). Unlike a linear Hebbian
        // matrix (where an orthogonal query would zero out), Ramsauer retrieval is
        // softmax over keys: with a single memory the softmax is [1] regardless of
        // the query, so even an orthogonal query [0,1] still retrieves the stored
        // value [0,1]. The temperature only matters once ≥2 memories compete.
>>>>>>> origin/master
        var mem = new AssociativeMemory<double>(dimension: 2);
        mem.Associate(
            new Vector<double>(new double[] { 1.0, 0.0 }),
            new Vector<double>(new double[] { 0.0, 1.0 }));

        var result = mem.Retrieve(new Vector<double>(new double[] { 0.0, 1.0 }));
<<<<<<< HEAD
        // Same softmax-epsilon caveat as the previous test.
        Assert.Equal(0.0, result[0], 1e-9);
        Assert.Equal(1.0, result[1], 1e-9);
||||||| 0d65f659c
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
=======
        // Single-memory softmax weight is 1/(1+1e-10) ≈ 1 - 1e-10 (the stability
        // epsilon in the normalizer), so allow 1e-9 around the stored value.
        const double SoftmaxEpsTolerance = 1e-9;
        Assert.Equal(0.0, result[0], SoftmaxEpsTolerance);
        Assert.Equal(1.0, result[1], SoftmaxEpsTolerance);
>>>>>>> origin/master
    }

    #endregion

    #region AssociativeMemory - Update and Properties

    [Fact(Timeout = 120000)]
    public async Task AssociativeMemory_Update_NewKey_AppendsAsUnscaledMemory()
    {
<<<<<<< HEAD
        // Update against an empty memory bank: no existing key matches (cosine
        // > 0.99 threshold), so Update degenerates to Associate — appending an
        // unscaled (input, target) pair. The lr argument only modulates blend
        // strength when an existing matching key is found.
        // After Update([1,0], [1,0], lr=0.5): W_diag = [[1,0],[0,0]].
||||||| 0d65f659c
        // Update with lr=0.5: W += 0.5 * [1,0]^T * [1,0] = [[0.5, 0], [0, 0]]
=======
        // Update on a NEW key stores the full (unscaled) value — the learning rate
        // only modulates the blend when a near-duplicate key (cosine > 0.99)
        // already exists, so that retrieval returns actual stored values rather
        // than lr-shrunk ones. Update([1,0], [1,0], lr=0.5) therefore stores the
        // pair as-is, giving W = [1,0]^T ⊗ [1,0] = [[1,0],[0,0]] (lr is not applied
        // to a first/unique association).
>>>>>>> origin/master
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
<<<<<<< HEAD
        // Update each standard basis vector with itself. None of the keys match
        // any other (cosine of orthogonal basis = 0 < 0.99 threshold), so each
        // Update appends — giving us 3 memories with mutually orthogonal keys.
        // The diagnostic matrix is then Σ e_i ⊗ e_i = I.
||||||| 0d65f659c
        // Associate each standard basis vector with itself using Update(lr=1.0)
        // This creates W = I (identity matrix)
        // Retrieving any vector should return itself (W*v = v)
=======
        // Associate each standard basis vector with itself using Update(lr=1.0).
        // The outer-product sum is W = Σ e_i ⊗ e_i = I (identity matrix).
>>>>>>> origin/master
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

<<<<<<< HEAD
        // Retrieve via Ramsauer softmax-attention: with orthogonal keys
        // {e_0, e_1, e_2} and query [0.5, -0.3, 0.8], the per-key scores are
        // β · q_i = 8 · {0.5, -0.3, 0.8} = {4, -2.4, 6.4}. After softmax,
        // weight on e_2 dominates (~0.917) since 0.8 is the largest |q_i|.
        // The result is a soft-argmax mixture of the basis targets, NOT q itself
        // (W·q only equals q for a *linear* Hebbian retrieval — Ramsauer is
        // softmax-attention).
||||||| 0d65f659c
        // Retrieve with arbitrary vector: W * v = I * v = v
        // Note: no memories in buffer (Update doesn't add to buffer), so no blending
=======
        // Ramsauer retrieval over the 3 orthonormal memories is
        // result = softmax(β · [q·e0, q·e1, q·e2]) · [e0; e1; e2], which (because
        // the values ARE the basis) equals the softmax distribution itself — a
        // convex combination, not the raw query. So the result is a probability
        // distribution: every component in (0,1), components sum to 1, and the
        // ordering follows the scores (q2=0.8 > q0=0.5 > q1=-0.3 → result[2] >
        // result[0] > result[1]). The dominant basis is the one most aligned with
        // the query (e2), demonstrating content-addressable recall.
>>>>>>> origin/master
        var query = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var result = mem.Retrieve(query);
<<<<<<< HEAD

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
||||||| 0d65f659c
        Assert.Equal(0.5, result[0], Tolerance);
        Assert.Equal(-0.3, result[1], Tolerance);
        Assert.Equal(0.8, result[2], Tolerance);
=======

        double sum = result[0] + result[1] + result[2];
        Assert.Equal(1.0, sum, Tolerance);
        for (int i = 0; i < dim; i++)
        {
            Assert.True(result[i] > 0.0 && result[i] < 1.0,
                $"result[{i}]={result[i]} must be a softmax weight in (0,1)");
        }
        Assert.True(result[2] > result[0], "e2 (highest score) must dominate e0");
        Assert.True(result[0] > result[1], "e0 (higher score) must exceed e1");
>>>>>>> origin/master
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
