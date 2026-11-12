using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// The convex base learners of MetaOptNet (Lee et al. 2019), solved in <typeparamref name="T"/> with engine ops and
/// the dense kernels of <see cref="LinalgDispatch{T}"/>. Each returns the constant matrix that turns its solution
/// into a differentiable one.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// A base learner solves <c>theta* = argmin L(theta; Z, Y)</c> for the support embeddings <c>Z</c>. The meta-gradient
/// needs <c>d theta* / d Z</c>, which the paper obtains by the implicit function theorem on the KKT conditions
/// (section 3.2) rather than by differentiating the solver's iterations. Each solver here therefore returns its
/// solution together with the inverse of the KKT Jacobian at that solution; the caller then writes
/// <c>theta = theta* - Jinv * F(theta*, Z)</c> with the residual <c>F</c> built from engine ops. That expression
/// equals <c>theta*</c> (the residual is zero at the optimum) and differentiates to the implicit gradient, so the
/// tape carries one multiplication by a constant instead of a solver.
/// </para>
/// <para>The solves are numerical, not part of any tape: callers run them inside a no-grad scope.</para>
/// </remarks>
internal static class ConvexBaseLearners<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Ridge regression in the dual (the paper's R2D2-style head): <c>M = (K + lambda I)^-1 Y</c>, from which the
    /// classifier is <c>W = Z' M</c> and the query logits are <c>K_query M</c>.
    /// </summary>
    /// <param name="kernel">The support Gram matrix <c>K = Z Z'</c>, <c>[n, n]</c>.</param>
    /// <param name="oneHot">The one-hot support labels, <c>[n, classes]</c>.</param>
    /// <param name="lambda">The ridge regularization.</param>
    /// <returns>
    /// The dual coefficients and the inverse of <c>K + lambda I</c>, the KKT Jacobian of the residual
    /// <c>(K + lambda I) M - Y</c>. The system is solved by Cholesky (it is positive definite for lambda &gt; 0),
    /// falling back to LU when the factorization reports failure.
    /// </returns>
    internal static (Tensor<T> Solution, Tensor<T> JacobianInverse) SolveRidgeDual(Tensor<T> kernel, Tensor<T> oneHot, double lambda)
    {
        int n = kernel.Shape[0];
        var system = Engine.TensorAdd(kernel, Engine.TensorMultiplyScalar(Engine.TensorEye<T>(n), Ops.FromDouble(lambda)));
        var (factor, info) = LinalgDispatch<T>.CholeskyEx(system);
        var status = info.ToArray();
        if (status.Length == 0 || status[0] == 0)
        {
            return (LinalgDispatch<T>.CholeskySolve(oneHot, factor), LinalgDispatch<T>.CholeskyInverse(factor));
        }

        var inverse = LinalgDispatch<T>.Inv(system);
        return (Engine.TensorMatMul(inverse, oneHot), inverse);
    }

    /// <summary>
    /// The Crammer and Singer multi-class SVM dual (eq. 10), solved by coordinate descent over examples: each step
    /// re-optimises one example's block <c>alpha_n</c> exactly, the classical solver for this dual.
    /// </summary>
    /// <param name="kernel">The support Gram matrix, <c>[n, n]</c>.</param>
    /// <param name="oneHot">The one-hot support labels, <c>[n, classes]</c>.</param>
    /// <param name="cost">The regularization parameter C.</param>
    /// <param name="maxIterations">Sweeps over the support set.</param>
    /// <param name="tolerance">Stops when no block entry moves by more than this.</param>
    /// <returns>The dual coefficients <c>[n, classes]</c>.</returns>
    /// <remarks>
    /// The dual constrains each example's block by <c>alpha_nk &lt;= C [k = y_n]</c> and <c>sum_k alpha_nk = 0</c>, and its
    /// gradient is <c>K alpha - Y</c>. With the other blocks fixed the objective in one block is a separable
    /// quadratic over that set, minimised exactly by <see cref="ProjectBlock"/>.
    /// </remarks>
    internal static Tensor<T> SolveCrammerSingerDual(
        Tensor<T> kernel, Tensor<T> oneHot, double cost, int maxIterations, double tolerance)
    {
        int n = kernel.Shape[0], classes = oneHot.Shape[1];
        var state = new DualState(
            new Tensor<T>(new[] { n, classes }),
            Engine.TensorNegate(oneHot),
            Engine.TensorMultiplyScalar(oneHot, Ops.FromDouble(cost)));
        var diagonal = LinalgDispatch<T>.Diagonal(kernel);
        var eye = Engine.TensorEye<T>(n);
        T none = Ops.Zero;

        for (int sweep = 0; sweep < maxIterations; sweep++)
        {
            T largestMove = none;
            for (int i = 0; i < n; i++)
            {
                largestMove = Larger(largestMove, UpdateBlock(kernel, diagonal, eye, state, i));
            }

            if (Below(largestMove, tolerance)) break;
        }

        return state.Alpha;
    }

    /// <summary>The dual iterate, its gradient <c>K alpha - Y</c> and the upper bounds <c>C Y</c>.</summary>
    private sealed class DualState
    {
        internal DualState(Tensor<T> alpha, Tensor<T> gradient, Tensor<T> upper)
        {
            Alpha = alpha;
            Gradient = gradient;
            Upper = upper;
        }

        internal Tensor<T> Alpha { get; set; }

        internal Tensor<T> Gradient { get; set; }

        internal Tensor<T> Upper { get; }
    }

    /// <summary>
    /// Re-optimises example <paramref name="i"/>'s block and updates the gradient by <c>K[:, i] (x) delta</c>;
    /// returns the largest entry move (0 when <c>K_ii &lt; 1e-12</c>, which leaves the block alone).
    /// </summary>
    private static T UpdateBlock(Tensor<T> kernel, Tensor<T> diagonal, Tensor<T> eye, DualState state, int i)
    {
        int n = kernel.Shape[0], classes = state.Alpha.Shape[1];
        T d = Engine.TensorSum(Engine.TensorSlice(diagonal, new[] { i }, new[] { 1 }));
        if (Ops.LessThan(d, Ops.FromDouble(1e-12))) return Ops.Zero;

        var rowShape = new[] { classes };
        var alphaRow = Engine.Reshape(Engine.TensorSlice(state.Alpha, new[] { i, 0 }, new[] { 1, classes }), rowShape);
        var gradientRow = Engine.Reshape(Engine.TensorSlice(state.Gradient, new[] { i, 0 }, new[] { 1, classes }), rowShape);
        var upperRow = Engine.Reshape(Engine.TensorSlice(state.Upper, new[] { i, 0 }, new[] { 1, classes }), rowShape);

        // Minimise 0.5 * d_ii * ||a||^2 + (g_i - d_ii * alpha_i) . a over the block's constraint set.
        var linear = Engine.TensorSubtract(gradientRow, Engine.TensorMultiplyScalar(alphaRow, d));
        var updated = ProjectBlock(linear, d, upperRow);
        var delta = Engine.TensorSubtract(updated, alphaRow);

        // Row i of alpha becomes the new block exactly; every gradient row j moves by K[j, i] * delta.
        var shape = new[] { n, classes };
        var rowMask = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorSlice(eye, new[] { i, 0 }, new[] { 1, n }), new[] { n, 1 }), shape);
        state.Alpha = Engine.TensorWhere(
            rowMask, Engine.TensorBroadcastTo(Engine.Reshape(updated, new[] { 1, classes }), shape), state.Alpha);
        state.Gradient = Engine.TensorAdd(state.Gradient, Engine.TensorMatMul(
            Engine.TensorSlice(kernel, new[] { 0, i }, new[] { n, 1 }), Engine.Reshape(delta, new[] { 1, classes })));

        return Engine.TensorMaxValue(Engine.TensorAbs(delta));
    }

    /// <summary>
    /// The exact minimiser of <c>0.5 * d ||a||^2 + linear . a</c> over <c>a_k &lt;= u_k</c>, <c>sum_k a_k = 0</c>:
    /// <c>a_k = min(u_k, -(linear_k + nu) / d)</c> with the multiplier <c>nu</c> solving <c>S(nu) = 0</c>.
    /// </summary>
    /// <remarks>
    /// <c>S(nu) = sum_k min(u_k, -(l_k + nu) / d)</c> is continuous, piecewise linear and decreasing, with breakpoints
    /// <c>b_k = -l_k - d u_k</c> (entry k leaves its bound for <c>nu &gt;= b_k</c>). <c>S(min_k b_k) = sum_k u_k = C &gt;= 0</c>,
    /// so the root lies past the largest breakpoint <c>b*</c> with <c>S(b*) &gt;= 0</c>, on the linear piece where the free
    /// set is <c>F = {k : b_k &lt;= b*}</c>: <c>nu = (d sum_{k not in F} u_k - sum_{k in F} l_k) / |F|</c>. This is exact,
    /// where the double-precision solver bisected for the same root.
    /// </remarks>
    private static Tensor<T> ProjectBlock(Tensor<T> linear, T d, Tensor<T> upper)
    {
        int classes = linear.Shape[0];
        var square = new[] { classes, classes };

        var breakpoints = Engine.TensorSubtract(Engine.TensorNegate(linear), Engine.TensorMultiplyScalar(upper, d));

        // S at every breakpoint: row m holds min(u_k, -(l_k + b_m) / d) over k.
        var atBreakpoint = Engine.TensorBroadcastTo(Engine.Reshape(breakpoints, new[] { classes, 1 }), square);
        var linearRows = Engine.TensorBroadcastTo(Engine.Reshape(linear, new[] { 1, classes }), square);
        var upperRows = Engine.TensorBroadcastTo(Engine.Reshape(upper, new[] { 1, classes }), square);
        var terms = Engine.TensorMin(upperRows,
            Engine.TensorDivideScalar(Engine.TensorNegate(Engine.TensorAdd(linearRows, atBreakpoint)), d));
        var sums = Engine.ReduceSum(terms, new[] { 1 }, keepDims: false);

        // b* = the largest breakpoint with S(b) >= 0 (the smallest always qualifies)
        var qualifies = Engine.TensorAdd(Engine.TensorGreaterThan(sums, Ops.Zero), Engine.TensorEquals(sums, Ops.Zero));
        T lowest = Engine.TensorMinValue(breakpoints);
        var floor = Engine.TensorAddScalar(Engine.TensorZerosLike(breakpoints), lowest);
        T bStar = Engine.TensorMaxValue(Engine.TensorWhere(qualifies, breakpoints, floor));

        // Free set F = {k : b_k <= b*} and the multiplier on its linear piece
        var free = Engine.TensorSubtract(Engine.TensorOnesLike(breakpoints), Engine.TensorGreaterThan(breakpoints, bStar));
        T freeCount = Engine.TensorSum(free);
        T freeLinear = Engine.TensorSum(Engine.TensorMultiply(free, linear));
        T boundUpper = Engine.TensorSum(Engine.TensorMultiply(Engine.TensorSubtract(Engine.TensorOnesLike(free), free), upper));
        T nu = Ops.Divide(Ops.Subtract(Ops.Multiply(d, boundUpper), freeLinear), freeCount);

        return Engine.TensorMin(upper, Engine.TensorDivideScalar(Engine.TensorNegate(Engine.TensorAddScalar(linear, nu)), d));
    }

    /// <summary>
    /// The inverse KKT Jacobian of the Crammer and Singer dual at <paramref name="alpha"/>, over the free variables:
    /// the constraints active at the solution (a block entry at its upper bound) are held fixed, and each example's
    /// equality constraint is kept, which is the active-set form of the implicit function theorem.
    /// </summary>
    /// <returns>
    /// <c>[n * classes, n * classes]</c>: the inverse KKT Jacobian over the free variables, so that the caller's
    /// <c>alpha* - Jinv F</c> carries the implicit derivative. Rows and columns of variables pinned by an active
    /// bound are zero: a pinned variable does not move.
    /// </returns>
    /// <remarks>
    /// The system <c>[Q_ff A_f'; A_f 0]</c> (Q the block-diagonal kernel <c>K (x) I</c>, A the per-example sums) is
    /// assembled at full size with every pinned variable decoupled into an identity row and column and a zero
    /// right-hand side, which leaves the free block's solution unchanged and pins the others at zero. An example
    /// with every entry pinned has an unconstrained multiplier, anchored by a unit diagonal.
    /// </remarks>
    internal static Tensor<T> CrammerSingerJacobianInverse(
        Tensor<T> kernel, Tensor<T> alpha, Tensor<T> oneHot, double cost, double activeTolerance)
    {
        int n = kernel.Shape[0], classes = alpha.Shape[1], size = n * classes;
        var upper = Engine.TensorMultiplyScalar(oneHot, Ops.FromDouble(cost));
        var freeGrid = Engine.TensorGreaterThan(Engine.TensorSubtract(upper, alpha), Ops.FromDouble(activeTolerance));
        var free = Engine.Reshape(freeGrid, new[] { size });

        var full = new[] { size, size };
        var freeRows = Engine.TensorBroadcastTo(Engine.Reshape(free, new[] { size, 1 }), full);
        var freeColumns = Engine.TensorBroadcastTo(Engine.Reshape(free, new[] { 1, size }), full);
        var pinned = Engine.TensorSubtract(Engine.TensorOnesLike(free), free);
        var eyeSize = Engine.TensorEye<T>(size);

        // Q = K (x) I_classes, index (example * classes + class)
        var blockShape = new[] { n, classes, n, classes };
        var q = Engine.Reshape(Engine.TensorMultiply(
            Engine.TensorBroadcastTo(Engine.Reshape(kernel, new[] { n, 1, n, 1 }), blockShape),
            Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorEye<T>(classes), new[] { 1, classes, 1, classes }), blockShape)), full);
        var qFree = Engine.TensorAdd(
            Engine.TensorMultiply(Engine.TensorMultiply(q, freeRows), freeColumns),
            Engine.TensorMultiply(eyeSize, Engine.TensorBroadcastTo(Engine.Reshape(pinned, new[] { size, 1 }), full)));

        // A[i, (j, k)] = [i = j] over the free variables
        var sums = Engine.TensorMultiply(
            Engine.Reshape(Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorEye<T>(n), new[] { n, n, 1 }), new[] { n, n, classes }), new[] { n, size }),
            Engine.TensorBroadcastTo(Engine.Reshape(free, new[] { 1, size }), new[] { n, size }));

        // Anchor the multiplier of an example with no free entry
        var freePerExample = Engine.ReduceSum(freeGrid, new[] { 1 }, keepDims: false);
        var allPinned = Engine.TensorSubtract(Engine.TensorOnesLike(freePerExample), Engine.TensorGreaterThan(freePerExample, Ops.Zero));
        var anchor = Engine.TensorMultiply(Engine.TensorEye<T>(n), Engine.TensorBroadcastTo(Engine.Reshape(allPinned, new[] { n, 1 }), new[] { n, n }));

        var system = Engine.TensorConcatenate(new[]
        {
            Engine.TensorConcatenate(new[] { qFree, Engine.TensorTranspose(sums) }, 1),
            Engine.TensorConcatenate(new[] { sums, anchor }, 1),
        }, 0);
        var rightHand = Engine.TensorConcatenate(new[]
        {
            Engine.TensorMultiply(eyeSize, freeRows),
            new Tensor<T>(new[] { n, size }),
        }, 0);

        var solved = LinalgDispatch<T>.Solve(system, rightHand);
        var inverse = Engine.TensorSlice(solved, new[] { 0, 0 }, new[] { size, size });
        return Engine.TensorMultiply(Engine.TensorMultiply(inverse, freeRows), freeColumns);
    }

    /// <summary>
    /// Multi-class logistic regression with L2 regularization, by Newton's method on the primal weights.
    /// </summary>
    /// <param name="features">The support embeddings, <c>[n, d]</c>.</param>
    /// <param name="oneHot">The one-hot support labels, <c>[n, classes]</c>.</param>
    /// <param name="lambda">The L2 regularization.</param>
    /// <param name="maxIterations">Newton steps.</param>
    /// <param name="tolerance">Stops when the largest weight change falls below this.</param>
    /// <returns>The weights <c>[classes, d]</c> and the inverse Hessian of the objective at them.</returns>
    internal static (Tensor<T> Weights, Tensor<T> HessianInverse) SolveLogistic(
        Tensor<T> features, Tensor<T> oneHot, double lambda, int maxIterations, double tolerance)
    {
        int n = features.Shape[0], d = features.Shape[1], classes = oneHot.Shape[1];
        int size = classes * d;
        var weights = new Tensor<T>(new[] { classes, d });
        T lambdaT = Ops.FromDouble(lambda);
        T inverseCount = Ops.FromDouble(1.0 / n);
        var gradientShape = new[] { size, 1 };
        var weightShape = new[] { classes, d };

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var probabilities = Probabilities(features, weights);

            // (P - Y)' Z / n + lambda W, flattened class-major like the Hessian
            var gradient = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(
                    Engine.TensorMatMul(Engine.TensorTranspose(Engine.TensorSubtract(probabilities, oneHot)), features),
                    inverseCount),
                Engine.TensorMultiplyScalar(weights, lambdaT));

            var hessian = LogisticHessian(features, probabilities, lambda);
            var step = LinalgDispatch<T>.Solve(hessian, Engine.Reshape(gradient, gradientShape));
            weights = Engine.TensorSubtract(weights, Engine.Reshape(step, weightShape));

            if (Below(Engine.TensorMaxValue(Engine.TensorAbs(step)), tolerance)) break;
        }

        var finalHessian = LogisticHessian(features, Probabilities(features, weights), lambda);
        return (weights, LinalgDispatch<T>.Inv(finalHessian));
    }

    /// <summary>Row-wise softmax of <c>Z W'</c>.</summary>
    private static Tensor<T> Probabilities(Tensor<T> features, Tensor<T> weights)
        => Engine.Softmax(Engine.TensorMatMul(features, Engine.TensorTranspose(weights)), 1);

    /// <summary>
    /// <c>sum_i (diag(p_i) - p_i p_i') (x) (z_i z_i') / n + lambda I</c>, indexed (class * d + feature).
    /// </summary>
    private static Tensor<T> LogisticHessian(Tensor<T> features, Tensor<T> probabilities, double lambda)
    {
        int n = features.Shape[0], d = features.Shape[1], classes = probabilities.Shape[1];
        int size = classes * d;

        // S_i[c, e] = p_ic ([c = e] - p_ie)
        var classSquare = new[] { n, classes, classes };
        var pColumn = Engine.TensorBroadcastTo(Engine.Reshape(probabilities, new[] { n, classes, 1 }), classSquare);
        var pRow = Engine.TensorBroadcastTo(Engine.Reshape(probabilities, new[] { n, 1, classes }), classSquare);
        var identity = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorEye<T>(classes), new[] { 1, classes, classes }), classSquare);
        var s = Engine.TensorMultiply(pColumn, Engine.TensorSubtract(identity, pRow));

        // (z_i z_i')[j, l]
        var featureSquare = new[] { n, d, d };
        var outer = Engine.TensorMultiply(
            Engine.TensorBroadcastTo(Engine.Reshape(features, new[] { n, d, 1 }), featureSquare),
            Engine.TensorBroadcastTo(Engine.Reshape(features, new[] { n, 1, d }), featureSquare));

        // Kronecker blocks [n, c, j, e, l], summed over the examples
        var blockShape = new[] { n, classes, d, classes, d };
        var blocks = Engine.TensorMultiply(
            Engine.TensorBroadcastTo(Engine.Reshape(s, new[] { n, classes, 1, classes, 1 }), blockShape),
            Engine.TensorBroadcastTo(Engine.Reshape(outer, new[] { n, 1, d, 1, d }), blockShape));
        var summed = Engine.Reshape(Engine.ReduceSum(blocks, new[] { 0 }, keepDims: false), new[] { size, size });

        return Engine.TensorAdd(
            Engine.TensorMultiplyScalar(summed, Ops.FromDouble(1.0 / n)),
            Engine.TensorMultiplyScalar(Engine.TensorEye<T>(size), Ops.FromDouble(lambda)));
    }

    /// <summary>The larger of two values; a NaN on the right is ignored, as Math.Max over a running 0 would not be.</summary>
    private static T Larger(T current, T candidate) => Ops.GreaterThan(candidate, current) ? candidate : current;

    /// <summary>True when <paramref name="value"/> is below <paramref name="tolerance"/> (NaN is not).</summary>
    private static bool Below(T value, double tolerance) => Ops.LessThan(value, Ops.FromDouble(tolerance));
}
