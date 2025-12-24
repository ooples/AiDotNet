using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Extension interface for hyperbolic manifold operations used in hyperbolic neural networks.
/// </summary>
/// <remarks>
/// <para>
/// IHyperbolicManifoldEngine provides operations on hyperbolic spaces, which are non-Euclidean
/// spaces with constant negative curvature. These are used in hyperbolic neural networks
/// for hierarchical data like trees, graphs, and taxonomies.
/// </para>
/// <para><b>For Beginners:</b> Hyperbolic space is like the surface of a saddle - it curves away
/// from itself in all directions. This geometry naturally represents tree-like (hierarchical) data:
///
/// - The center represents the root of a hierarchy
/// - Moving toward the edge represents going down the hierarchy
/// - There's "more room" near the edge, perfect for many leaves
///
/// Two common models:
/// - Poincare Ball: Points inside a unit sphere
/// - Hyperboloid: Points on a hyperbola in Minkowski space
/// </para>
/// </remarks>
public interface IHyperbolicManifoldEngine
{
    #region Poincare Ball Model Operations

    /// <summary>
    /// Exponential map: Projects a tangent vector at a point onto the Poincare ball.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="basePoint">The base point on the manifold.</param>
    /// <param name="tangentVector">The tangent vector at the base point.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The point on the manifold reached by following the tangent vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The exponential map "shoots" from a point in a direction.
    /// If you're standing at a point and want to walk in a direction, the exp map tells you
    /// where you end up after following the curved surface.
    /// </para>
    /// <para>
    /// Formula: exp_x(v) = x ⊕ tanh(√c ||v|| / (1 - c||x||²)) * (v / ||v||)
    /// where ⊕ is Möbius addition and c = -curvature.
    /// </para>
    /// </remarks>
    Vector<T> PoincareExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature);

    /// <summary>
    /// Logarithmic map: Computes the tangent vector from one point to another.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="basePoint">The base point on the manifold.</param>
    /// <param name="targetPoint">The target point on the manifold.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The tangent vector at basePoint pointing toward targetPoint.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The logarithmic map is the inverse of exp.
    /// Given two points, it tells you the direction and distance (in tangent space)
    /// from one to the other.
    /// </para>
    /// </remarks>
    Vector<T> PoincareLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature);

    /// <summary>
    /// Möbius addition: The hyperbolic analog of vector addition.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="x">First point in the Poincare ball.</param>
    /// <param name="y">Second point in the Poincare ball.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The Möbius sum x ⊕ y.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regular addition doesn't work in hyperbolic space
    /// because it might take you outside the ball. Möbius addition is a special
    /// operation that always stays inside the ball.
    /// </para>
    /// <para>
    /// Formula: x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
    /// </para>
    /// </remarks>
    Vector<T> MobiusAdd<T>(Vector<T> x, Vector<T> y, T curvature);

    /// <summary>
    /// Computes the geodesic (shortest path) distance between two points.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="x">First point in the Poincare ball.</param>
    /// <param name="y">Second point in the Poincare ball.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The geodesic distance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The geodesic distance is the length of the shortest
    /// path along the curved surface. Points near the edge of the ball can be very
    /// far apart in geodesic distance even if they look close in Euclidean terms.
    /// </para>
    /// <para>
    /// Formula: d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
    /// </para>
    /// </remarks>
    T PoincareDistance<T>(Vector<T> x, Vector<T> y, T curvature);

    /// <summary>
    /// Parallel transport: Moves a tangent vector from one point to another along the geodesic.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="x">Source point.</param>
    /// <param name="y">Destination point.</param>
    /// <param name="v">Tangent vector at x.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <returns>The transported tangent vector at y.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When you move along a curved surface, directions change.
    /// Parallel transport tells you how to "carry" a direction vector from one point to another
    /// while keeping it as parallel as possible to itself.
    /// </para>
    /// </remarks>
    Vector<T> PoincareParallelTransport<T>(Vector<T> x, Vector<T> y, Vector<T> v, T curvature);

    /// <summary>
    /// Projects a point onto the Poincare ball (clamps to valid region).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="point">The point to project.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <param name="epsilon">Small value to stay away from the boundary (default 1e-5).</param>
    /// <returns>The projected point inside the ball.</returns>
    /// <remarks>
    /// <para>
    /// Numerical operations can push points outside the valid region (||x|| &lt; 1/√c).
    /// This projects them back inside for numerical stability.
    /// </para>
    /// </remarks>
    Vector<T> PoincareProject<T>(Vector<T> point, T curvature, T epsilon);

    #endregion

    #region Hyperboloid Model Operations

    /// <summary>
    /// Exponential map on the hyperboloid model.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="basePoint">The base point on the hyperboloid.</param>
    /// <param name="tangentVector">The tangent vector at the base point.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The point on the hyperboloid reached by following the tangent vector.</returns>
    /// <remarks>
    /// <para>
    /// Formula: exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * (v / ||v||_L)
    /// where ||·||_L is the Lorentzian norm.
    /// </para>
    /// </remarks>
    Vector<T> HyperboloidExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature);

    /// <summary>
    /// Logarithmic map on the hyperboloid model.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="basePoint">The base point on the hyperboloid.</param>
    /// <param name="targetPoint">The target point on the hyperboloid.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The tangent vector at basePoint pointing toward targetPoint.</returns>
    Vector<T> HyperboloidLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature);

    /// <summary>
    /// Computes the geodesic distance on the hyperboloid using the Minkowski inner product.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="x">First point on the hyperboloid.</param>
    /// <param name="y">Second point on the hyperboloid.</param>
    /// <param name="curvature">The curvature parameter (typically -1).</param>
    /// <returns>The geodesic distance.</returns>
    /// <remarks>
    /// <para>
    /// Formula: d(x,y) = (1/√c) * arcosh(-c * ⟨x,y⟩_L)
    /// where ⟨·,·⟩_L is the Minkowski inner product.
    /// </para>
    /// </remarks>
    T HyperboloidDistance<T>(Vector<T> x, Vector<T> y, T curvature);

    /// <summary>
    /// Projects a point onto the hyperboloid (ensures it satisfies the constraint).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="point">The point to project.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <returns>The projected point on the hyperboloid.</returns>
    /// <remarks>
    /// <para>
    /// The hyperboloid constraint is: -x₀² + x₁² + ... + xₙ² = -1/c (for curvature c &lt; 0).
    /// This adjusts x₀ to satisfy the constraint while keeping other coordinates fixed.
    /// </para>
    /// </remarks>
    Vector<T> HyperboloidProject<T>(Vector<T> point, T curvature);

    #endregion

    #region Model Conversions

    /// <summary>
    /// Converts a point from Poincare ball model to hyperboloid model.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="poincarePoint">Point in the Poincare ball.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <returns>The equivalent point on the hyperboloid.</returns>
    Vector<T> PoincareToHyperboloid<T>(Vector<T> poincarePoint, T curvature);

    /// <summary>
    /// Converts a point from hyperboloid model to Poincare ball model.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="hyperboloidPoint">Point on the hyperboloid.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <returns>The equivalent point in the Poincare ball.</returns>
    Vector<T> HyperboloidToPoincare<T>(Vector<T> hyperboloidPoint, T curvature);

    #endregion

    #region Batch Operations

    /// <summary>
    /// Batched exponential map on the Poincare ball.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="basePoints">Matrix where each row is a base point.</param>
    /// <param name="tangentVectors">Matrix where each row is a tangent vector.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <returns>Matrix where each row is the result of exp map.</returns>
    Matrix<T> PoincareExpMapBatch<T>(Matrix<T> basePoints, Matrix<T> tangentVectors, T curvature);

    /// <summary>
    /// Batched geodesic distance computation on the Poincare ball.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="x">Matrix where each row is a point.</param>
    /// <param name="y">Matrix where each row is a point.</param>
    /// <param name="curvature">The curvature parameter.</param>
    /// <returns>Vector of pairwise distances.</returns>
    Vector<T> PoincareDistanceBatch<T>(Matrix<T> x, Matrix<T> y, T curvature);

    #endregion
}
