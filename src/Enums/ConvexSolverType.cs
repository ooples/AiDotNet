namespace AiDotNet.Enums;

/// <summary>
/// Types of convex solvers available for MetaOptNet.
/// </summary>
/// <remarks>
/// <para>
/// MetaOptNet uses convex optimization in the inner loop instead of gradient descent.
/// These solver types provide different trade-offs between speed and classification power.
/// </para>
/// <para><b>For Beginners:</b>
/// Instead of iteratively updating weights, convex solvers find the optimal solution directly.
/// Think of it like finding the lowest point in a bowl - convex problems have a single
/// lowest point, so we can find it mathematically without searching around.
/// </para>
/// </remarks>
public enum ConvexSolverType
{
    /// <summary>
    /// Ridge regression (L2-regularized least squares).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The simplest and fastest option. Solves: minimize ||Xw - y||² + λ||w||²
    /// Has a closed-form solution: w = (X^T X + λI)^(-1) X^T y
    /// </para>
    /// <para><b>For Beginners:</b> Ridge regression finds a simple line (or hyperplane)
    /// that best separates the classes while staying simple (not too wiggly).
    /// It's fast because we can compute the answer with a single matrix equation.
    /// </para>
    /// </remarks>
    RidgeRegression,

    /// <summary>
    /// Support Vector Machine with quadratic programming.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More powerful but slower. Finds the maximum-margin hyperplane.
    /// Better for tasks where classes have clear boundaries.
    /// </para>
    /// <para><b>For Beginners:</b> SVM finds a boundary that maximizes the gap
    /// between classes. Like drawing a road between two countries - SVM makes
    /// the road as wide as possible for safety.
    /// </para>
    /// </remarks>
    SVM,

    /// <summary>
    /// Logistic regression solved via Newton's method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Good for probabilistic outputs. Provides calibrated confidence scores.
    /// </para>
    /// <para><b>For Beginners:</b> Logistic regression gives you probabilities
    /// ("80% confident this is a cat") rather than just classifications.
    /// Useful when you need to know how certain the model is.
    /// </para>
    /// </remarks>
    LogisticRegression
}
