using AiDotNet.Tensors.Groups;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Extension interface for advanced algebraic operations used in specialized neural networks.
/// </summary>
/// <remarks>
/// <para>
/// IAdvancedAlgebraEngine provides batch operations for octonions, multivectors (Clifford algebras),
/// and Lie groups. These are used in octonion neural networks, geometric algebra transformers,
/// and equivariant neural networks.
/// </para>
/// <para><b>For Beginners:</b> This interface handles mathematical operations on complex algebraic
/// structures that go beyond regular numbers:
///
/// - Octonions: 8-dimensional numbers (like super-quaternions) for advanced image processing
/// - Multivectors: Objects from geometric algebra for 3D geometry and physics
/// - Lie Groups: Rotation and transformation groups for equivariant learning
/// </para>
/// </remarks>
public interface IAdvancedAlgebraEngine
{
    #region Octonion Batch Operations

    /// <summary>
    /// Batch octonion-octonion multiplication (non-associative).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Array of left operand octonions.</param>
    /// <param name="right">Array of right operand octonions.</param>
    /// <returns>Array of octonion products.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Octonion multiplication is more complex than regular
    /// multiplication - the order matters AND grouping matters (non-associative).
    /// </para>
    /// <para>
    /// Warning: Octonion multiplication is non-associative: (a*b)*c != a*(b*c).
    /// This implementation uses left-to-right evaluation order.
    /// </para>
    /// </remarks>
    Octonion<T>[] OctonionMultiplyBatch<T>(Octonion<T>[] left, Octonion<T>[] right);

    /// <summary>
    /// Batch octonion addition.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Array of left operand octonions.</param>
    /// <param name="right">Array of right operand octonions.</param>
    /// <returns>Array of octonion sums.</returns>
    Octonion<T>[] OctonionAddBatch<T>(Octonion<T>[] left, Octonion<T>[] right);

    /// <summary>
    /// Batch octonion conjugation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="octonions">Array of octonions.</param>
    /// <returns>Array of conjugate octonions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The conjugate of an octonion flips the sign of all
    /// imaginary parts (e1 through e7), keeping the scalar part unchanged.
    /// It's similar to the conjugate of complex numbers.
    /// </para>
    /// </remarks>
    Octonion<T>[] OctonionConjugateBatch<T>(Octonion<T>[] octonions);

    /// <summary>
    /// Computes batch octonion norms.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="octonions">Array of octonions.</param>
    /// <returns>Array of norm values.</returns>
    T[] OctonionNormBatch<T>(Octonion<T>[] octonions);

    /// <summary>
    /// Octonion-matrix multiplication for neural network layers.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">Input octonion tensor (batch x features).</param>
    /// <param name="weight">Weight octonion matrix (output x input features).</param>
    /// <returns>Output octonion tensor.</returns>
    /// <remarks>
    /// <para>
    /// This is the core operation for OctonionLinear layers. Each output element is
    /// computed as the sum of input elements multiplied by corresponding weight octonions.
    /// </para>
    /// </remarks>
    Octonion<T>[,] OctonionMatMul<T>(Octonion<T>[,] input, Octonion<T>[,] weight);

    #endregion

    #region Multivector/Clifford Batch Operations

    /// <summary>
    /// Batch geometric product of multivectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Array of left operand multivectors.</param>
    /// <param name="right">Array of right operand multivectors.</param>
    /// <returns>Array of geometric products.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The geometric product is the fundamental multiplication
    /// in geometric algebra. It combines the inner product (like dot product) and outer
    /// product (like cross product) into one operation.
    /// </para>
    /// <para>
    /// AB = A·B + A∧B (inner product + outer product)
    /// </para>
    /// </remarks>
    Multivector<T>[] GeometricProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>
    /// Batch wedge (outer) product of multivectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Array of left operand multivectors.</param>
    /// <param name="right">Array of right operand multivectors.</param>
    /// <returns>Array of wedge products.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The wedge product creates higher-dimensional objects:
    /// - Two vectors wedged together make a bivector (like an oriented area)
    /// - A vector wedged with a bivector makes a trivector (like an oriented volume)
    /// </para>
    /// </remarks>
    Multivector<T>[] WedgeProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>
    /// Batch inner product of multivectors.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Array of left operand multivectors.</param>
    /// <param name="right">Array of right operand multivectors.</param>
    /// <returns>Array of inner products.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The inner product (contraction) reduces grades:
    /// - Vector · Vector = Scalar (like dot product)
    /// - Bivector · Vector = Vector
    /// </para>
    /// </remarks>
    Multivector<T>[] InnerProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>
    /// Batch multivector addition.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="left">Array of left operand multivectors.</param>
    /// <param name="right">Array of right operand multivectors.</param>
    /// <returns>Array of multivector sums.</returns>
    Multivector<T>[] MultivectorAddBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>
    /// Batch multivector reverse operation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="multivectors">Array of multivectors.</param>
    /// <returns>Array of reversed multivectors.</returns>
    /// <remarks>
    /// <para>
    /// The reverse operation flips the order of basis vectors in each blade.
    /// For a blade of grade k, this multiplies by (-1)^(k(k-1)/2).
    /// </para>
    /// </remarks>
    Multivector<T>[] MultivectorReverseBatch<T>(Multivector<T>[] multivectors);

    /// <summary>
    /// Batch grade projection - extracts components of a specific grade.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="multivectors">Array of multivectors.</param>
    /// <param name="grade">The grade to project onto (0=scalar, 1=vector, 2=bivector, etc.).</param>
    /// <returns>Array of projected multivectors.</returns>
    Multivector<T>[] GradeProjectBatch<T>(Multivector<T>[] multivectors, int grade);

    #endregion

    #region Lie Group Batch Operations

    /// <summary>
    /// Batch exponential map for SO(3) (rotation group).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SO(3) group implementation.</param>
    /// <param name="tangentVectors">Array of axis-angle vectors (3D) in the Lie algebra.</param>
    /// <returns>Array of SO(3) rotation elements.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The exponential map converts a rotation axis and angle
    /// (represented as a 3D vector whose direction is the axis and magnitude is the angle)
    /// into an actual 3x3 rotation matrix. This uses the Rodrigues formula.
    /// </para>
    /// </remarks>
    So3<T>[] So3ExpBatch<T>(So3Group<T> group, Vector<T>[] tangentVectors);

    /// <summary>
    /// Batch logarithm map for SO(3).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SO(3) group implementation.</param>
    /// <param name="rotations">Array of SO(3) rotation elements.</param>
    /// <returns>Array of axis-angle vectors.</returns>
    Vector<T>[] So3LogBatch<T>(So3Group<T> group, So3<T>[] rotations);

    /// <summary>
    /// Batch group composition for SO(3).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SO(3) group implementation.</param>
    /// <param name="left">Array of left SO(3) rotations.</param>
    /// <param name="right">Array of right SO(3) rotations.</param>
    /// <returns>Array of composed rotations (left * right).</returns>
    So3<T>[] So3ComposeBatch<T>(So3Group<T> group, So3<T>[] left, So3<T>[] right);

    /// <summary>
    /// Batch exponential map for SE(3) (rigid transformation group).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SE(3) group implementation.</param>
    /// <param name="tangentVectors">Array of 6D twist vectors in the Lie algebra.</param>
    /// <returns>Array of SE(3) transformation elements (rotation + translation).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> SE(3) represents rigid body transformations (rotation + translation)
    /// in 3D space. The 6D tangent vector contains both the rotation axis-angle (first 3) and
    /// translation direction (last 3).
    /// </para>
    /// </remarks>
    Se3<T>[] Se3ExpBatch<T>(Se3Group<T> group, Vector<T>[] tangentVectors);

    /// <summary>
    /// Batch logarithm map for SE(3).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SE(3) group implementation.</param>
    /// <param name="transforms">Array of SE(3) transformations.</param>
    /// <returns>Array of 6D twist vectors.</returns>
    Vector<T>[] Se3LogBatch<T>(Se3Group<T> group, Se3<T>[] transforms);

    /// <summary>
    /// Batch group composition for SE(3).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SE(3) group implementation.</param>
    /// <param name="left">Array of left SE(3) transformations.</param>
    /// <param name="right">Array of right SE(3) transformations.</param>
    /// <returns>Array of composed transformations.</returns>
    Se3<T>[] Se3ComposeBatch<T>(Se3Group<T> group, Se3<T>[] left, Se3<T>[] right);

    /// <summary>
    /// Batch adjoint representation for SO(3).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="group">The SO(3) group implementation.</param>
    /// <param name="rotations">Array of SO(3) rotation elements.</param>
    /// <returns>Array of 3x3 adjoint matrices.</returns>
    /// <remarks>
    /// <para>
    /// The adjoint representation maps Lie algebra elements to Lie algebra elements.
    /// For SO(3), Ad_R(v) = R * v (rotation of the tangent vector).
    /// </para>
    /// </remarks>
    Matrix<T>[] So3AdjointBatch<T>(So3Group<T> group, So3<T>[] rotations);

    #endregion
}
