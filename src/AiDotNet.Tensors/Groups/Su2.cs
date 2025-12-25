using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Represents an element of SU(2) as a unit quaternion.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public readonly struct Su2<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public T W { get; }
    public T X { get; }
    public T Y { get; }
    public T Z { get; }

    /// <summary>
    /// Creates an SU(2) element from quaternion components.
    /// </summary>
    /// <param name="w">The scalar component.</param>
    /// <param name="x">The i component.</param>
    /// <param name="y">The j component.</param>
    /// <param name="z">The k component.</param>
    /// <param name="validate">If true, validates that the quaternion is unit length. Default is true.</param>
    public Su2(T w, T x, T y, T z, bool validate = true)
    {
        if (validate)
        {
            ValidateUnitQuaternion(w, x, y, z);
        }

        W = w;
        X = x;
        Y = y;
        Z = z;
    }

    /// <summary>
    /// Validates that the quaternion has unit length (w² + x² + y² + z² = 1).
    /// </summary>
    private static void ValidateUnitQuaternion(T w, T x, T y, T z)
    {
        const double tolerance = 1e-6;

        double wD = NumOps.ToDouble(w);
        double xD = NumOps.ToDouble(x);
        double yD = NumOps.ToDouble(y);
        double zD = NumOps.ToDouble(z);

        double normSquared = wD * wD + xD * xD + yD * yD + zD * zD;

        if (Math.Abs(normSquared - 1.0) > tolerance)
        {
            throw new ArgumentException(
                $"Quaternion must be unit length for SU(2). Got norm² = {normSquared}, expected 1.0.");
        }
    }

    /// <summary>
    /// Gets the identity element (1, 0, 0, 0).
    /// </summary>
    public static Su2<T> Identity => new Su2<T>(NumOps.One, NumOps.Zero, NumOps.Zero, NumOps.Zero, validate: false);
}
