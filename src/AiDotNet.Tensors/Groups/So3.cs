using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Represents an element of SO(3) as a 3x3 rotation matrix.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public readonly struct So3<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public Matrix<T> Matrix { get; }

    /// <summary>
    /// Creates an SO(3) element from a 3x3 rotation matrix.
    /// </summary>
    /// <param name="matrix">The rotation matrix.</param>
    /// <param name="validate">If true, validates orthogonality and determinant. Default is true.</param>
    public So3(Matrix<T> matrix, bool validate = true)
    {
        if (matrix is null)
            throw new ArgumentNullException(nameof(matrix));
        if (matrix.Rows != 3 || matrix.Columns != 3)
            throw new ArgumentException("SO(3) matrices must be 3x3.", nameof(matrix));

        if (validate)
        {
            ValidateRotationMatrix(matrix);
        }

        Matrix = matrix;
    }

    /// <summary>
    /// Validates that the matrix is a proper rotation matrix (orthogonal with det = +1).
    /// </summary>
    private static void ValidateRotationMatrix(Matrix<T> matrix)
    {
        const double tolerance = 1e-6;

        // Check orthogonality: R^T * R should be close to identity
        var transpose = matrix.Transpose();
        var product = transpose.Multiply(matrix);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = i == j ? 1.0 : 0.0;
                double actual = NumOps.ToDouble(product[i, j]);
                if (Math.Abs(actual - expected) > tolerance)
                {
                    throw new ArgumentException(
                        $"Matrix is not orthogonal. R^T * R[{i},{j}] = {actual}, expected {expected}.",
                        nameof(matrix));
                }
            }
        }

        // Check determinant is +1 (not -1) using 3x3 formula
        // det = a(ek - fh) - b(dk - fg) + c(dh - eg)
        double a = NumOps.ToDouble(matrix[0, 0]);
        double b = NumOps.ToDouble(matrix[0, 1]);
        double c = NumOps.ToDouble(matrix[0, 2]);
        double d = NumOps.ToDouble(matrix[1, 0]);
        double e = NumOps.ToDouble(matrix[1, 1]);
        double f = NumOps.ToDouble(matrix[1, 2]);
        double g = NumOps.ToDouble(matrix[2, 0]);
        double h = NumOps.ToDouble(matrix[2, 1]);
        double k = NumOps.ToDouble(matrix[2, 2]);
        double det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);

        if (Math.Abs(det - 1.0) > tolerance)
        {
            throw new ArgumentException(
                $"Matrix determinant is {det}, expected +1 for SO(3).",
                nameof(matrix));
        }
    }

    public static So3<T> Identity => new So3<T>(Matrix<T>.CreateIdentityMatrix(3), validate: false);
}
