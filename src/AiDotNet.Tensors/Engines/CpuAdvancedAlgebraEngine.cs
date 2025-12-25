using System;
using AiDotNet.Tensors.Groups;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// CPU implementation of advanced algebraic operations.
/// </summary>
/// <remarks>
/// <para>
/// CpuAdvancedAlgebraEngine provides batch operations for octonions, multivectors,
/// and Lie groups on the CPU. It delegates to the existing algebraic type methods
/// while providing efficient batch processing.
/// </para>
/// <para><b>For Beginners:</b> This class handles the actual computation for advanced
/// mathematical operations like octonion multiplication and Lie group exponentials.
/// It runs on your CPU - a GPU version would run the same operations faster on graphics hardware.
/// </para>
/// </remarks>
public sealed class CpuAdvancedAlgebraEngine : IAdvancedAlgebraEngine
{
    /// <summary>
    /// Singleton instance for convenience.
    /// </summary>
    public static CpuAdvancedAlgebraEngine Instance { get; } = new CpuAdvancedAlgebraEngine();

    #region Octonion Batch Operations

    /// <inheritdoc/>
    public Octonion<T>[] OctonionMultiplyBatch<T>(Octonion<T>[] left, Octonion<T>[] right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Octonion<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i] * right[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Octonion<T>[] OctonionAddBatch<T>(Octonion<T>[] left, Octonion<T>[] right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Octonion<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i] + right[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Octonion<T>[] OctonionConjugateBatch<T>(Octonion<T>[] octonions)
    {
        if (octonions is null) throw new ArgumentNullException(nameof(octonions));

        var result = new Octonion<T>[octonions.Length];
        for (int i = 0; i < octonions.Length; i++)
        {
            result[i] = octonions[i].Conjugate();
        }

        return result;
    }

    /// <inheritdoc/>
    public T[] OctonionNormBatch<T>(Octonion<T>[] octonions)
    {
        if (octonions is null) throw new ArgumentNullException(nameof(octonions));

        var result = new T[octonions.Length];
        for (int i = 0; i < octonions.Length; i++)
        {
            result[i] = octonions[i].Magnitude;
        }

        return result;
    }

    /// <inheritdoc/>
    public Octonion<T>[,] OctonionMatMul<T>(Octonion<T>[,] input, Octonion<T>[,] weight)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weight is null) throw new ArgumentNullException(nameof(weight));

        int batchSize = input.GetLength(0);
        int inputFeatures = input.GetLength(1);
        int outputFeatures = weight.GetLength(0);
        int weightInputFeatures = weight.GetLength(1);

        if (inputFeatures != weightInputFeatures)
        {
            throw new ArgumentException(
                $"Input features ({inputFeatures}) must match weight input dimension ({weightInputFeatures}).");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Octonion<T>[batchSize, outputFeatures];

        // Initialize with zero octonions
        var zero = new Octonion<T>(ops.Zero, ops.Zero, ops.Zero, ops.Zero,
                                   ops.Zero, ops.Zero, ops.Zero, ops.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                Octonion<T> sum = zero;
                for (int i = 0; i < inputFeatures; i++)
                {
                    // Note: Octonion multiplication is non-associative, so order matters
                    sum = sum + (weight[o, i] * input[b, i]);
                }
                result[b, o] = sum;
            }
        }

        return result;
    }

    #endregion

    #region Multivector/Clifford Batch Operations

    /// <inheritdoc/>
    public Multivector<T>[] GeometricProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Multivector<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i].GeometricProduct(right[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Multivector<T>[] WedgeProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Multivector<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i].OuterProduct(right[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Multivector<T>[] InnerProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Multivector<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i].InnerProduct(right[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Multivector<T>[] MultivectorAddBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
    {
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Multivector<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = left[i] + right[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Multivector<T>[] MultivectorReverseBatch<T>(Multivector<T>[] multivectors)
    {
        if (multivectors is null) throw new ArgumentNullException(nameof(multivectors));

        var result = new Multivector<T>[multivectors.Length];
        for (int i = 0; i < multivectors.Length; i++)
        {
            result[i] = multivectors[i].Reverse();
        }

        return result;
    }

    /// <inheritdoc/>
    public Multivector<T>[] GradeProjectBatch<T>(Multivector<T>[] multivectors, int grade)
    {
        if (multivectors is null) throw new ArgumentNullException(nameof(multivectors));
        if (grade < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(grade), "Grade must be non-negative.");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Multivector<T>[multivectors.Length];

        for (int i = 0; i < multivectors.Length; i++)
        {
            var mv = multivectors[i];
            var algebra = mv.Algebra;
            var coeffs = new T[mv.BasisCount];

            // Only keep coefficients of the specified grade
            for (int blade = 0; blade < mv.BasisCount; blade++)
            {
                if (algebra.GetGrade(blade) == grade)
                {
                    coeffs[blade] = mv[blade];
                }
                else
                {
                    coeffs[blade] = ops.Zero;
                }
            }

            result[i] = new Multivector<T>(algebra, coeffs);
        }

        return result;
    }

    #endregion

    #region Lie Group Batch Operations

    /// <inheritdoc/>
    public So3<T>[] So3ExpBatch<T>(So3Group<T> group, Vector<T>[] tangentVectors)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (tangentVectors is null) throw new ArgumentNullException(nameof(tangentVectors));

        var result = new So3<T>[tangentVectors.Length];
        for (int i = 0; i < tangentVectors.Length; i++)
        {
            result[i] = group.Exp(tangentVectors[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T>[] So3LogBatch<T>(So3Group<T> group, So3<T>[] rotations)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (rotations is null) throw new ArgumentNullException(nameof(rotations));

        var result = new Vector<T>[rotations.Length];
        for (int i = 0; i < rotations.Length; i++)
        {
            result[i] = group.Log(rotations[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public So3<T>[] So3ComposeBatch<T>(So3Group<T> group, So3<T>[] left, So3<T>[] right)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new So3<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = group.Compose(left[i], right[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Se3<T>[] Se3ExpBatch<T>(Se3Group<T> group, Vector<T>[] tangentVectors)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (tangentVectors is null) throw new ArgumentNullException(nameof(tangentVectors));

        var result = new Se3<T>[tangentVectors.Length];
        for (int i = 0; i < tangentVectors.Length; i++)
        {
            result[i] = group.Exp(tangentVectors[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T>[] Se3LogBatch<T>(Se3Group<T> group, Se3<T>[] transforms)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (transforms is null) throw new ArgumentNullException(nameof(transforms));

        var result = new Vector<T>[transforms.Length];
        for (int i = 0; i < transforms.Length; i++)
        {
            result[i] = group.Log(transforms[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Se3<T>[] Se3ComposeBatch<T>(Se3Group<T> group, Se3<T>[] left, Se3<T>[] right)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (left is null) throw new ArgumentNullException(nameof(left));
        if (right is null) throw new ArgumentNullException(nameof(right));
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Left and right arrays must have the same length.");
        }

        var result = new Se3<T>[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            result[i] = group.Compose(left[i], right[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T>[] So3AdjointBatch<T>(So3Group<T> group, So3<T>[] rotations)
    {
        if (group is null) throw new ArgumentNullException(nameof(group));
        if (rotations is null) throw new ArgumentNullException(nameof(rotations));

        var result = new Matrix<T>[rotations.Length];
        for (int i = 0; i < rotations.Length; i++)
        {
            result[i] = group.Adjoint(rotations[i]);
        }

        return result;
    }

    #endregion
}
