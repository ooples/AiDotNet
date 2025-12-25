using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Lie group operations for SE(3).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class Se3Group<T> : ILieGroup<T, Se3<T>>
{
    private readonly INumericOperations<T> _ops;
    private readonly So3Group<T> _so3;

    public Se3Group()
    {
        _ops = MathHelper.GetNumericOperations<T>();
        _so3 = new So3Group<T>();
    }

    public Se3<T> Identity => new Se3<T>(So3<T>.Identity, new Vector<T>(new[] { _ops.Zero, _ops.Zero, _ops.Zero }));

    public Se3<T> Compose(Se3<T> a, Se3<T> b)
    {
        var rotation = _so3.Compose(a.Rotation, b.Rotation);
        var rotated = Multiply(a.Rotation.Matrix, b.Translation);
        var translation = Add(a.Translation, rotated);
        return new Se3<T>(rotation, translation);
    }

    public Se3<T> Inverse(Se3<T> value)
    {
        var rotationInv = _so3.Inverse(value.Rotation);
        var neg = Negate(value.Translation);
        var translationInv = Multiply(rotationInv.Matrix, neg);
        return new Se3<T>(rotationInv, translationInv);
    }

    public Se3<T> Exp(Vector<T> tangent)
    {
        if (tangent.Length != 6)
            throw new ArgumentException("SE(3) tangent vectors must be length 6.");

        var omega = new Vector<T>(new[] { tangent[0], tangent[1], tangent[2] });
        var v = new Vector<T>(new[] { tangent[3], tangent[4], tangent[5] });

        var rotation = _so3.Exp(omega);
        var vMatrix = BuildVMatrix(omega);
        var translation = Multiply(vMatrix, v);
        return new Se3<T>(rotation, translation);
    }

    public Vector<T> Log(Se3<T> value)
    {
        var omega = _so3.Log(value.Rotation);
        double wx = Convert.ToDouble(omega[0]);
        double wy = Convert.ToDouble(omega[1]);
        double wz = Convert.ToDouble(omega[2]);
        double theta = Math.Sqrt(wx * wx + wy * wy + wz * wz);

        double[,] k = new double[3, 3]
        {
            { 0.0, -wz,  wy },
            {  wz, 0.0, -wx },
            { -wy,  wx, 0.0 }
        };
        double[,] k2 = Multiply(k, k);

        double a;
        if (theta < 1e-8)
        {
            a = 1.0 / 12.0;
        }
        else
        {
            a = (1.0 / (theta * theta)) - (1.0 + Math.Cos(theta)) / (2.0 * theta * Math.Sin(theta));
        }

        double[,] vInv = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double identity = i == j ? 1.0 : 0.0;
                vInv[i, j] = identity - 0.5 * k[i, j] + a * k2[i, j];
            }
        }

        var v = Multiply(vInv, value.Translation);
        return new Vector<T>(new[]
        {
            omega[0],
            omega[1],
            omega[2],
            v[0],
            v[1],
            v[2]
        });
    }

    public Matrix<T> Adjoint(Se3<T> value)
    {
        var adj = new Matrix<T>(6, 6);
        var r = value.Rotation.Matrix;
        var tSkew = Skew(value.Translation);
        var tr = Multiply(tSkew, r);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                adj[i, j] = r[i, j];
                adj[i + 3, j + 3] = r[i, j];
                adj[i + 3, j] = tr[i, j];
            }
        }

        return adj;
    }

    private Matrix<T> BuildVMatrix(Vector<T> omega)
    {
        double wx = Convert.ToDouble(omega[0]);
        double wy = Convert.ToDouble(omega[1]);
        double wz = Convert.ToDouble(omega[2]);
        double theta = Math.Sqrt(wx * wx + wy * wy + wz * wz);

        double a;
        double b;
        if (theta < 1e-8)
        {
            a = 0.5 - theta * theta / 24.0;
            b = 1.0 / 6.0 - theta * theta / 120.0;
        }
        else
        {
            a = (1.0 - Math.Cos(theta)) / (theta * theta);
            b = (theta - Math.Sin(theta)) / (theta * theta * theta);
        }

        double[,] k = new double[3, 3]
        {
            { 0.0, -wz,  wy },
            {  wz, 0.0, -wx },
            { -wy,  wx, 0.0 }
        };
        double[,] k2 = Multiply(k, k);

        var vMatrix = new Matrix<T>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double identity = i == j ? 1.0 : 0.0;
                double value = identity + a * k[i, j] + b * k2[i, j];
                vMatrix[i, j] = _ops.FromDouble(value);
            }
        }

        return vMatrix;
    }

    private Vector<T> Multiply(Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Columns != vector.Length)
            throw new ArgumentException("Matrix and vector dimensions are incompatible.");

        var result = new T[matrix.Rows];
        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = _ops.Zero;
            for (int j = 0; j < matrix.Columns; j++)
                sum = _ops.Add(sum, _ops.Multiply(matrix[i, j], vector[j]));
            result[i] = sum;
        }

        return new Vector<T>(result);
    }
    private Matrix<T> Multiply(Matrix<T> a, Matrix<T> b)
    {
        if (a.Columns != b.Rows)
            throw new ArgumentException("Matrix dimensions are incompatible for multiplication.");

        var result = new Matrix<T>(a.Rows, b.Columns);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Columns; j++)
            {
                T sum = _ops.Zero;
                for (int k = 0; k < a.Columns; k++)
                    sum = _ops.Add(sum, _ops.Multiply(a[i, k], b[k, j]));
                result[i, j] = sum;
            }
        }

        return result;
    }

    private Vector<T> Multiply(double[,] matrix, Vector<T> vector)
    {
        if (matrix.GetLength(1) != vector.Length)
            throw new ArgumentException("Matrix and vector dimensions are incompatible.");

        var result = new T[matrix.GetLength(0)];
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            double sum = 0.0;
            for (int j = 0; j < matrix.GetLength(1); j++)
                sum += matrix[i, j] * Convert.ToDouble(vector[j]);
            result[i] = _ops.FromDouble(sum);
        }

        return new Vector<T>(result);
    }

    private Matrix<T> Skew(Vector<T> vector)
    {
        if (vector.Length != 3)
            throw new ArgumentException("Skew requires a 3D vector.");

        double x = Convert.ToDouble(vector[0]);
        double y = Convert.ToDouble(vector[1]);
        double z = Convert.ToDouble(vector[2]);

        var result = new Matrix<T>(3, 3);
        result[0, 0] = _ops.Zero;
        result[0, 1] = _ops.FromDouble(-z);
        result[0, 2] = _ops.FromDouble(y);
        result[1, 0] = _ops.FromDouble(z);
        result[1, 1] = _ops.Zero;
        result[1, 2] = _ops.FromDouble(-x);
        result[2, 0] = _ops.FromDouble(-y);
        result[2, 1] = _ops.FromDouble(x);
        result[2, 2] = _ops.Zero;

        return result;
    }

    private Vector<T> Add(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        var result = new T[a.Length];
        _ops.Add(a.AsSpan(), b.AsSpan(), result);
        return new Vector<T>(result);
    }

    private Vector<T> Negate(Vector<T> v)
    {
        var result = new T[v.Length];
        _ops.Negate(v.AsSpan(), result);
        return new Vector<T>(result);
    }

    private static double[,] Multiply(double[,] a, double[,] b)
    {
        var result = new double[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < 3; k++)
                    sum += a[i, k] * b[k, j];
                result[i, j] = sum;
            }
        }
        return result;
    }
}
