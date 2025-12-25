using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Lie group operations for SO(3).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class So3Group<T> : ILieGroup<T, So3<T>>
{
    private readonly INumericOperations<T> _ops;
    public So3Group()
    {
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public So3<T> Identity => So3<T>.Identity;

    public So3<T> Compose(So3<T> a, So3<T> b)
        => new So3<T>((Matrix<T>)a.Matrix.Multiply(b.Matrix));

    public So3<T> Inverse(So3<T> value)
        => new So3<T>(value.Matrix.Transpose());

    public So3<T> Exp(Vector<T> tangent)
    {
        if (tangent.Length != 3)
            throw new ArgumentException("SO(3) tangent vectors must be length 3.");

        double wx = Convert.ToDouble(tangent[0]);
        double wy = Convert.ToDouble(tangent[1]);
        double wz = Convert.ToDouble(tangent[2]);

        double theta = Math.Sqrt(wx * wx + wy * wy + wz * wz);
        double a;
        double b;

        if (theta < 1e-8)
        {
            a = 1.0 - theta * theta / 6.0;
            b = 0.5 - theta * theta / 24.0;
        }
        else
        {
            a = Math.Sin(theta) / theta;
            b = (1.0 - Math.Cos(theta)) / (theta * theta);
        }

        double[,] k = new double[3, 3]
        {
            { 0.0, -wz,  wy },
            {  wz, 0.0, -wx },
            { -wy,  wx, 0.0 }
        };

        double[,] k2 = Multiply(k, k);
        double[,] r = new double[3, 3];

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double identity = i == j ? 1.0 : 0.0;
                r[i, j] = identity + a * k[i, j] + b * k2[i, j];
            }
        }

        var result = new Matrix<T>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                result[i, j] = _ops.FromDouble(r[i, j]);
            }
        }

        return new So3<T>(result);
    }

    public Vector<T> Log(So3<T> value)
    {
        double r00 = Convert.ToDouble(value.Matrix[0, 0]);
        double r11 = Convert.ToDouble(value.Matrix[1, 1]);
        double r22 = Convert.ToDouble(value.Matrix[2, 2]);
        double trace = r00 + r11 + r22;
        double cosTheta = (trace - 1.0) / 2.0;
        cosTheta = Math.Max(-1.0, Math.Min(1.0, cosTheta));
        double theta = Math.Acos(cosTheta);

        if (theta < 1e-8)
        {
            return new Vector<T>(new[]
            {
                _ops.Zero,
                _ops.Zero,
                _ops.Zero
            });
        }

        // Handle θ near π where sin(θ) approaches 0
        if (theta > Math.PI - 1e-6)
        {
            // For θ ≈ π, extract axis from diagonal of (R + I)/2
            // The axis is the eigenvector with eigenvalue 1
            double rx = Math.Sqrt(Math.Max(0.0, (r00 + 1.0) / 2.0));
            double ry = Math.Sqrt(Math.Max(0.0, (r11 + 1.0) / 2.0));
            double rz = Math.Sqrt(Math.Max(0.0, (r22 + 1.0) / 2.0));

            // Determine signs from off-diagonal elements
            if (Convert.ToDouble(value.Matrix[0, 1]) + Convert.ToDouble(value.Matrix[1, 0]) < 0) ry = -ry;
            if (Convert.ToDouble(value.Matrix[0, 2]) + Convert.ToDouble(value.Matrix[2, 0]) < 0) rz = -rz;

            return new Vector<T>(new[]
            {
                _ops.FromDouble(rx * theta),
                _ops.FromDouble(ry * theta),
                _ops.FromDouble(rz * theta)
            });
        }

        double factor = theta / (2.0 * Math.Sin(theta));
        double wx = factor * (Convert.ToDouble(value.Matrix[2, 1]) - Convert.ToDouble(value.Matrix[1, 2]));
        double wy = factor * (Convert.ToDouble(value.Matrix[0, 2]) - Convert.ToDouble(value.Matrix[2, 0]));
        double wz = factor * (Convert.ToDouble(value.Matrix[1, 0]) - Convert.ToDouble(value.Matrix[0, 1]));

        return new Vector<T>(new[]
        {
            _ops.FromDouble(wx),
            _ops.FromDouble(wy),
            _ops.FromDouble(wz)
        });
    }

    public Matrix<T> Adjoint(So3<T> value)
        => value.Matrix;

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
