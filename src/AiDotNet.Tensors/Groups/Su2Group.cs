using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Groups;

/// <summary>
/// Lie group operations for SU(2).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class Su2Group<T> : ILieGroup<T, Su2<T>>
{
    private readonly INumericOperations<T> _ops;

    public Su2Group()
    {
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public Su2<T> Identity => new Su2<T>(_ops.One, _ops.Zero, _ops.Zero, _ops.Zero);

    public Su2<T> Compose(Su2<T> a, Su2<T> b)
    {
        T w = _ops.Subtract(_ops.Subtract(_ops.Subtract(_ops.Multiply(a.W, b.W), _ops.Multiply(a.X, b.X)), _ops.Multiply(a.Y, b.Y)), _ops.Multiply(a.Z, b.Z));
        T x = _ops.Add(_ops.Add(_ops.Multiply(a.W, b.X), _ops.Multiply(a.X, b.W)), _ops.Subtract(_ops.Multiply(a.Y, b.Z), _ops.Multiply(a.Z, b.Y)));
        T y = _ops.Add(_ops.Add(_ops.Multiply(a.W, b.Y), _ops.Multiply(a.Y, b.W)), _ops.Subtract(_ops.Multiply(a.Z, b.X), _ops.Multiply(a.X, b.Z)));
        T z = _ops.Add(_ops.Add(_ops.Multiply(a.W, b.Z), _ops.Multiply(a.Z, b.W)), _ops.Subtract(_ops.Multiply(a.X, b.Y), _ops.Multiply(a.Y, b.X)));

        return new Su2<T>(w, x, y, z);
    }

    public Su2<T> Inverse(Su2<T> value)
        => new Su2<T>(value.W, _ops.Negate(value.X), _ops.Negate(value.Y), _ops.Negate(value.Z));

    public Su2<T> Exp(Vector<T> tangent)
    {
        if (tangent.Length != 3)
            throw new ArgumentException("SU(2) tangent vectors must be length 3.");

        double x = Convert.ToDouble(tangent[0]);
        double y = Convert.ToDouble(tangent[1]);
        double z = Convert.ToDouble(tangent[2]);
        double theta = Math.Sqrt(x * x + y * y + z * z);

        if (theta < 1e-8)
        {
            return new Su2<T>(_ops.One,
                _ops.FromDouble(0.5 * x),
                _ops.FromDouble(0.5 * y),
                _ops.FromDouble(0.5 * z));
        }

        double halfTheta = 0.5 * theta;
        double sinHalf = Math.Sin(halfTheta);
        double scale = sinHalf / theta;

        return new Su2<T>(
            _ops.FromDouble(Math.Cos(halfTheta)),
            _ops.FromDouble(x * scale),
            _ops.FromDouble(y * scale),
            _ops.FromDouble(z * scale));
    }

    public Vector<T> Log(Su2<T> value)
    {
        double w = Convert.ToDouble(value.W);
        double x = Convert.ToDouble(value.X);
        double y = Convert.ToDouble(value.Y);
        double z = Convert.ToDouble(value.Z);
        double norm = Math.Sqrt(x * x + y * y + z * z);

        if (norm < 1e-8)
        {
            return new Vector<T>(new[] { _ops.Zero, _ops.Zero, _ops.Zero });
        }

        double theta = 2.0 * Math.Atan2(norm, w);
        double scale = theta / norm;

        return new Vector<T>(new[]
        {
            _ops.FromDouble(x * scale),
            _ops.FromDouble(y * scale),
            _ops.FromDouble(z * scale)
        });
    }

    public Matrix<T> Adjoint(Su2<T> value)
    {
        double w = Convert.ToDouble(value.W);
        double x = Convert.ToDouble(value.X);
        double y = Convert.ToDouble(value.Y);
        double z = Convert.ToDouble(value.Z);

        double r00 = 1.0 - 2.0 * (y * y + z * z);
        double r01 = 2.0 * (x * y - w * z);
        double r02 = 2.0 * (x * z + w * y);

        double r10 = 2.0 * (x * y + w * z);
        double r11 = 1.0 - 2.0 * (x * x + z * z);
        double r12 = 2.0 * (y * z - w * x);

        double r20 = 2.0 * (x * z - w * y);
        double r21 = 2.0 * (y * z + w * x);
        double r22 = 1.0 - 2.0 * (x * x + y * y);

        var result = new Matrix<T>(3, 3);
        result[0, 0] = _ops.FromDouble(r00);
        result[0, 1] = _ops.FromDouble(r01);
        result[0, 2] = _ops.FromDouble(r02);
        result[1, 0] = _ops.FromDouble(r10);
        result[1, 1] = _ops.FromDouble(r11);
        result[1, 2] = _ops.FromDouble(r12);
        result[2, 0] = _ops.FromDouble(r20);
        result[2, 1] = _ops.FromDouble(r21);
        result[2, 2] = _ops.FromDouble(r22);

        return result;
    }
}
