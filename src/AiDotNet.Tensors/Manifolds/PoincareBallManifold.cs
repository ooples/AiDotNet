using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Manifolds;

/// <summary>
/// Provides core operations for the Poincare ball model of hyperbolic space.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
public sealed class PoincareBallManifold<T>
{
    private readonly INumericOperations<T> _ops;
    private readonly T _epsilon;

    public T Curvature { get; }

    public PoincareBallManifold()
    {
        _ops = MathHelper.GetNumericOperations<T>();
        Curvature = _ops.FromDouble(1.0);
        _epsilon = _ops.FromDouble(1e-12);
        ValidateCurvature();
    }

    public PoincareBallManifold(T curvature)
    {
        _ops = MathHelper.GetNumericOperations<T>();
        Curvature = curvature;
        _epsilon = _ops.FromDouble(1e-12);
        ValidateCurvature();
    }

    public Vector<T> MobiusAdd(Vector<T> x, Vector<T> y)
    {
        EnsureSameLength(x, y);

        T x2 = Dot(x, x);
        T y2 = Dot(y, y);
        T xy = Dot(x, y);
        T two = _ops.FromDouble(2.0);

        T cxy2 = _ops.Multiply(two, _ops.Multiply(Curvature, xy));
        T numeratorScaleX = _ops.Add(_ops.Add(_ops.One, cxy2), _ops.Multiply(Curvature, y2));
        T numeratorScaleY = _ops.Subtract(_ops.One, _ops.Multiply(Curvature, x2));
        T denom = _ops.Add(_ops.One, _ops.Add(cxy2, _ops.Multiply(_ops.Multiply(Curvature, Curvature), _ops.Multiply(x2, y2))));

        denom = SafeDenominator(denom);
        var numerator = Add(Scale(x, numeratorScaleX), Scale(y, numeratorScaleY));
        return Scale(numerator, _ops.Divide(_ops.One, denom));
    }

    public Vector<T> ExpMap(Vector<T> tangent, Vector<T> basePoint)
    {
        EnsureSameLength(tangent, basePoint);

        T norm = Norm(tangent);
        if (_ops.Equals(norm, _ops.Zero))
            return basePoint;

        T sqrtC = _ops.Sqrt(Curvature);
        T lambda = Lambda(basePoint);
        T scaledNorm = _ops.Divide(_ops.Multiply(_ops.Multiply(sqrtC, lambda), norm), _ops.FromDouble(2.0));
        double tanhArg = Convert.ToDouble(scaledNorm);
        T tanh = _ops.FromDouble(Math.Tanh(tanhArg));
        T scale = _ops.Divide(tanh, _ops.Multiply(sqrtC, norm));

        var direction = Scale(tangent, scale);
        return MobiusAdd(basePoint, direction);
    }

    public Vector<T> LogMap(Vector<T> point, Vector<T> basePoint)
    {
        EnsureSameLength(point, basePoint);

        var diff = MobiusAdd(Negate(basePoint), point);
        T norm = Norm(diff);
        if (_ops.Equals(norm, _ops.Zero))
            return CreateZeroVector(point.Length);

        T sqrtC = _ops.Sqrt(Curvature);
        T lambda = Lambda(basePoint);
        T arg = _ops.Multiply(sqrtC, norm);
        double atanh = MathHelper.Atanh(Convert.ToDouble(arg));
        T factor = _ops.Divide(_ops.Multiply(_ops.FromDouble(2.0), _ops.FromDouble(atanh)), _ops.Multiply(sqrtC, lambda));
        T scale = _ops.Divide(factor, norm);
        return Scale(diff, scale);
    }

    public T Distance(Vector<T> x, Vector<T> y)
    {
        EnsureSameLength(x, y);
        var diff = MobiusAdd(Negate(x), y);
        T norm = Norm(diff);
        T sqrtC = _ops.Sqrt(Curvature);
        double atanh = MathHelper.Atanh(Convert.ToDouble(_ops.Multiply(sqrtC, norm)));
        T factor = _ops.Multiply(_ops.FromDouble(2.0), _ops.FromDouble(atanh));
        return _ops.Divide(factor, sqrtC);
    }

    public Vector<T> ParallelTransport(Vector<T> x, Vector<T> y, Vector<T> v)
    {
        EnsureSameLength(x, y);
        EnsureSameLength(x, v);

        var gyr = Gyration(y, Negate(x), v);
        T lambdaX = Lambda(x);
        T lambdaY = Lambda(y);
        T scale = _ops.Divide(lambdaX, lambdaY);
        return Scale(gyr, scale);
    }

    private Vector<T> Gyration(Vector<T> a, Vector<T> b, Vector<T> v)
    {
        var ab = MobiusAdd(a, b);
        var bv = MobiusAdd(b, v);
        var abv = MobiusAdd(a, bv);
        return MobiusAdd(abv, Negate(ab));
    }

    private T Lambda(Vector<T> x)
    {
        T x2 = Dot(x, x);
        T denom = _ops.Subtract(_ops.One, _ops.Multiply(Curvature, x2));
        denom = SafeDenominator(denom);
        return _ops.Divide(_ops.FromDouble(2.0), denom);
    }

    private T Dot(Vector<T> a, Vector<T> b)
    {
        EnsureSameLength(a, b);
        return _ops.Dot(a.AsSpan(), b.AsSpan());
    }

    private T Norm(Vector<T> v)
    {
        return _ops.Sqrt(Dot(v, v));
    }

    private Vector<T> Add(Vector<T> a, Vector<T> b)
    {
        EnsureSameLength(a, b);
        var result = new T[a.Length];
        _ops.Add(a.AsSpan(), b.AsSpan(), result);
        return new Vector<T>(result);
    }

    private Vector<T> Scale(Vector<T> v, T scalar)
    {
        var result = new T[v.Length];
        _ops.MultiplyScalar(v.AsSpan(), scalar, result);
        return new Vector<T>(result);
    }

    private Vector<T> Negate(Vector<T> v)
    {
        var result = new T[v.Length];
        _ops.Negate(v.AsSpan(), result);
        return new Vector<T>(result);
    }

    private Vector<T> CreateZeroVector(int length)
    {
        var data = new T[length];
        _ops.Fill(data, _ops.Zero);
        return new Vector<T>(data);
    }

    private T SafeDenominator(T value)
    {
        if (_ops.GreaterThan(_ops.Abs(value), _epsilon))
            return value;

        return _ops.GreaterThanOrEquals(value, _ops.Zero) ? _epsilon : _ops.Negate(_epsilon);
    }

    private void EnsureSameLength(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");
    }

    private void ValidateCurvature()
    {
        if (Convert.ToDouble(Curvature) <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(Curvature), "Curvature must be positive.");
    }
}
