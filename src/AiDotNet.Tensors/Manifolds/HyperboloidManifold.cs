using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Manifolds;

/// <summary>
/// Provides core operations for the hyperboloid model of hyperbolic space.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
public sealed class HyperboloidManifold<T>
{
    private readonly INumericOperations<T> _ops;
    private readonly T _epsilon;

    public T Curvature { get; }

    public HyperboloidManifold()
    {
        _ops = MathHelper.GetNumericOperations<T>();
        Curvature = _ops.FromDouble(1.0);
        _epsilon = _ops.FromDouble(1e-12);
        ValidateCurvature();
    }

    public HyperboloidManifold(T curvature)
    {
        _ops = MathHelper.GetNumericOperations<T>();
        Curvature = curvature;
        _epsilon = _ops.FromDouble(1e-12);
        ValidateCurvature();
    }

    public Vector<T> ExpMap(Vector<T> tangent, Vector<T> basePoint)
    {
        EnsureSameLength(tangent, basePoint);

        T sqrtC = _ops.Sqrt(Curvature);
        var baseScaled = Scale(basePoint, sqrtC);
        var tangentScaled = Scale(tangent, sqrtC);

        T norm = MinkowskiNorm(tangentScaled);
        if (_ops.Equals(norm, _ops.Zero))
            return basePoint;

        double normD = Convert.ToDouble(norm);
        T cosh = _ops.FromDouble(Math.Cosh(normD));
        T sinh = _ops.FromDouble(Math.Sinh(normD));
        T scale = _ops.Divide(sinh, norm);

        var term1 = Scale(baseScaled, cosh);
        var term2 = Scale(tangentScaled, scale);
        var resultScaled = Add(term1, term2);
        return Scale(resultScaled, _ops.Divide(_ops.One, sqrtC));
    }

    public Vector<T> LogMap(Vector<T> point, Vector<T> basePoint)
    {
        EnsureSameLength(point, basePoint);

        T sqrtC = _ops.Sqrt(Curvature);
        var baseScaled = Scale(basePoint, sqrtC);
        var pointScaled = Scale(point, sqrtC);

        T dot = MinkowskiDot(baseScaled, pointScaled);
        double alpha = -Convert.ToDouble(dot);
        if (alpha < 1.0 + Convert.ToDouble(_epsilon))
            return CreateZeroVector(point.Length);

        double dist = Acosh(alpha);
        double denom = Math.Sqrt(alpha * alpha - 1.0);
        double factor = dist / denom;

        var scaled = Scale(Subtract(pointScaled, Scale(baseScaled, _ops.FromDouble(alpha))), _ops.FromDouble(factor));
        return Scale(scaled, _ops.Divide(_ops.One, sqrtC));
    }

    public T Distance(Vector<T> x, Vector<T> y)
    {
        EnsureSameLength(x, y);

        // Scale inputs by sqrt(Curvature) for consistency with ExpMap, LogMap, ParallelTransport
        T sqrtC = _ops.Sqrt(Curvature);
        var xScaled = Scale(x, sqrtC);
        var yScaled = Scale(y, sqrtC);

        T dot = MinkowskiDot(xScaled, yScaled);
        double alpha = -Convert.ToDouble(dot);
        alpha = Math.Max(alpha, 1.0);
        double dist = Acosh(alpha);
        return _ops.Divide(_ops.FromDouble(dist), sqrtC);
    }

    public Vector<T> ParallelTransport(Vector<T> x, Vector<T> y, Vector<T> v)
    {
        EnsureSameLength(x, y);
        EnsureSameLength(x, v);

        T sqrtC = _ops.Sqrt(Curvature);
        var xScaled = Scale(x, sqrtC);
        var yScaled = Scale(y, sqrtC);
        var vScaled = Scale(v, sqrtC);

        T denom = _ops.Add(_ops.One, MinkowskiDot(xScaled, yScaled));
        denom = SafeDenominator(denom);
        T factor = _ops.Divide(MinkowskiDot(yScaled, vScaled), denom);
        var correction = Scale(Add(xScaled, yScaled), factor);
        var transported = Subtract(vScaled, correction);
        return Scale(transported, _ops.Divide(_ops.One, sqrtC));
    }

    private T MinkowskiDot(Vector<T> a, Vector<T> b)
    {
        EnsureSameLength(a, b);
        if (a.Length == 0)
            throw new ArgumentException("Vectors must have at least one dimension.");

        T sum = _ops.Negate(_ops.Multiply(a[0], b[0]));
        for (int i = 1; i < a.Length; i++)
            sum = _ops.Add(sum, _ops.Multiply(a[i], b[i]));
        return sum;
    }

    private T MinkowskiNorm(Vector<T> v)
    {
        T dot = MinkowskiDot(v, v);
        return _ops.Sqrt(_ops.Abs(dot));
    }

    private Vector<T> Add(Vector<T> a, Vector<T> b)
    {
        EnsureSameLength(a, b);
        var result = new T[a.Length];
        _ops.Add(a.AsSpan(), b.AsSpan(), result);
        return new Vector<T>(result);
    }

    private Vector<T> Subtract(Vector<T> a, Vector<T> b)
    {
        EnsureSameLength(a, b);
        var result = new T[a.Length];
        _ops.Subtract(a.AsSpan(), b.AsSpan(), result);
        return new Vector<T>(result);
    }

    private Vector<T> Scale(Vector<T> v, T scalar)
    {
        var result = new T[v.Length];
        _ops.MultiplyScalar(v.AsSpan(), scalar, result);
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

    private static double Acosh(double x)
    {
        if (x < 1.0)
            throw new ArgumentOutOfRangeException(nameof(x), "Acosh is only defined for x >= 1.");

        return Math.Log(x + Math.Sqrt(x * x - 1.0));
    }
}
