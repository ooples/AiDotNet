namespace AiDotNet.NumericOperations;

public class ComplexOperations<T> : INumericOperations<Complex<T>>
{
    private readonly INumericOperations<T> _ops;

    public ComplexOperations()
    {
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public Complex<T> Add(Complex<T> a, Complex<T> b) => a + b;
    public Complex<T> Subtract(Complex<T> a, Complex<T> b) => a - b;
    public Complex<T> Multiply(Complex<T> a, Complex<T> b) => a * b;
    public Complex<T> Divide(Complex<T> a, Complex<T> b) => a / b;
    public Complex<T> Negate(Complex<T> a) => new(_ops.Negate(a.Real), _ops.Negate(a.Imaginary));
    public Complex<T> Zero => new(_ops.Zero, _ops.Zero);
    public Complex<T> One => new(_ops.One, _ops.Zero);
    public Complex<T> Sqrt(Complex<T> value)
    {
        var r = _ops.Sqrt(_ops.Add(_ops.Square(value.Real), _ops.Square(value.Imaginary)));
        var theta = _ops.Divide(value.Phase, _ops.FromDouble(2));

        return new Complex<T>(
            _ops.Multiply(r, _ops.FromDouble(Math.Cos(Convert.ToDouble(theta)))),
            _ops.Multiply(r, _ops.FromDouble(Math.Sin(Convert.ToDouble(theta))))
        );
    }
    public Complex<T> FromDouble(double value) => new(_ops.FromDouble(value), _ops.Zero);
    public bool GreaterThan(Complex<T> a, Complex<T> b) => _ops.GreaterThan(a.Magnitude, b.Magnitude);
    public bool LessThan(Complex<T> a, Complex<T> b) => _ops.LessThan(a.Magnitude, b.Magnitude);
    public Complex<T> Abs(Complex<T> value) => new(value.Magnitude, _ops.Zero);
    public Complex<T> Square(Complex<T> value)
    {
        var a = value.Real;
        var b = value.Imaginary;

        return new Complex<T>(
            _ops.Subtract(_ops.Square(a), _ops.Square(b)),
            _ops.Multiply(_ops.FromDouble(2), _ops.Multiply(a, b))
        );
    }
    public Complex<T> Exp(Complex<T> value)
    {
        var expReal = _ops.Exp(value.Real);
        return new Complex<T>(
            _ops.Multiply(expReal, _ops.FromDouble(Math.Cos(Convert.ToDouble(value.Imaginary)))),
            _ops.Multiply(expReal, _ops.FromDouble(Math.Sin(Convert.ToDouble(value.Imaginary))))
        );
    }
    public bool Equals(Complex<T> a, Complex<T> b) => a == b;
    public Complex<T> Power(Complex<T> baseValue, Complex<T> exponent)
    {
        if (baseValue == Zero && exponent == Zero)
            return One;

        return Exp(Multiply(Log(baseValue), exponent));
    }
    public Complex<T> Log(Complex<T> value)
    {
        return new Complex<T>(_ops.Log(value.Magnitude), value.Phase);
    }
    public bool GreaterThanOrEquals(Complex<T> a, Complex<T> b)
    {
        return _ops.GreaterThanOrEquals(a.Magnitude, b.Magnitude);
    }
    public bool LessThanOrEquals(Complex<T> a, Complex<T> b)
    {
        return _ops.LessThanOrEquals(a.Magnitude, b.Magnitude);
    }
    public Complex<T> Round(Complex<T> value) => new(_ops.Round(value.Real), _ops.Round(value.Imaginary));
    public Complex<T> MinValue => new(_ops.MinValue, _ops.MinValue);
    public Complex<T> MaxValue => new(_ops.MaxValue, _ops.MaxValue);
    public bool IsNaN(Complex<T> value) => _ops.IsNaN(value.Real) || _ops.IsNaN(value.Imaginary);
    public bool IsInfinity(Complex<T> value) => _ops.IsInfinity(value.Real) || _ops.IsInfinity(value.Imaginary);
    public Complex<T> SignOrZero(Complex<T> value)
    {
        if (_ops.GreaterThan(value.Real, _ops.Zero) || (_ops.Equals(value.Real, _ops.Zero) && _ops.GreaterThan(value.Imaginary, _ops.Zero)))
            return One;
        if (_ops.LessThan(value.Real, _ops.Zero) || (_ops.Equals(value.Real, _ops.Zero) && _ops.LessThan(value.Imaginary, _ops.Zero)))
            return Negate(One);
        return Zero;
    }
}