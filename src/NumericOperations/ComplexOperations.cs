namespace AiDotNet.NumericOperations;

public class ComplexOperations : INumericOperations<Complex>
{
    public Complex Add(Complex a, Complex b) => a + b;
    public Complex Subtract(Complex a, Complex b) => a - b;
    public Complex Multiply(Complex a, Complex b) => a * b;
    public Complex Divide(Complex a, Complex b) => a / b;
    public Complex Negate(Complex a) => -a;
    public Complex Zero => new(0, 0);
    public Complex One => new(1, 0);
    public Complex Sqrt(Complex value)
    {
        double r = Math.Sqrt(value.Magnitude);
        double theta = value.Phase / 2;

        return new Complex(r * Math.Cos(theta), r * Math.Sin(theta));
    }
    public Complex FromDouble(double value) => new(value, 0);
    public bool GreaterThan(Complex a, Complex b) => a.Magnitude > b.Magnitude;
    public bool LessThan(Complex a, Complex b) => a.Magnitude < b.Magnitude;
    public Complex Abs(Complex value) => new(value.Magnitude, 0);
    public Complex Square(Complex value)
    {
        double a = value.Real;
        double b = value.Imaginary;

        return new Complex(a * a - b * b, 2 * a * b);
    }
    public Complex Exp(Complex value)
    {
        double expReal = Math.Exp(value.Real);
        return new Complex(expReal * Math.Cos(value.Imaginary), expReal * Math.Sin(value.Imaginary));
    }
    public bool Equals(Complex a, Complex b) => a == b;
    public Complex Power(Complex baseValue, Complex exponent)
    {
        if (baseValue == Zero && exponent == Zero)
            return One;

        return Exp(Multiply(Log(baseValue), exponent));
    }
    public Complex Log(Complex value)
    {
        return new Complex(Math.Log(value.Magnitude), value.Phase);
    }
    public bool GreaterThanOrEquals(Complex a, Complex b)
    {
        return a.Magnitude >= b.Magnitude;
    }
    public bool LessThanOrEquals(Complex a, Complex b)
    {
        return a.Magnitude <= b.Magnitude;
    }
    public int ToInt32(Complex value) => (int)Math.Round(value.Real);
    public Complex Round(Complex value) => new(Math.Round(value.Real), Math.Round(value.Imaginary));
}