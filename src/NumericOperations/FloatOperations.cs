namespace AiDotNet.NumericOperations;

public class FloatOperations : INumericOperations<float>
{
    public float Add(float a, float b) => a + b;
    public float Subtract(float a, float b) => a - b;
    public float Multiply(float a, float b) => a * b;
    public float Divide(float a, float b) => a / b;
    public float Negate(float a) => -a;
    public float Zero => 0f;
    public float One => 1f;
    public float Sqrt(float value) => (float)Math.Sqrt(value);
    public float FromDouble(double value) => (float)value;
    public bool GreaterThan(float a, float b) => a > b;
    public bool LessThan(float a, float b) => a < b;
    public float Abs(float value) => Math.Abs(value);
    public float Square(float value) => Multiply(value, value);
    public float Exp(float value) => (float)Math.Exp(value);
    public bool Equals(float a, float b) => a == b;
    public float Power(float baseValue, float exponent) => (float)Math.Pow(baseValue, exponent);
    public float Log(float value) => (float)Math.Log(value);
    public bool GreaterThanOrEquals(float a, float b) => a >= b;
    public bool LessThanOrEquals(float a, float b) => a <= b;
    public int ToInt32(float value) => (int)Math.Round(value);
    public float Round(float value) => (float)Math.Round((double)value);
    public float MinValue => float.MinValue;
    public float MaxValue => float.MaxValue;
    public bool IsNaN(float value) => float.IsNaN(value);
    public bool IsInfinity(float value) => float.IsInfinity(value);
    public float SignOrZero(float value)
    {
        if (value > 0) return 1f;
        if (value < 0) return -1f;
        return 0f;
    }
}