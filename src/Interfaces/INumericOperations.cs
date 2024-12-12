namespace AiDotNet.Interfaces;

public interface INumericOperations<T>
{
    T Add(T a, T b);
    T Subtract(T a, T b);
    T Multiply(T a, T b);
    T Divide(T a, T b);
    T Negate(T a);
    T Zero { get; }
    T One { get; }
    T Sqrt(T value);
    T FromDouble(double value);
    bool GreaterThan(T a, T b);
    bool LessThan(T a, T b);
    T Abs(T value);
    T Square(T value);
    T Exp(T value);
    bool Equals(T a, T b);
    T Power(T baseValue, T exponent);
    T Log(T value);
    bool GreaterThanOrEquals(T a, T b);
    bool LessThanOrEquals(T a, T b);
    int ToInt32(T value);
    T Round(T value);
    T MinValue { get; }
    T MaxValue { get; }
    bool IsNaN(T value);
    bool IsInfinity(T value);
}