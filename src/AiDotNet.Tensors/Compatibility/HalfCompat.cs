using System;
using AiDotNet.Tensors.Helpers;

#if !NET5_0_OR_GREATER
namespace System
{
    /// <summary>
    /// Compatibility shim for Half (FP16) type on .NET Framework 4.6.2 and .NET Standard.
    /// Uses float internally but provides Half interface for API compatibility.
    /// </summary>
    public readonly struct Half : IComparable, IFormattable, IComparable<Half>, IEquatable<Half>
    {
        private readonly float _value;

        private Half(float value)
        {
            _value = value;
        }

        public static Half MinValue => new Half(float.MinValue);
        public static Half MaxValue => new Half(float.MaxValue);
        public static Half Epsilon => new Half(float.Epsilon);
        public static Half NaN => new Half(float.NaN);
        public static Half NegativeInfinity => new Half(float.NegativeInfinity);
        public static Half PositiveInfinity => new Half(float.PositiveInfinity);

        public static implicit operator Half(float value) => new Half(value);
        public static explicit operator float(Half value) => value._value;
        public static explicit operator Half(double value) => new Half((float)value);
        public static explicit operator double(Half value) => value._value;

        public static explicit operator Half(int value) => new Half(value);
        public static explicit operator Half(long value) => new Half(value);
        public static explicit operator Half(byte value) => new Half(value);
        public static explicit operator Half(short value) => new Half(value);
        public static explicit operator Half(uint value) => new Half(value);
        public static explicit operator Half(ulong value) => new Half(value);
        public static explicit operator Half(ushort value) => new Half(value);
        public static explicit operator Half(sbyte value) => new Half(value);
        public static explicit operator Half(decimal value) => new Half((float)value);

        public static bool IsNaN(Half value) => float.IsNaN(value._value);
        public static bool IsInfinity(Half value) => float.IsInfinity(value._value);
        public static bool IsPositiveInfinity(Half value) => float.IsPositiveInfinity(value._value);
        public static bool IsNegativeInfinity(Half value) => float.IsNegativeInfinity(value._value);

        public int CompareTo(object obj)
        {
            if (obj is Half other)
                return _value.CompareTo(other._value);
            throw new ArgumentException("Object must be of type Half");
        }

        public int CompareTo(Half other) => _value.CompareTo(other._value);
        public bool Equals(Half other) => _value.Equals(other._value);
        public override bool Equals(object obj) => obj is Half other && Equals(other);
        public override int GetHashCode() => _value.GetHashCode();
        public override string ToString() => _value.ToString();
        public string ToString(string format) => _value.ToString(format);
        public string ToString(IFormatProvider provider) => _value.ToString(provider);
        public string ToString(string format, IFormatProvider provider) => _value.ToString(format, provider);

        public static bool operator ==(Half left, Half right) => left._value == right._value;
        public static bool operator !=(Half left, Half right) => left._value != right._value;
        public static bool operator <(Half left, Half right) => left._value < right._value;
        public static bool operator >(Half left, Half right) => left._value > right._value;
        public static bool operator <=(Half left, Half right) => left._value <= right._value;
        public static bool operator >=(Half left, Half right) => left._value >= right._value;
        public static Half operator -(Half value) => new Half(-value._value);
    }
}
#endif

namespace System
{
    public static class MathExtensions
    {
#if !NET5_0_OR_GREATER
        public static T Clamp<T>(T value, T min, T max) where T : IComparable<T>
        {
            if (value.CompareTo(min) < 0) return min;
            if (value.CompareTo(max) > 0) return max;
            return value;
        }

        public static int Clamp(int value, int min, int max)
        {
            return MathHelper.Clamp(value, min, max);
        }

        public static long Clamp(long value, long min, long max)
        {
            return MathHelper.Clamp(value, min, max);
        }
#else
        // For NET5+, delegate int/long to Math.Clamp; generic T uses custom implementation
        public static T Clamp<T>(T value, T min, T max) where T : IComparable<T>
        {
            if (value.CompareTo(min) < 0) return min;
            if (value.CompareTo(max) > 0) return max;
            return value;
        }

        public static int Clamp(int value, int min, int max)
        {
            return MathHelper.Clamp(value, min, max);
        }

        public static long Clamp(long value, long min, long max)
        {
            return MathHelper.Clamp(value, min, max);
        }
#endif
    }
}
