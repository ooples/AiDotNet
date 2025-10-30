using System;
using AiDotNet.NumericOperations;
using Xunit;

namespace AiDotNetTests.UnitTests.NumericOperations
{
    public class DoubleOperationsTests
    {
        private readonly DoubleOperations _ops;

        public DoubleOperationsTests()
        {
            _ops = new DoubleOperations();
        }

        [Fact]
        public void Zero_ReturnsZero()
        {
            Assert.Equal(0.0, _ops.Zero);
        }

        [Fact]
        public void One_ReturnsOne()
        {
            Assert.Equal(1.0, _ops.One);
        }

        [Fact]
        public void Add_TwoNumbers_ReturnsCorrectSum()
        {
            var result = _ops.Add(3.5, 2.5);
            Assert.Equal(6.0, result, 10);
        }

        [Fact]
        public void Subtract_TwoNumbers_ReturnsCorrectDifference()
        {
            var result = _ops.Subtract(10.0, 3.5);
            Assert.Equal(6.5, result, 10);
        }

        [Fact]
        public void Multiply_TwoNumbers_ReturnsCorrectProduct()
        {
            var result = _ops.Multiply(4.0, 2.5);
            Assert.Equal(10.0, result, 10);
        }

        [Fact]
        public void Divide_TwoNumbers_ReturnsCorrectQuotient()
        {
            var result = _ops.Divide(15.0, 3.0);
            Assert.Equal(5.0, result, 10);
        }

        [Fact]
        public void Divide_ByZero_ReturnsInfinity()
        {
            var result = _ops.Divide(10.0, 0.0);
            Assert.True(double.IsInfinity(result));
        }

        [Fact]
        public void Negate_PositiveNumber_ReturnsNegative()
        {
            var result = _ops.Negate(5.0);
            Assert.Equal(-5.0, result, 10);
        }

        [Fact]
        public void Negate_NegativeNumber_ReturnsPositive()
        {
            var result = _ops.Negate(-7.0);
            Assert.Equal(7.0, result, 10);
        }

        [Fact]
        public void Abs_PositiveNumber_ReturnsSameValue()
        {
            var result = _ops.Abs(5.0);
            Assert.Equal(5.0, result, 10);
        }

        [Fact]
        public void Abs_NegativeNumber_ReturnsPositiveValue()
        {
            var result = _ops.Abs(-5.0);
            Assert.Equal(5.0, result, 10);
        }

        [Fact]
        public void Sqrt_PositiveNumber_ReturnsCorrectRoot()
        {
            var result = _ops.Sqrt(25.0);
            Assert.Equal(5.0, result, 10);
        }

        [Fact]
        public void Sqrt_Zero_ReturnsZero()
        {
            var result = _ops.Sqrt(0.0);
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void Square_Number_ReturnsCorrectSquare()
        {
            var result = _ops.Square(5.0);
            Assert.Equal(25.0, result, 10);
        }

        [Fact]
        public void Exp_Zero_ReturnsOne()
        {
            var result = _ops.Exp(0.0);
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void Exp_One_ReturnsE()
        {
            var result = _ops.Exp(1.0);
            Assert.Equal(Math.E, result, 10);
        }

        [Fact]
        public void Log_E_ReturnsOne()
        {
            var result = _ops.Log(Math.E);
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void Log_One_ReturnsZero()
        {
            var result = _ops.Log(1.0);
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void Power_BaseAndExponent_ReturnsCorrectResult()
        {
            var result = _ops.Power(2.0, 3.0);
            Assert.Equal(8.0, result, 10);
        }

        [Fact]
        public void Power_ZeroExponent_ReturnsOne()
        {
            var result = _ops.Power(5.0, 0.0);
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void Sin_Zero_ReturnsZero()
        {
            var result = _ops.Sin(0.0);
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void Sin_PiOver2_ReturnsOne()
        {
            var result = _ops.Sin(Math.PI / 2.0);
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void Cos_Zero_ReturnsOne()
        {
            var result = _ops.Cos(0.0);
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void Cos_Pi_ReturnsNegativeOne()
        {
            var result = _ops.Cos(Math.PI);
            Assert.Equal(-1.0, result, 10);
        }

        [Fact]
        public void Tan_Zero_ReturnsZero()
        {
            var result = _ops.Tan(0.0);
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CompareTo_FirstLessThanSecond_ReturnsNegative()
        {
            var result = _ops.CompareTo(3.0, 5.0);
            Assert.True(result < 0);
        }

        [Fact]
        public void CompareTo_FirstGreaterThanSecond_ReturnsPositive()
        {
            var result = _ops.CompareTo(7.0, 3.0);
            Assert.True(result > 0);
        }

        [Fact]
        public void CompareTo_EqualValues_ReturnsZero()
        {
            var result = _ops.CompareTo(5.0, 5.0);
            Assert.Equal(0, result);
        }

        [Fact]
        public void GreaterThan_FirstGreater_ReturnsTrue()
        {
            var result = _ops.GreaterThan(10.0, 5.0);
            Assert.True(result);
        }

        [Fact]
        public void GreaterThan_FirstLess_ReturnsFalse()
        {
            var result = _ops.GreaterThan(3.0, 7.0);
            Assert.False(result);
        }

        [Fact]
        public void LessThan_FirstLess_ReturnsTrue()
        {
            var result = _ops.LessThan(3.0, 7.0);
            Assert.True(result);
        }

        [Fact]
        public void LessThan_FirstGreater_ReturnsFalse()
        {
            var result = _ops.LessThan(10.0, 5.0);
            Assert.False(result);
        }

        [Fact]
        public void Equals_SameValues_ReturnsTrue()
        {
            var result = _ops.Equals(5.0, 5.0);
            Assert.True(result);
        }

        [Fact]
        public void Equals_DifferentValues_ReturnsFalse()
        {
            var result = _ops.Equals(5.0, 6.0);
            Assert.False(result);
        }

        [Fact]
        public void Max_FirstGreater_ReturnsFirst()
        {
            var result = _ops.Max(10.0, 5.0);
            Assert.Equal(10.0, result);
        }

        [Fact]
        public void Max_SecondGreater_ReturnsSecond()
        {
            var result = _ops.Max(5.0, 10.0);
            Assert.Equal(10.0, result);
        }

        [Fact]
        public void Min_FirstLess_ReturnsFirst()
        {
            var result = _ops.Min(3.0, 8.0);
            Assert.Equal(3.0, result);
        }

        [Fact]
        public void Min_SecondLess_ReturnsSecond()
        {
            var result = _ops.Min(8.0, 3.0);
            Assert.Equal(3.0, result);
        }

        [Fact]
        public void ConvertFromDouble_ConvertsCorrectly()
        {
            var result = _ops.ConvertFromDouble(42.5);
            Assert.Equal(42.5, result);
        }

        [Fact]
        public void ConvertToDouble_ConvertsCorrectly()
        {
            var result = _ops.ConvertToDouble(42.5);
            Assert.Equal(42.5, result);
        }

        [Fact]
        public void IsNaN_WithNaN_ReturnsTrue()
        {
            var result = _ops.IsNaN(double.NaN);
            Assert.True(result);
        }

        [Fact]
        public void IsNaN_WithValidNumber_ReturnsFalse()
        {
            var result = _ops.IsNaN(5.0);
            Assert.False(result);
        }

        [Fact]
        public void IsInfinity_WithInfinity_ReturnsTrue()
        {
            var result = _ops.IsInfinity(double.PositiveInfinity);
            Assert.True(result);
        }

        [Fact]
        public void IsInfinity_WithValidNumber_ReturnsFalse()
        {
            var result = _ops.IsInfinity(5.0);
            Assert.False(result);
        }

        [Fact]
        public void Clamp_ValueBelowMin_ReturnsMin()
        {
            var result = _ops.Clamp(2.0, 5.0, 10.0);
            Assert.Equal(5.0, result);
        }

        [Fact]
        public void Clamp_ValueAboveMax_ReturnsMax()
        {
            var result = _ops.Clamp(15.0, 5.0, 10.0);
            Assert.Equal(10.0, result);
        }

        [Fact]
        public void Clamp_ValueWithinRange_ReturnsValue()
        {
            var result = _ops.Clamp(7.0, 5.0, 10.0);
            Assert.Equal(7.0, result);
        }
    }
}
