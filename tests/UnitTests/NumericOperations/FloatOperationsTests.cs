using System;
using AiDotNet.NumericOperations;
using Xunit;

namespace AiDotNetTests.UnitTests.NumericOperations
{
    public class FloatOperationsTests
    {
        private readonly FloatOperations _ops;

        public FloatOperationsTests()
        {
            _ops = new FloatOperations();
        }

        [Fact]
        public void Zero_ReturnsZero()
        {
            Assert.Equal(0.0f, _ops.Zero);
        }

        [Fact]
        public void One_ReturnsOne()
        {
            Assert.Equal(1.0f, _ops.One);
        }

        [Fact]
        public void Add_TwoNumbers_ReturnsCorrectSum()
        {
            var result = _ops.Add(3.5f, 2.5f);
            Assert.Equal(6.0f, result, 5);
        }

        [Fact]
        public void Subtract_TwoNumbers_ReturnsCorrectDifference()
        {
            var result = _ops.Subtract(10.0f, 3.5f);
            Assert.Equal(6.5f, result, 5);
        }

        [Fact]
        public void Multiply_TwoNumbers_ReturnsCorrectProduct()
        {
            var result = _ops.Multiply(4.0f, 2.5f);
            Assert.Equal(10.0f, result, 5);
        }

        [Fact]
        public void Divide_TwoNumbers_ReturnsCorrectQuotient()
        {
            var result = _ops.Divide(15.0f, 3.0f);
            Assert.Equal(5.0f, result, 5);
        }

        [Fact]
        public void Divide_ByZero_ReturnsInfinity()
        {
            var result = _ops.Divide(10.0f, 0.0f);
            Assert.True(float.IsInfinity(result));
        }

        [Fact]
        public void Negate_PositiveNumber_ReturnsNegative()
        {
            var result = _ops.Negate(5.0f);
            Assert.Equal(-5.0f, result, 5);
        }

        [Fact]
        public void Abs_NegativeNumber_ReturnsPositiveValue()
        {
            var result = _ops.Abs(-5.0f);
            Assert.Equal(5.0f, result, 5);
        }

        [Fact]
        public void Sqrt_PositiveNumber_ReturnsCorrectRoot()
        {
            var result = _ops.Sqrt(25.0f);
            Assert.Equal(5.0f, result, 5);
        }

        [Fact]
        public void Square_Number_ReturnsCorrectSquare()
        {
            var result = _ops.Square(5.0f);
            Assert.Equal(25.0f, result, 5);
        }

        [Fact]
        public void Exp_Zero_ReturnsOne()
        {
            var result = _ops.Exp(0.0f);
            Assert.Equal(1.0f, result, 5);
        }

        [Fact]
        public void Log_One_ReturnsZero()
        {
            var result = _ops.Log(1.0f);
            Assert.Equal(0.0f, result, 5);
        }

        [Fact]
        public void Power_BaseAndExponent_ReturnsCorrectResult()
        {
            var result = _ops.Power(2.0f, 3.0f);
            Assert.Equal(8.0f, result, 5);
        }

        [Fact]
        public void Sin_Zero_ReturnsZero()
        {
            var result = _ops.Sin(0.0f);
            Assert.Equal(0.0f, result, 5);
        }

        [Fact]
        public void Cos_Zero_ReturnsOne()
        {
            var result = _ops.Cos(0.0f);
            Assert.Equal(1.0f, result, 5);
        }

        [Fact]
        public void CompareTo_FirstLessThanSecond_ReturnsNegative()
        {
            var result = _ops.CompareTo(3.0f, 5.0f);
            Assert.True(result < 0);
        }

        [Fact]
        public void CompareTo_FirstGreaterThanSecond_ReturnsPositive()
        {
            var result = _ops.CompareTo(7.0f, 3.0f);
            Assert.True(result > 0);
        }

        [Fact]
        public void GreaterThan_FirstGreater_ReturnsTrue()
        {
            var result = _ops.GreaterThan(10.0f, 5.0f);
            Assert.True(result);
        }

        [Fact]
        public void LessThan_FirstLess_ReturnsTrue()
        {
            var result = _ops.LessThan(3.0f, 7.0f);
            Assert.True(result);
        }

        [Fact]
        public void Max_FirstGreater_ReturnsFirst()
        {
            var result = _ops.Max(10.0f, 5.0f);
            Assert.Equal(10.0f, result);
        }

        [Fact]
        public void Min_SecondLess_ReturnsSecond()
        {
            var result = _ops.Min(8.0f, 3.0f);
            Assert.Equal(3.0f, result);
        }

        [Fact]
        public void ConvertFromDouble_ConvertsCorrectly()
        {
            var result = _ops.ConvertFromDouble(42.5);
            Assert.Equal(42.5f, result, 5);
        }

        [Fact]
        public void ConvertToDouble_ConvertsCorrectly()
        {
            var result = _ops.ConvertToDouble(42.5f);
            Assert.Equal(42.5, result, 5);
        }

        [Fact]
        public void IsNaN_WithNaN_ReturnsTrue()
        {
            var result = _ops.IsNaN(float.NaN);
            Assert.True(result);
        }

        [Fact]
        public void IsNaN_WithValidNumber_ReturnsFalse()
        {
            var result = _ops.IsNaN(5.0f);
            Assert.False(result);
        }

        [Fact]
        public void Clamp_ValueBelowMin_ReturnsMin()
        {
            var result = _ops.Clamp(2.0f, 5.0f, 10.0f);
            Assert.Equal(5.0f, result);
        }

        [Fact]
        public void Clamp_ValueAboveMax_ReturnsMax()
        {
            var result = _ops.Clamp(15.0f, 5.0f, 10.0f);
            Assert.Equal(10.0f, result);
        }

        [Fact]
        public void Clamp_ValueWithinRange_ReturnsValue()
        {
            var result = _ops.Clamp(7.0f, 5.0f, 10.0f);
            Assert.Equal(7.0f, result);
        }
    }
}
