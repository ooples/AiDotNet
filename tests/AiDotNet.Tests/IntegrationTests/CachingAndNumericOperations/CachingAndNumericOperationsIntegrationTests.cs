using AiDotNet.Caching;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NumericOperations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNetTests.IntegrationTests.CachingAndNumericOperations
{
    /// <summary>
    /// Comprehensive integration tests for Caching and NumericOperations utilities.
    /// These tests validate caching behavior, cache key generation, and all numeric type operations.
    /// </summary>
    public class CachingAndNumericOperationsIntegrationTests
    {
        private const double DoubleTolerance = 1e-10;
        private const float FloatTolerance = 1e-6f;

        #region Caching Tests - DefaultModelCache

        [Fact]
        public void ModelCache_StoreAndRetrieve_ReturnsCorrectStepData()
        {
            // Arrange
            var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
            var key = "model_step_1";
            var stepData = new OptimizationStepData<double, Matrix<double>, Vector<double>>
            {
                Step = 1,
                Loss = 0.5
            };

            // Act
            cache.CacheStepData(key, stepData);
            var retrieved = cache.GetCachedStepData(key);

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(1, retrieved.Step);
            Assert.Equal(0.5, retrieved.Loss);
        }

        [Fact]
        public void ModelCache_RetrieveNonExistentKey_ReturnsNewInstance()
        {
            // Arrange
            var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
            var key = "nonexistent_key";

            // Act
            var retrieved = cache.GetCachedStepData(key);

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(0, retrieved.Step);
            Assert.Equal(0.0, retrieved.Loss);
        }

        [Fact]
        public void ModelCache_ClearCache_RemovesAllEntries()
        {
            // Arrange
            var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
            var stepData1 = new OptimizationStepData<double, Matrix<double>, Vector<double>> { Step = 1, Loss = 0.5 };
            var stepData2 = new OptimizationStepData<double, Matrix<double>, Vector<double>> { Step = 2, Loss = 0.3 };

            cache.CacheStepData("key1", stepData1);
            cache.CacheStepData("key2", stepData2);

            // Act
            cache.ClearCache();
            var retrieved1 = cache.GetCachedStepData("key1");
            var retrieved2 = cache.GetCachedStepData("key2");

            // Assert - Should return new instances with default values
            Assert.Equal(0, retrieved1.Step);
            Assert.Equal(0, retrieved2.Step);
        }

        [Fact]
        public void ModelCache_OverwriteExistingKey_UpdatesValue()
        {
            // Arrange
            var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
            var key = "model_step";
            var stepData1 = new OptimizationStepData<double, Matrix<double>, Vector<double>> { Step = 1, Loss = 0.5 };
            var stepData2 = new OptimizationStepData<double, Matrix<double>, Vector<double>> { Step = 2, Loss = 0.3 };

            // Act
            cache.CacheStepData(key, stepData1);
            cache.CacheStepData(key, stepData2);
            var retrieved = cache.GetCachedStepData(key);

            // Assert
            Assert.Equal(2, retrieved.Step);
            Assert.Equal(0.3, retrieved.Loss);
        }

        [Fact]
        public void ModelCache_StoreMultipleModels_AllRetrievableCorrectly()
        {
            // Arrange
            var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
            var models = new Dictionary<string, OptimizationStepData<double, Matrix<double>, Vector<double>>>();

            for (int i = 0; i < 10; i++)
            {
                var key = $"model_{i}";
                var stepData = new OptimizationStepData<double, Matrix<double>, Vector<double>>
                {
                    Step = i,
                    Loss = i * 0.1
                };
                models[key] = stepData;
                cache.CacheStepData(key, stepData);
            }

            // Act & Assert
            foreach (var kvp in models)
            {
                var retrieved = cache.GetCachedStepData(kvp.Key);
                Assert.Equal(kvp.Value.Step, retrieved.Step);
                Assert.Equal(kvp.Value.Loss, retrieved.Loss, precision: 10);
            }
        }

        [Fact]
        public void ModelCache_DifferentNumericTypes_WorksCorrectly()
        {
            // Test with float
            var floatCache = new DefaultModelCache<float, Matrix<float>, Vector<float>>();
            var floatData = new OptimizationStepData<float, Matrix<float>, Vector<float>> { Step = 1, Loss = 0.5f };
            floatCache.CacheStepData("float_key", floatData);
            var floatRetrieved = floatCache.GetCachedStepData("float_key");
            Assert.Equal(0.5f, floatRetrieved.Loss, precision: 6);

            // Test with decimal
            var decimalCache = new DefaultModelCache<decimal, Matrix<decimal>, Vector<decimal>>();
            var decimalData = new OptimizationStepData<decimal, Matrix<decimal>, Vector<decimal>> { Step = 1, Loss = 0.5m };
            decimalCache.CacheStepData("decimal_key", decimalData);
            var decimalRetrieved = decimalCache.GetCachedStepData("decimal_key");
            Assert.Equal(0.5m, decimalRetrieved.Loss);
        }

        #endregion

        #region Caching Tests - DefaultGradientCache

        [Fact]
        public void GradientCache_StoreAndRetrieve_ReturnsCorrectGradient()
        {
            // Arrange
            var cache = new DefaultGradientCache<double>();
            var key = "gradient_1";
            var gradient = new TestGradientModel<double>();

            // Act
            cache.CacheGradient(key, gradient);
            var retrieved = cache.GetCachedGradient(key);

            // Assert
            Assert.NotNull(retrieved);
            Assert.IsType<TestGradientModel<double>>(retrieved);
        }

        [Fact]
        public void GradientCache_RetrieveNonExistentKey_ReturnsNull()
        {
            // Arrange
            var cache = new DefaultGradientCache<double>();
            var key = "nonexistent_gradient";

            // Act
            var retrieved = cache.GetCachedGradient(key);

            // Assert
            Assert.Null(retrieved);
        }

        [Fact]
        public void GradientCache_ClearCache_RemovesAllGradients()
        {
            // Arrange
            var cache = new DefaultGradientCache<double>();
            cache.CacheGradient("grad1", new TestGradientModel<double>());
            cache.CacheGradient("grad2", new TestGradientModel<double>());

            // Act
            cache.ClearCache();
            var retrieved1 = cache.GetCachedGradient("grad1");
            var retrieved2 = cache.GetCachedGradient("grad2");

            // Assert
            Assert.Null(retrieved1);
            Assert.Null(retrieved2);
        }

        [Fact]
        public void GradientCache_OverwriteExistingKey_UpdatesGradient()
        {
            // Arrange
            var cache = new DefaultGradientCache<double>();
            var key = "gradient";
            var gradient1 = new TestGradientModel<double> { TestValue = 1.0 };
            var gradient2 = new TestGradientModel<double> { TestValue = 2.0 };

            // Act
            cache.CacheGradient(key, gradient1);
            cache.CacheGradient(key, gradient2);
            var retrieved = cache.GetCachedGradient(key) as TestGradientModel<double>;

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(2.0, retrieved.TestValue);
        }

        [Fact]
        public void GradientCache_StoreMultipleGradients_AllRetrievableCorrectly()
        {
            // Arrange
            var cache = new DefaultGradientCache<double>();
            var gradients = new Dictionary<string, TestGradientModel<double>>();

            for (int i = 0; i < 10; i++)
            {
                var key = $"gradient_{i}";
                var gradient = new TestGradientModel<double> { TestValue = i * 1.5 };
                gradients[key] = gradient;
                cache.CacheGradient(key, gradient);
            }

            // Act & Assert
            foreach (var kvp in gradients)
            {
                var retrieved = cache.GetCachedGradient(kvp.Key) as TestGradientModel<double>;
                Assert.NotNull(retrieved);
                Assert.Equal(kvp.Value.TestValue, retrieved.TestValue);
            }
        }

        [Fact]
        public async Task GradientCache_ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            var cache = new DefaultGradientCache<double>();
            var tasks = new List<Task>();

            // Act - Multiple threads writing to cache simultaneously
            for (int i = 0; i < 100; i++)
            {
                var index = i;
                tasks.Add(Task.Run(() =>
                {
                    var key = $"gradient_{index}";
                    var gradient = new TestGradientModel<double> { TestValue = index };
                    cache.CacheGradient(key, gradient);
                }));
            }

            await Task.WhenAll(tasks);

            // Assert - All values should be retrievable
            for (int i = 0; i < 100; i++)
            {
                var retrieved = cache.GetCachedGradient($"gradient_{i}") as TestGradientModel<double>;
                Assert.NotNull(retrieved);
                Assert.Equal(i, retrieved.TestValue);
            }
        }

        #endregion

        #region NumericOperations Tests - Byte

        [Fact]
        public void ByteOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new ByteOperations();

            // Act & Assert
            Assert.Equal((byte)15, ops.Add(10, 5));
            Assert.Equal((byte)5, ops.Subtract(10, 5));
            Assert.Equal((byte)50, ops.Multiply(10, 5));
            Assert.Equal((byte)2, ops.Divide(10, 5));
        }

        [Fact]
        public void ByteOperations_OverflowBehavior_WrapsAround()
        {
            // Arrange
            var ops = new ByteOperations();

            // Act & Assert - Addition overflow
            Assert.Equal((byte)4, ops.Add(250, 10)); // 260 wraps to 4

            // Multiplication overflow
            Assert.Equal((byte)0, ops.Multiply(16, 16)); // 256 wraps to 0

            // Subtraction underflow
            Assert.Equal((byte)246, ops.Subtract(10, 20)); // -10 wraps to 246
        }

        [Fact]
        public void ByteOperations_ComparisonOperations_WorkCorrectly()
        {
            // Arrange
            var ops = new ByteOperations();

            // Act & Assert
            Assert.True(ops.GreaterThan(10, 5));
            Assert.False(ops.GreaterThan(5, 10));
            Assert.True(ops.LessThan(5, 10));
            Assert.False(ops.LessThan(10, 5));
            Assert.True(ops.Equals(5, 5));
            Assert.False(ops.Equals(5, 10));
        }

        [Fact]
        public void ByteOperations_SpecialOperations_ProduceCorrectResults()
        {
            // Arrange
            var ops = new ByteOperations();

            // Act & Assert
            Assert.Equal((byte)0, ops.Zero);
            Assert.Equal((byte)1, ops.One);
            Assert.Equal((byte)3, ops.Sqrt(9));
            Assert.Equal((byte)4, ops.Sqrt(16));
            Assert.Equal((byte)25, ops.Abs(25));
            Assert.Equal((byte)100, ops.Square(10));
        }

        [Fact]
        public void ByteOperations_MinMaxValues_AreCorrect()
        {
            // Arrange
            var ops = new ByteOperations();

            // Act & Assert
            Assert.Equal(byte.MinValue, ops.MinValue); // 0
            Assert.Equal(byte.MaxValue, ops.MaxValue); // 255
        }

        #endregion

        #region NumericOperations Tests - SByte

        [Fact]
        public void SByteOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new SByteOperations();

            // Act & Assert
            Assert.Equal((sbyte)15, ops.Add(10, 5));
            Assert.Equal((sbyte)5, ops.Subtract(10, 5));
            Assert.Equal((sbyte)50, ops.Multiply(10, 5));
            Assert.Equal((sbyte)2, ops.Divide(10, 5));
            Assert.Equal((sbyte)-10, ops.Negate(10));
        }

        [Fact]
        public void SByteOperations_NegativeNumbers_WorkCorrectly()
        {
            // Arrange
            var ops = new SByteOperations();

            // Act & Assert
            Assert.Equal((sbyte)-5, ops.Add(-10, 5));
            Assert.Equal((sbyte)-15, ops.Subtract(-10, -5));
            Assert.Equal((sbyte)-50, ops.Multiply(-10, 5));
            Assert.Equal((sbyte)10, ops.Abs(-10));
        }

        [Fact]
        public void SByteOperations_SignOrZero_ReturnsCorrectSign()
        {
            // Arrange
            var ops = new SByteOperations();

            // Act & Assert
            Assert.Equal((sbyte)1, ops.SignOrZero(42));
            Assert.Equal((sbyte)-1, ops.SignOrZero(-42));
            Assert.Equal((sbyte)0, ops.SignOrZero(0));
        }

        #endregion

        #region NumericOperations Tests - Int16/Short

        [Fact]
        public void ShortOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new ShortOperations();

            // Act & Assert
            Assert.Equal((short)150, ops.Add(100, 50));
            Assert.Equal((short)50, ops.Subtract(100, 50));
            Assert.Equal((short)5000, ops.Multiply(100, 50));
            Assert.Equal((short)2, ops.Divide(100, 50));
        }

        [Fact]
        public void ShortOperations_ComparisonOperations_WorkCorrectly()
        {
            // Arrange
            var ops = new ShortOperations();

            // Act & Assert
            Assert.True(ops.GreaterThan(1000, 500));
            Assert.True(ops.LessThan(500, 1000));
            Assert.True(ops.GreaterThanOrEquals(1000, 1000));
            Assert.True(ops.LessThanOrEquals(500, 500));
        }

        [Fact]
        public void ShortOperations_MinMaxValues_AreCorrect()
        {
            // Arrange
            var ops = new ShortOperations();

            // Act & Assert
            Assert.Equal(short.MinValue, ops.MinValue); // -32768
            Assert.Equal(short.MaxValue, ops.MaxValue); // 32767
        }

        #endregion

        #region NumericOperations Tests - UInt16

        [Fact]
        public void UInt16Operations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new UInt16Operations();

            // Act & Assert
            Assert.Equal((ushort)150, ops.Add(100, 50));
            Assert.Equal((ushort)50, ops.Subtract(100, 50));
            Assert.Equal((ushort)5000, ops.Multiply(100, 50));
            Assert.Equal((ushort)2, ops.Divide(100, 50));
        }

        [Fact]
        public void UInt16Operations_MinMaxValues_AreCorrect()
        {
            // Arrange
            var ops = new UInt16Operations();

            // Act & Assert
            Assert.Equal(ushort.MinValue, ops.MinValue); // 0
            Assert.Equal(ushort.MaxValue, ops.MaxValue); // 65535
        }

        #endregion

        #region NumericOperations Tests - Int32

        [Fact]
        public void Int32Operations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new Int32Operations();

            // Act & Assert
            Assert.Equal(15000, ops.Add(10000, 5000));
            Assert.Equal(5000, ops.Subtract(10000, 5000));
            Assert.Equal(50000000, ops.Multiply(10000, 5000));
            Assert.Equal(2, ops.Divide(10000, 5000));
        }

        [Fact]
        public void Int32Operations_MathematicalFunctions_ProduceCorrectResults()
        {
            // Arrange
            var ops = new Int32Operations();

            // Act & Assert
            Assert.Equal(4, ops.Sqrt(16));
            Assert.Equal(100, ops.Square(10));
            Assert.Equal(10, ops.Abs(-10));
            Assert.Equal(8, ops.Power(2, 3));
        }

        [Fact]
        public void Int32Operations_ConversionFunctions_WorkCorrectly()
        {
            // Arrange
            var ops = new Int32Operations();

            // Act & Assert
            Assert.Equal(3, ops.FromDouble(3.7));
            Assert.Equal(42, ops.ToInt32(42));
            Assert.Equal(5, ops.Round(5));
        }

        [Fact]
        public void Int32Operations_SpecialValueChecks_ReturnCorrectResults()
        {
            // Arrange
            var ops = new Int32Operations();

            // Act & Assert
            Assert.False(ops.IsNaN(42)); // Integers can't be NaN
            Assert.False(ops.IsInfinity(42)); // Integers can't be infinity
        }

        #endregion

        #region NumericOperations Tests - UInt32

        [Fact]
        public void UInt32Operations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new UInt32Operations();

            // Act & Assert
            Assert.Equal((uint)15000, ops.Add(10000, 5000));
            Assert.Equal((uint)5000, ops.Subtract(10000, 5000));
            Assert.Equal((uint)50000000, ops.Multiply(10000, 5000));
            Assert.Equal((uint)2, ops.Divide(10000, 5000));
        }

        [Fact]
        public void UInt32Operations_LargeValues_WorkCorrectly()
        {
            // Arrange
            var ops = new UInt32Operations();

            // Act & Assert
            Assert.Equal((uint)4000000000, ops.Add(3000000000, 1000000000));
            Assert.True(ops.GreaterThan(3000000000, 2000000000));
        }

        #endregion

        #region NumericOperations Tests - UInt (alias for UInt32)

        [Fact]
        public void UIntOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new UIntOperations();

            // Act & Assert
            Assert.Equal((uint)150, ops.Add(100, 50));
            Assert.Equal((uint)50, ops.Subtract(100, 50));
            Assert.Equal((uint)5000, ops.Multiply(100, 50));
            Assert.Equal((uint)2, ops.Divide(100, 50));
        }

        #endregion

        #region NumericOperations Tests - Int64/Long

        [Fact]
        public void Int64Operations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new Int64Operations();

            // Act & Assert
            Assert.Equal(15000000000L, ops.Add(10000000000L, 5000000000L));
            Assert.Equal(5000000000L, ops.Subtract(10000000000L, 5000000000L));
            Assert.Equal(50000000000000000L, ops.Multiply(10000000L, 5000000000L));
            Assert.Equal(2L, ops.Divide(10000000000L, 5000000000L));
        }

        [Fact]
        public void Int64Operations_LargeValues_WorkCorrectly()
        {
            // Arrange
            var ops = new Int64Operations();
            long largeValue = 9223372036854775000L; // Near max value

            // Act & Assert
            Assert.True(ops.GreaterThan(largeValue, 1000000L));
            Assert.Equal(largeValue, ops.Abs(largeValue));
        }

        [Fact]
        public void Int64Operations_MinMaxValues_AreCorrect()
        {
            // Arrange
            var ops = new Int64Operations();

            // Act & Assert
            Assert.Equal(long.MinValue, ops.MinValue);
            Assert.Equal(long.MaxValue, ops.MaxValue);
        }

        #endregion

        #region NumericOperations Tests - UInt64

        [Fact]
        public void UInt64Operations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new UInt64Operations();

            // Act & Assert
            Assert.Equal((ulong)15000000000, ops.Add(10000000000, 5000000000));
            Assert.Equal((ulong)5000000000, ops.Subtract(10000000000, 5000000000));
            Assert.Equal((ulong)50000000000000000, ops.Multiply(10000000, 5000000000));
            Assert.Equal((ulong)2, ops.Divide(10000000000, 5000000000));
        }

        [Fact]
        public void UInt64Operations_VeryLargeValues_WorkCorrectly()
        {
            // Arrange
            var ops = new UInt64Operations();
            ulong veryLarge = 18446744073709551000UL; // Near max value

            // Act & Assert
            Assert.True(ops.GreaterThan(veryLarge, 1000000UL));
            Assert.Equal(veryLarge, ops.Abs(veryLarge));
        }

        #endregion

        #region NumericOperations Tests - Float

        [Fact]
        public void FloatOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new FloatOperations();

            // Act & Assert
            Assert.Equal(15.5f, ops.Add(10.0f, 5.5f), precision: 6);
            Assert.Equal(4.5f, ops.Subtract(10.0f, 5.5f), precision: 6);
            Assert.Equal(55.0f, ops.Multiply(10.0f, 5.5f), precision: 6);
            Assert.Equal(2.0f, ops.Divide(10.0f, 5.0f), precision: 6);
        }

        [Fact]
        public void FloatOperations_MathematicalFunctions_ProduceCorrectResults()
        {
            // Arrange
            var ops = new FloatOperations();

            // Act & Assert
            Assert.Equal(4.0f, ops.Sqrt(16.0f), precision: 6);
            Assert.Equal(100.0f, ops.Square(10.0f), precision: 6);
            Assert.Equal(10.5f, ops.Abs(-10.5f), precision: 6);
            Assert.Equal(8.0f, ops.Power(2.0f, 3.0f), precision: 6);
            Assert.Equal((float)Math.E, ops.Exp(1.0f), precision: 5);
            Assert.Equal(0.0f, ops.Log(1.0f), precision: 6);
        }

        [Fact]
        public void FloatOperations_SpecialValues_HandledCorrectly()
        {
            // Arrange
            var ops = new FloatOperations();

            // Act & Assert
            Assert.True(ops.IsNaN(float.NaN));
            Assert.False(ops.IsNaN(42.0f));
            Assert.True(ops.IsInfinity(float.PositiveInfinity));
            Assert.True(ops.IsInfinity(float.NegativeInfinity));
            Assert.False(ops.IsInfinity(42.0f));
        }

        [Fact]
        public void FloatOperations_SignOrZero_ReturnsCorrectSign()
        {
            // Arrange
            var ops = new FloatOperations();

            // Act & Assert
            Assert.Equal(1.0f, ops.SignOrZero(42.5f));
            Assert.Equal(-1.0f, ops.SignOrZero(-42.5f));
            Assert.Equal(0.0f, ops.SignOrZero(0.0f));
        }

        [Fact]
        public void FloatOperations_Rounding_WorksCorrectly()
        {
            // Arrange
            var ops = new FloatOperations();

            // Act & Assert
            Assert.Equal(4.0f, ops.Round(3.7f));
            Assert.Equal(3.0f, ops.Round(3.2f));
            Assert.Equal(4, ops.ToInt32(3.7f));
        }

        #endregion

        #region NumericOperations Tests - Double

        [Fact]
        public void DoubleOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new DoubleOperations();

            // Act & Assert
            Assert.Equal(15.5, ops.Add(10.0, 5.5), precision: 10);
            Assert.Equal(4.5, ops.Subtract(10.0, 5.5), precision: 10);
            Assert.Equal(55.0, ops.Multiply(10.0, 5.5), precision: 10);
            Assert.Equal(2.0, ops.Divide(10.0, 5.0), precision: 10);
        }

        [Fact]
        public void DoubleOperations_MathematicalFunctions_ProduceCorrectResults()
        {
            // Arrange
            var ops = new DoubleOperations();

            // Act & Assert
            Assert.Equal(4.0, ops.Sqrt(16.0), precision: 10);
            Assert.Equal(100.0, ops.Square(10.0), precision: 10);
            Assert.Equal(10.5, ops.Abs(-10.5), precision: 10);
            Assert.Equal(8.0, ops.Power(2.0, 3.0), precision: 10);
            Assert.Equal(Math.E, ops.Exp(1.0), precision: 10);
            Assert.Equal(0.0, ops.Log(1.0), precision: 10);
            Assert.Equal(1.0, ops.Log(Math.E), precision: 10);
        }

        [Fact]
        public void DoubleOperations_HighPrecisionCalculations_MaintainAccuracy()
        {
            // Arrange
            var ops = new DoubleOperations();

            // Act
            double result1 = ops.Add(0.1, 0.2);
            double result2 = ops.Multiply(Math.PI, 2.0);
            double result3 = ops.Divide(1.0, 3.0);

            // Assert
            Assert.Equal(0.3, result1, precision: 10);
            Assert.Equal(2.0 * Math.PI, result2, precision: 10);
            Assert.Equal(1.0 / 3.0, result3, precision: 10);
        }

        [Fact]
        public void DoubleOperations_ComparisonOperations_WorkCorrectly()
        {
            // Arrange
            var ops = new DoubleOperations();

            // Act & Assert
            Assert.True(ops.GreaterThan(10.5, 5.5));
            Assert.True(ops.LessThan(5.5, 10.5));
            Assert.True(ops.GreaterThanOrEquals(10.5, 10.5));
            Assert.True(ops.LessThanOrEquals(5.5, 5.5));
            Assert.True(ops.Equals(5.5, 5.5));
        }

        [Fact]
        public void DoubleOperations_SpecialValues_HandledCorrectly()
        {
            // Arrange
            var ops = new DoubleOperations();

            // Act & Assert
            Assert.True(ops.IsNaN(double.NaN));
            Assert.False(ops.IsNaN(42.0));
            Assert.True(ops.IsInfinity(double.PositiveInfinity));
            Assert.True(ops.IsInfinity(double.NegativeInfinity));
            Assert.False(ops.IsInfinity(42.0));
        }

        #endregion

        #region NumericOperations Tests - Decimal

        [Fact]
        public void DecimalOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new DecimalOperations();

            // Act & Assert
            Assert.Equal(15.5m, ops.Add(10.0m, 5.5m));
            Assert.Equal(4.5m, ops.Subtract(10.0m, 5.5m));
            Assert.Equal(55.0m, ops.Multiply(10.0m, 5.5m));
            Assert.Equal(2.0m, ops.Divide(10.0m, 5.0m));
        }

        [Fact]
        public void DecimalOperations_HighPrecisionArithmetic_MaintainsExactness()
        {
            // Arrange
            var ops = new DecimalOperations();

            // Act - Decimal should handle this exactly unlike double/float
            decimal result1 = ops.Add(0.1m, 0.2m);
            decimal result2 = ops.Divide(1.0m, 3.0m);
            decimal result3 = ops.Multiply(result2, 3.0m);

            // Assert
            Assert.Equal(0.3m, result1);
            Assert.Equal(0.3333333333333333333333333333m, result2);
            Assert.Equal(0.9999999999999999999999999999m, result3);
        }

        [Fact]
        public void DecimalOperations_FinancialCalculations_AreAccurate()
        {
            // Arrange
            var ops = new DecimalOperations();
            decimal price = 19.99m;
            decimal taxRate = 0.08m;

            // Act
            decimal tax = ops.Multiply(price, taxRate);
            decimal total = ops.Add(price, tax);

            // Assert
            Assert.Equal(1.5992m, tax);
            Assert.Equal(21.5892m, total);
        }

        [Fact]
        public void DecimalOperations_ComparisonOperations_WorkCorrectly()
        {
            // Arrange
            var ops = new DecimalOperations();

            // Act & Assert
            Assert.True(ops.GreaterThan(10.5m, 5.5m));
            Assert.True(ops.LessThan(5.5m, 10.5m));
            Assert.True(ops.Equals(5.5m, 5.5m));
        }

        [Fact]
        public void DecimalOperations_SpecialValueChecks_ReturnCorrectResults()
        {
            // Arrange
            var ops = new DecimalOperations();

            // Act & Assert - Decimals can't be NaN or Infinity
            Assert.False(ops.IsNaN(42.5m));
            Assert.False(ops.IsInfinity(42.5m));
        }

        #endregion

        #region NumericOperations Tests - Complex

        [Fact]
        public void ComplexOperations_BasicArithmetic_ProducesCorrectResults()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var a = new Complex<double>(3.0, 4.0); // 3 + 4i
            var b = new Complex<double>(1.0, 2.0); // 1 + 2i

            // Act
            var sum = ops.Add(a, b);
            var diff = ops.Subtract(a, b);
            var product = ops.Multiply(a, b);

            // Assert
            Assert.Equal(4.0, sum.Real, precision: 10);
            Assert.Equal(6.0, sum.Imaginary, precision: 10);
            Assert.Equal(2.0, diff.Real, precision: 10);
            Assert.Equal(2.0, diff.Imaginary, precision: 10);
            // (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
            Assert.Equal(-5.0, product.Real, precision: 10);
            Assert.Equal(10.0, product.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_ComplexDivision_ProducesCorrectResult()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var a = new Complex<double>(3.0, 2.0); // 3 + 2i
            var b = new Complex<double>(1.0, 1.0); // 1 + 1i

            // Act
            var quotient = ops.Divide(a, b);

            // Assert
            // (3 + 2i) / (1 + 1i) = (3 + 2i)(1 - 1i) / ((1 + 1i)(1 - 1i))
            // = (3 - 3i + 2i - 2i²) / (1 - i²) = (3 - i + 2) / 2 = (5 - i) / 2 = 2.5 - 0.5i
            Assert.Equal(2.5, quotient.Real, precision: 10);
            Assert.Equal(-0.5, quotient.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_Magnitude_CalculatedCorrectly()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var complex = new Complex<double>(3.0, 4.0); // 3 + 4i

            // Act
            var absValue = ops.Abs(complex);

            // Assert
            // Magnitude of 3 + 4i is sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
            Assert.Equal(5.0, absValue.Real, precision: 10);
            Assert.Equal(0.0, absValue.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_SquareRoot_ProducesCorrectResult()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var complex = new Complex<double>(0.0, 4.0); // 4i (pure imaginary)

            // Act
            var sqrt = ops.Sqrt(complex);

            // Assert
            // sqrt(4i) should be approximately sqrt(2) + sqrt(2)i
            Assert.Equal(Math.Sqrt(2.0), sqrt.Real, precision: 8);
            Assert.Equal(Math.Sqrt(2.0), sqrt.Imaginary, precision: 8);
        }

        [Fact]
        public void ComplexOperations_ExponentialFunction_ProducesCorrectResult()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var complex = new Complex<double>(0.0, Math.PI); // πi

            // Act
            var expValue = ops.Exp(complex);

            // Assert
            // e^(πi) = cos(π) + i*sin(π) = -1 + 0i (Euler's formula)
            Assert.Equal(-1.0, expValue.Real, precision: 10);
            Assert.Equal(0.0, expValue.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_NaturalLogarithm_ProducesCorrectResult()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var complex = new Complex<double>(Math.E, 0.0); // e + 0i

            // Act
            var logValue = ops.Log(complex);

            // Assert
            // ln(e) = 1 + 0i
            Assert.Equal(1.0, logValue.Real, precision: 10);
            Assert.Equal(0.0, logValue.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_Power_ProducesCorrectResult()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var baseValue = new Complex<double>(2.0, 0.0); // 2 + 0i
            var exponent = new Complex<double>(3.0, 0.0); // 3 + 0i

            // Act
            var result = ops.Power(baseValue, exponent);

            // Assert
            // 2³ = 8
            Assert.Equal(8.0, result.Real, precision: 10);
            Assert.Equal(0.0, result.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_ComparisonByMagnitude_WorksCorrectly()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var a = new Complex<double>(3.0, 4.0); // Magnitude 5
            var b = new Complex<double>(1.0, 2.0); // Magnitude sqrt(5) ≈ 2.236

            // Act & Assert
            Assert.True(ops.GreaterThan(a, b));
            Assert.True(ops.LessThan(b, a));
            Assert.True(ops.GreaterThanOrEquals(a, a));
            Assert.True(ops.LessThanOrEquals(b, b));
        }

        [Fact]
        public void ComplexOperations_Equality_ChecksBothComponents()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var a = new Complex<double>(3.0, 4.0);
            var b = new Complex<double>(3.0, 4.0);
            var c = new Complex<double>(3.0, 5.0);

            // Act & Assert
            Assert.True(ops.Equals(a, b));
            Assert.False(ops.Equals(a, c));
        }

        [Fact]
        public void ComplexOperations_ZeroAndOne_HaveCorrectValues()
        {
            // Arrange
            var ops = new ComplexOperations<double>();

            // Act
            var zero = ops.Zero;
            var one = ops.One;

            // Assert
            Assert.Equal(0.0, zero.Real);
            Assert.Equal(0.0, zero.Imaginary);
            Assert.Equal(1.0, one.Real);
            Assert.Equal(0.0, one.Imaginary);
        }

        [Fact]
        public void ComplexOperations_Negate_ReversesBothComponents()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var complex = new Complex<double>(3.0, 4.0);

            // Act
            var negated = ops.Negate(complex);

            // Assert
            Assert.Equal(-3.0, negated.Real);
            Assert.Equal(-4.0, negated.Imaginary);
        }

        [Fact]
        public void ComplexOperations_Square_ProducesCorrectResult()
        {
            // Arrange
            var ops = new ComplexOperations<double>();
            var complex = new Complex<double>(3.0, 2.0); // 3 + 2i

            // Act
            var squared = ops.Square(complex);

            // Assert
            // (3 + 2i)² = 9 + 12i + 4i² = 9 + 12i - 4 = 5 + 12i
            Assert.Equal(5.0, squared.Real, precision: 10);
            Assert.Equal(12.0, squared.Imaginary, precision: 10);
        }

        [Fact]
        public void ComplexOperations_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var ops = new ComplexOperations<float>();
            var a = new Complex<float>(3.0f, 4.0f);
            var b = new Complex<float>(1.0f, 2.0f);

            // Act
            var sum = ops.Add(a, b);
            var product = ops.Multiply(a, b);

            // Assert
            Assert.Equal(4.0f, sum.Real, precision: 6);
            Assert.Equal(6.0f, sum.Imaginary, precision: 6);
            Assert.Equal(-5.0f, product.Real, precision: 6);
            Assert.Equal(10.0f, product.Imaginary, precision: 6);
        }

        #endregion

        #region Cross-Type Numeric Operations Tests

        [Fact]
        public void NumericOperations_TypeConversions_WorkCorrectly()
        {
            // Test conversions between types
            var doubleOps = new DoubleOperations();
            var floatOps = new FloatOperations();
            var intOps = new Int32Operations();

            // Double to Float
            float floatValue = floatOps.FromDouble(3.14159);
            Assert.Equal(3.14159f, floatValue, precision: 5);

            // Double to Int
            int intValue = intOps.FromDouble(3.7);
            Assert.Equal(3, intValue);

            // Int to Int (identity)
            int intValue2 = intOps.ToInt32(42);
            Assert.Equal(42, intValue2);
        }

        [Fact]
        public void NumericOperations_PrecisionComparison_FloatVsDoubleVsDecimal()
        {
            // Arrange
            var floatOps = new FloatOperations();
            var doubleOps = new DoubleOperations();
            var decimalOps = new DecimalOperations();

            // Act - Compute 1/3 with each type
            float floatResult = floatOps.Divide(1.0f, 3.0f);
            double doubleResult = doubleOps.Divide(1.0, 3.0);
            decimal decimalResult = decimalOps.Divide(1.0m, 3.0m);

            // Assert - Decimal should have highest precision
            Assert.Equal(0.333333f, floatResult, precision: 6);
            Assert.Equal(0.333333333333333, doubleResult, precision: 15);
            Assert.Equal(0.3333333333333333333333333333m, decimalResult);
        }

        [Fact]
        public void NumericOperations_AllTypesHaveConsistentInterface()
        {
            // Verify all types implement the same basic operations
            var byteOps = new ByteOperations();
            var sbyteOps = new SByteOperations();
            var shortOps = new ShortOperations();
            var ushortOps = new UInt16Operations();
            var intOps = new Int32Operations();
            var uintOps = new UInt32Operations();
            var longOps = new Int64Operations();
            var ulongOps = new UInt64Operations();
            var floatOps = new FloatOperations();
            var doubleOps = new DoubleOperations();
            var decimalOps = new DecimalOperations();

            // All should have Zero and One
            Assert.Equal((byte)0, byteOps.Zero);
            Assert.Equal((byte)1, byteOps.One);
            Assert.Equal(0, intOps.Zero);
            Assert.Equal(1, intOps.One);
            Assert.Equal(0.0, doubleOps.Zero);
            Assert.Equal(1.0, doubleOps.One);
        }

        #endregion

        #region Helper Classes

        /// <summary>
        /// Test implementation of IGradientModel for testing purposes.
        /// </summary>
        private class TestGradientModel<T> : IGradientModel<T>
        {
            public double TestValue { get; set; }

            public Vector<T> ComputeGradient(Vector<T> parameters)
            {
                throw new NotImplementedException();
            }

            public Vector<T> ComputeGradient(Vector<T> parameters, object context)
            {
                throw new NotImplementedException();
            }
        }

        #endregion
    }
}
