using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class EnumHelperTests
    {
        // Test enum for testing purposes
        private enum TestEnum
        {
            Value1,
            Value2,
            Value3,
            Value4,
            Value5
        }

        private enum SingleValueEnum
        {
            OnlyValue
        }

        private enum EmptyEnum
        {
        }

        [Fact]
        public void GetEnumValues_WithNoIgnore_ReturnsAllValues()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>();

            // Assert
            Assert.NotNull(result);
            Assert.Equal(5, result.Count);
            Assert.Contains(TestEnum.Value1, result);
            Assert.Contains(TestEnum.Value2, result);
            Assert.Contains(TestEnum.Value3, result);
            Assert.Contains(TestEnum.Value4, result);
            Assert.Contains(TestEnum.Value5, result);
        }

        [Fact]
        public void GetEnumValues_WithIgnoreName_ExcludesSpecifiedValue()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("Value3");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(TestEnum.Value3, result);
        }

        [Fact]
        public void GetEnumValues_WithActivationFunction_ReturnsValues()
        {
            // Act - exclude "ReLU" to test the exclude functionality
            var result = EnumHelper.GetEnumValues<ActivationFunction>("ReLU");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(ActivationFunction.ReLU, result);
        }

        [Fact]
        public void GetEnumValues_WithPoolingType_ReturnsValues()
        {
            // Act
            var result = EnumHelper.GetEnumValues<PoolingType>();

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void GetEnumValues_WithNullIgnoreName_ReturnsAllValues()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>(null);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(5, result.Count);
        }

        [Fact]
        public void GetEnumValues_WithEmptyStringIgnoreName_ReturnsAllValues()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("");

            // Assert
            Assert.NotNull(result);
            Assert.Equal(5, result.Count);
        }

        [Fact]
        public void GetEnumValues_WithNonExistentIgnoreName_ReturnsAllValues()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("NonExistentValue");

            // Assert
            Assert.NotNull(result);
            Assert.Equal(5, result.Count);
        }

        [Fact]
        public void GetEnumValues_WithSingleValueEnum_ReturnsCorrectly()
        {
            // Act
            var result = EnumHelper.GetEnumValues<SingleValueEnum>("OnlyValue");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(SingleValueEnum.OnlyValue, result);
        }

        [Fact]
        public void GetEnumValues_WithActivationFunctionIgnoreReLU_ExcludesReLU()
        {
            // Act
            var result = EnumHelper.GetEnumValues<ActivationFunction>("ReLU");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(ActivationFunction.ReLU, result);
        }

        [Fact]
        public void GetEnumValues_WithActivationFunctionIgnoreSigmoid_ExcludesSigmoid()
        {
            // Act
            var result = EnumHelper.GetEnumValues<ActivationFunction>("Sigmoid");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(ActivationFunction.Sigmoid, result);
        }

        [Fact]
        public void GetEnumValues_WithActivationFunctionIgnoreTanh_ExcludesTanh()
        {
            // Act
            var result = EnumHelper.GetEnumValues<ActivationFunction>("Tanh");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(ActivationFunction.Tanh, result);
        }

        [Fact]
        public void GetEnumValues_WithPoolingTypeIgnoreMax_ExcludesMax()
        {
            // Act
            var result = EnumHelper.GetEnumValues<PoolingType>("Max");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(PoolingType.Max, result);
        }

        [Fact]
        public void GetEnumValues_WithPoolingTypeIgnoreAverage_ExcludesAverage()
        {
            // Act
            var result = EnumHelper.GetEnumValues<PoolingType>("Average");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(PoolingType.Average, result);
        }

        [Fact]
        public void GetEnumValues_ReturnsListNotArray()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("Value1");

            // Assert
            Assert.IsAssignableFrom<System.Collections.Generic.List<TestEnum>>(result);
        }

        [Fact]
        public void GetEnumValues_WithCaseSensitiveIgnore_IsCaseSensitive()
        {
            // Act - trying with wrong case
            var result = EnumHelper.GetEnumValues<TestEnum>("value1");

            // Assert
            Assert.NotNull(result);
            // Since it's case sensitive, "value1" won't match "Value1", so Value1 should be included
            Assert.Contains(TestEnum.Value1, result);
        }

        [Fact]
        public void GetEnumValues_MultipleCallsWithSameEnum_ReturnsConsistentResults()
        {
            // Act
            var result1 = EnumHelper.GetEnumValues<TestEnum>("Value2");
            var result2 = EnumHelper.GetEnumValues<TestEnum>("Value2");

            // Assert
            Assert.Equal(result1.Count, result2.Count);
        }

        [Fact]
        public void GetEnumValues_WithDifferentIgnoreValues_ReturnsCorrectResults()
        {
            // Act
            var result1 = EnumHelper.GetEnumValues<TestEnum>("Value1");
            var result2 = EnumHelper.GetEnumValues<TestEnum>("Value2");

            // Assert
            Assert.NotNull(result1);
            Assert.NotNull(result2);
            Assert.DoesNotContain(TestEnum.Value1, result1);
            Assert.DoesNotContain(TestEnum.Value2, result2);
        }

        [Fact]
        public void GetEnumValues_WithTestEnum_DoesNotReturnIgnoredValue()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("Value4");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(TestEnum.Value4, result);
        }

        [Fact]
        public void GetEnumValues_ResultCanBeEnumerated()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("Value5");

            // Assert
            Assert.NotNull(result);
            foreach (var value in result)
            {
                Assert.IsType<TestEnum>(value);
            }
        }

        [Fact]
        public void GetEnumValues_WithActivationFunction_CanBeUsedInLinq()
        {
            // Act
            var result = EnumHelper.GetEnumValues<ActivationFunction>("None");
            var count = result.Count();

            // Assert
            Assert.True(count >= 0);
        }

        [Fact]
        public void GetEnumValues_ReturnsOnlyEnumValues()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("NonExistent");

            // Assert
            Assert.NotNull(result);
            Assert.All(result, item => Assert.True(Enum.IsDefined(typeof(TestEnum), item)));
        }

        [Fact]
        public void GetEnumValues_WithWhitespaceIgnoreName_TreatsAsNonMatching()
        {
            // Act
            var result = EnumHelper.GetEnumValues<TestEnum>("  Value1  ");

            // Assert
            Assert.NotNull(result);
            // Whitespace won't match, so Value1 should be included
            Assert.Contains(TestEnum.Value1, result);
        }

        [Fact]
        public void GetEnumValues_CalledTwiceWithSameParameters_ReturnsDifferentInstances()
        {
            // Act
            var result1 = EnumHelper.GetEnumValues<TestEnum>("Value1");
            var result2 = EnumHelper.GetEnumValues<TestEnum>("Value1");

            // Assert
            Assert.NotSame(result1, result2);
        }

        [Fact]
        public void GetEnumValues_WithActivationFunctionLeakyReLU_WorksCorrectly()
        {
            // Act
            var result = EnumHelper.GetEnumValues<ActivationFunction>("LeakyReLU");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(ActivationFunction.LeakyReLU, result);
        }

        [Fact]
        public void GetEnumValues_WithActivationFunctionSoftmax_WorksCorrectly()
        {
            // Act
            var result = EnumHelper.GetEnumValues<ActivationFunction>("Softmax");

            // Assert
            Assert.NotNull(result);
            Assert.DoesNotContain(ActivationFunction.Softmax, result);
        }
    }
}
