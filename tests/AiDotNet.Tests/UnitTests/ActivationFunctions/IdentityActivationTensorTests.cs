using System;
using AiDotNet.ActivationFunctions;
using Xunit;

namespace AiDotNetTests.UnitTests.ActivationFunctions
{
    /// <summary>
    /// Unit tests for the new <see cref="IdentityActivation{T}.Activate(Tensor{T})"/> tensor overload
    /// added in this PR.  The key contract is that the method returns the *same* reference that was
    /// passed in — no allocation, no value mutation, tape chain preserved.
    /// </summary>
    public class IdentityActivationTensorTests
    {
        // ──────────────────────────────────────────────────────────────────────
        // Reference-identity: the tensor overload must return the exact same
        // object so that autodiff tape chains are preserved and no heap pressure
        // is added during the forward pass.
        // ──────────────────────────────────────────────────────────────────────

        [Fact]
        public void Activate_Tensor_ReturnsSameReference_ForDouble()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([3]);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.True(ReferenceEquals(input, result),
                "Activate(Tensor<T>) must return the exact same tensor reference — no allocation allowed.");
        }

        [Fact]
        public void Activate_Tensor_ReturnsSameReference_ForFloat()
        {
            // Arrange
            var activation = new IdentityActivation<float>();
            var input = new Tensor<float>([4]);
            input[0] = -1f; input[1] = 0f; input[2] = 1f; input[3] = float.MaxValue;

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.True(ReferenceEquals(input, result),
                "Activate(Tensor<T>) must return the exact same tensor reference for float.");
        }

        // ──────────────────────────────────────────────────────────────────────
        // Value preservation: even though we're returning the same reference the
        // values must of course still be correct (i.e. unchanged).
        // ──────────────────────────────────────────────────────────────────────

        [Fact]
        public void Activate_Tensor_ValuesAreUnchanged()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var data = new double[] { -5.0, -1.0, 0.0, 1.0, 5.0, double.MaxValue, double.MinValue };
            var input = new Tensor<double>(data, [data.Length]);

            // Act
            var result = activation.Activate(input);

            // Assert — every element is bitwise-identical
            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], result.GetFlat(i),
                    $"Element {i} was modified — Identity must not change values.");
            }
        }

        [Fact]
        public void Activate_Tensor_WithAllZeros_ReturnsUnchanged()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([5]);
            // Default-filled tensor should be all zeros already

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.True(ReferenceEquals(input, result));
            for (int i = 0; i < 5; i++)
                Assert.Equal(0.0, result.GetFlat(i));
        }

        [Fact]
        public void Activate_Tensor_WithNegativeValues_ReturnsUnchanged()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([-10.0, -3.14, -0.001], [3]);

            // Act
            var result = activation.Activate(input);

            // Assert — same reference and same negative values
            Assert.True(ReferenceEquals(input, result));
            Assert.Equal(-10.0, result.GetFlat(0), 15);
            Assert.Equal(-3.14, result.GetFlat(1), 15);
            Assert.Equal(-0.001, result.GetFlat(2), 15);
        }

        // ──────────────────────────────────────────────────────────────────────
        // Shape preservation: the returned tensor must keep the same rank and
        // dimension sizes.
        // ──────────────────────────────────────────────────────────────────────

        [Fact]
        public void Activate_Tensor_PreservesShape_OneDimensional()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([7]);

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.Equal(1, result.Rank);
            Assert.Equal(7, result.Shape[0]);
        }

        [Fact]
        public void Activate_Tensor_PreservesShape_TwoDimensional()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([3, 4]);
            for (int i = 0; i < input.Length; i++)
                input.SetFlat(i, (double)i);

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.True(ReferenceEquals(input, result));
            Assert.Equal(2, result.Rank);
            Assert.Equal(3, result.Shape[0]);
            Assert.Equal(4, result.Shape[1]);
        }

        [Fact]
        public void Activate_Tensor_ReturnsSameReference_MultiDimensional()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([2, 3, 4]); // 3D tensor
            for (int i = 0; i < input.Length; i++)
                input.SetFlat(i, i * 0.5);

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.True(ReferenceEquals(input, result),
                "3D tensor: Activate must return the same reference.");
            Assert.Equal(3, result.Rank);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(3, result.Shape[1]);
            Assert.Equal(4, result.Shape[2]);
        }

        // ──────────────────────────────────────────────────────────────────────
        // Regression: calling Activate twice on the same tensor must not
        // accumulate side-effects (idempotency).
        // ──────────────────────────────────────────────────────────────────────

        [Fact]
        public void Activate_Tensor_CalledTwice_StillReturnsSameReference()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([3]);
            input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;

            // Act
            var result1 = activation.Activate(input);
            var result2 = activation.Activate(result1);

            // Assert — all three variables should be the exact same object
            Assert.True(ReferenceEquals(input, result1));
            Assert.True(ReferenceEquals(input, result2));
        }

        // ──────────────────────────────────────────────────────────────────────
        // Single-element edge case
        // ──────────────────────────────────────────────────────────────────────

        [Fact]
        public void Activate_Tensor_SingleElement_ReturnsSameReference()
        {
            // Arrange
            var activation = new IdentityActivation<double>();
            var input = new Tensor<double>([1]);
            input[0] = 42.0;

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.True(ReferenceEquals(input, result));
            Assert.Equal(42.0, result.GetFlat(0));
        }
    }
}