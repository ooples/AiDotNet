using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Integration tests to validate AiDotNet.Tensors 0.9.3 and AiDotNet ecosystem package upgrades.
/// Exercises core Tensor, Vector, Matrix, NumericOperations, and Engine paths
/// to catch any breaking changes from the 0.9.1 → 0.9.3 upgrade.
/// </summary>
public class PackageUpgradeValidationTests
{
    private const double Tolerance = 1e-10;
    private static readonly IEngine Engine = AiDotNetEngine.Current;

    #region Tensor Core Operations

    [Fact]
    public void Tensor_CreateAndIndex_RoundTrips()
    {
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        var tensor = new Tensor<double>(new[] { 2, 3 }, data);

        Assert.Equal(2, tensor.Rank);
        Assert.Equal(6, tensor.Length);
        Assert.Equal(1.0, tensor[0, 0]);
        Assert.Equal(4.0, tensor[1, 0]);
        Assert.Equal(6.0, tensor[1, 2]);
    }

    [Fact]
    public void Tensor_CreateDefault_FillsWithValue()
    {
        var tensor = Tensor<double>.CreateDefault(new[] { 3, 4 }, 7.5);

        Assert.Equal(12, tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.Equal(7.5, tensor[i]);
        }
    }

    [Fact]
    public void Tensor_1D_AutoReshape_PreservesRank()
    {
        var tensor = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++)
        {
            tensor[i] = i + 1.0;
        }

        Assert.Equal(1, tensor.Rank);
        Assert.Equal(5, tensor.Shape[0]);
        Assert.Equal(3.0, tensor[2]);
    }

    [Fact]
    public void Tensor_3D_ShapePreserved()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });

        Assert.Equal(3, tensor.Rank);
        Assert.Equal(24, tensor.Length);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(4, tensor.Shape[2]);
    }

    [Fact]
    public void Tensor_IndexCopy_RoundTrip()
    {
        var source = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++)
        {
            source[i] = (i + 1) * 10.0;
        }

        var target = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++)
        {
            target[i] = source[i];
        }

        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(source[i], target[i]);
        }
    }

    [Fact]
    public void Tensor_Fill_SetsAllElements()
    {
        var tensor = new Tensor<double>(new[] { 3, 3 });
        tensor.Fill(42.0);

        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.Equal(42.0, tensor[i]);
        }
    }

    #endregion

    #region Vector Operations

    [Fact]
    public void Vector_ArithmeticOperations_Correct()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        var sum = a + b;
        Assert.Equal(5.0, sum[0], Tolerance);
        Assert.Equal(7.0, sum[1], Tolerance);
        Assert.Equal(9.0, sum[2], Tolerance);

        var diff = b - a;
        Assert.Equal(3.0, diff[0], Tolerance);
        Assert.Equal(3.0, diff[1], Tolerance);
        Assert.Equal(3.0, diff[2], Tolerance);
    }

    [Fact]
    public void Vector_DotProduct_Correct()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Instance method DotProduct
        var dot = a.DotProduct(b);
        Assert.Equal(32.0, dot, Tolerance); // 1*4 + 2*5 + 3*6
    }

    [Fact]
    public void Vector_ToArray_RoundTrip()
    {
        var original = new double[] { 1.5, 2.5, 3.5 };
        var vector = new Vector<double>(original);
        var result = vector.ToArray();

        Assert.Equal(original.Length, result.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], result[i]);
        }
    }

    [Fact]
    public void Vector_Construction_FromLength()
    {
        var vector = new Vector<double>(10);

        Assert.Equal(10, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.Equal(0.0, vector[i]);
        }
    }

    [Fact]
    public void Vector_Indexer_SetAndGet()
    {
        var vector = new Vector<double>(3);
        vector[0] = 10.0;
        vector[1] = 20.0;
        vector[2] = 30.0;

        Assert.Equal(10.0, vector[0]);
        Assert.Equal(20.0, vector[1]);
        Assert.Equal(30.0, vector[2]);
    }

    #endregion

    #region Matrix Operations

    [Fact]
    public void Matrix_Construction_FromArray()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        Assert.Equal(3, matrix.Rows);
        Assert.Equal(2, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(6.0, matrix[2, 1]);
    }

    [Fact]
    public void Matrix_Transpose_CorrectDimensions()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        var transposed = matrix.Transpose();

        Assert.Equal(3, transposed.Rows);
        Assert.Equal(2, transposed.Columns);
        Assert.Equal(1.0, transposed[0, 0]);
        Assert.Equal(4.0, transposed[0, 1]);
        Assert.Equal(3.0, transposed[2, 0]);
        Assert.Equal(6.0, transposed[2, 1]);
    }

    [Fact]
    public void Matrix_RowsAndColumns_Correct()
    {
        var matrix = new Matrix<double>(4, 7);

        Assert.Equal(4, matrix.Rows);
        Assert.Equal(7, matrix.Columns);
    }

    [Fact]
    public void Matrix_Indexer_SetAndGet()
    {
        var matrix = new Matrix<double>(2, 2);
        matrix[0, 0] = 1.0;
        matrix[0, 1] = 2.0;
        matrix[1, 0] = 3.0;
        matrix[1, 1] = 4.0;

        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(2.0, matrix[0, 1]);
        Assert.Equal(3.0, matrix[1, 0]);
        Assert.Equal(4.0, matrix[1, 1]);
    }

    #endregion

    #region NumericOperations

    [Fact]
    public void NumericOperations_Double_BasicArithmetic()
    {
        var ops = MathHelper.GetNumericOperations<double>();

        Assert.Equal(5.0, ops.Add(2.0, 3.0));
        Assert.Equal(6.0, ops.Multiply(2.0, 3.0));
        Assert.Equal(1.0, ops.Subtract(3.0, 2.0));
        Assert.Equal(2.5, ops.Divide(5.0, 2.0));
    }

    [Fact]
    public void NumericOperations_Double_Conversions()
    {
        var ops = MathHelper.GetNumericOperations<double>();

        Assert.Equal(3.14, ops.FromDouble(3.14));
        Assert.Equal(3.14, ops.ToDouble(3.14));
        Assert.Equal(0.0, ops.Zero);
        Assert.Equal(1.0, ops.One);
    }

    [Fact]
    public void NumericOperations_Double_MathFunctions()
    {
        var ops = MathHelper.GetNumericOperations<double>();

        Assert.Equal(4.0, ops.Sqrt(16.0), Tolerance);
        Assert.Equal(Math.E, ops.Exp(1.0), Tolerance);
        Assert.Equal(0.0, ops.Log(1.0), Tolerance);
        Assert.Equal(8.0, ops.Power(2.0, 3.0), Tolerance);
    }

    [Fact]
    public void NumericOperations_Float_BasicArithmetic()
    {
        var ops = MathHelper.GetNumericOperations<float>();

        Assert.Equal(5.0f, ops.Add(2.0f, 3.0f));
        Assert.Equal(6.0f, ops.Multiply(2.0f, 3.0f));
        Assert.Equal(0.0f, ops.Zero);
        Assert.Equal(1.0f, ops.One);
    }

    [Fact]
    public void NumericOperations_Comparison()
    {
        var ops = MathHelper.GetNumericOperations<double>();

        Assert.True(ops.GreaterThan(5.0, 3.0));
        Assert.False(ops.GreaterThan(3.0, 5.0));
        Assert.True(ops.LessThan(3.0, 5.0));
        Assert.False(ops.LessThan(5.0, 3.0));
    }

    #endregion

    #region Engine Operations

    [Fact]
    public void Engine_TensorAdd_Correct()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 4.0, 5.0, 6.0 }));

        var result = Engine.TensorAdd(a, b);

        Assert.Equal(5.0, result[0], Tolerance);
        Assert.Equal(7.0, result[1], Tolerance);
        Assert.Equal(9.0, result[2], Tolerance);
    }

    [Fact]
    public void Engine_TensorSubtract_Correct()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 10.0, 20.0, 30.0 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));

        var result = Engine.TensorSubtract(a, b);

        Assert.Equal(9.0, result[0], Tolerance);
        Assert.Equal(18.0, result[1], Tolerance);
        Assert.Equal(27.0, result[2], Tolerance);
    }

    [Fact]
    public void Engine_TensorMultiply_ElementWise()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 2.0, 3.0, 4.0 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 5.0, 6.0, 7.0 }));

        var result = Engine.TensorMultiply(a, b);

        Assert.Equal(10.0, result[0], Tolerance);
        Assert.Equal(18.0, result[1], Tolerance);
        Assert.Equal(28.0, result[2], Tolerance);
    }

    [Fact]
    public void Engine_Sigmoid_ValueRange()
    {
        var input = new Tensor<double>(new[] { 5 }, new Vector<double>(new[] { -10.0, -1.0, 0.0, 1.0, 10.0 }));

        var result = Engine.Sigmoid(input);

        // Sigmoid output should be in (0, 1)
        for (int i = 0; i < result.Length; i++)
        {
            Assert.True(result[i] > 0.0, $"Sigmoid output at index {i} should be > 0");
            Assert.True(result[i] < 1.0, $"Sigmoid output at index {i} should be < 1");
        }

        // Sigmoid(0) = 0.5
        Assert.Equal(0.5, result[2], 1e-6);

        // Sigmoid should be monotonically increasing
        for (int i = 1; i < result.Length; i++)
        {
            Assert.True(result[i] > result[i - 1], $"Sigmoid should be monotonically increasing at index {i}");
        }
    }

    [Fact]
    public void Engine_VectorOperations_AddSubtractMultiply()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        var sum = Engine.Add(a, b);
        Assert.Equal(5.0, sum[0], Tolerance);
        Assert.Equal(7.0, sum[1], Tolerance);
        Assert.Equal(9.0, sum[2], Tolerance);

        var diff = Engine.Subtract(b, a);
        Assert.Equal(3.0, diff[0], Tolerance);
        Assert.Equal(3.0, diff[1], Tolerance);
        Assert.Equal(3.0, diff[2], Tolerance);

        var product = Engine.Multiply(a, b);
        Assert.Equal(4.0, product[0], Tolerance);  // 1*4
        Assert.Equal(10.0, product[1], Tolerance); // 2*5
        Assert.Equal(18.0, product[2], Tolerance); // 3*6
    }

    [Fact]
    public void Engine_DotProduct_Correct()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        var dot = Engine.DotProduct(a, b);
        Assert.Equal(32.0, dot, Tolerance); // 1*4 + 2*5 + 3*6
    }

    #endregion

    #region Neural Network Layer Smoke Tests

    [Fact]
    public void FullyConnectedLayer_ForwardBackward_RoundTrip()
    {
        var layer = new FullyConnectedLayer<double>(4, 3, (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));

        var output = layer.Forward(input);

        Assert.Equal(3, output.Length);

        // Verify weights are applied: output = W*input + bias, so at least one value should be non-zero
        // (random init with non-zero input guarantees this)
        bool hasNonZeroOutput = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) > 1e-15)
            {
                hasNonZeroOutput = true;
                break;
            }
        }
        Assert.True(hasNonZeroOutput, "FCL forward should produce non-zero output with non-zero input and random weights");

        // Backward should produce non-zero gradients when output gradient is non-zero
        var grad = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 0.1, 0.2, 0.3 }));
        var inputGrad = layer.Backward(grad);

        Assert.Equal(4, inputGrad.Length);

        bool hasNonZeroGrad = false;
        for (int i = 0; i < inputGrad.Length; i++)
        {
            if (Math.Abs(inputGrad[i]) > 1e-15)
            {
                hasNonZeroGrad = true;
                break;
            }
        }
        Assert.True(hasNonZeroGrad, "FCL backward should produce non-zero input gradients with non-zero output gradient");
    }

    [Fact]
    public void DenseLayer_ForwardBackward_RoundTrip()
    {
        var layer = new DenseLayer<double>(4, 3);
        var input = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));

        var output = layer.Forward(input);

        Assert.Equal(3, output.Length);

        // Verify weights are applied: non-zero input should produce non-zero output
        bool hasNonZeroOutput = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) > 1e-15)
            {
                hasNonZeroOutput = true;
                break;
            }
        }
        Assert.True(hasNonZeroOutput, "DenseLayer forward should produce non-zero output with non-zero input");

        var grad = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 0.1, 0.2, 0.3 }));
        var inputGrad = layer.Backward(grad);

        Assert.Equal(4, inputGrad.Length);

        bool hasNonZeroGrad = false;
        for (int i = 0; i < inputGrad.Length; i++)
        {
            if (Math.Abs(inputGrad[i]) > 1e-15)
            {
                hasNonZeroGrad = true;
                break;
            }
        }
        Assert.True(hasNonZeroGrad, "DenseLayer backward should produce non-zero input gradients");
    }

    [Fact]
    public void BatchNormLayer_Forward_NormalizesValues()
    {
        var layer = new BatchNormalizationLayer<double>(4);
        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 }));

        var output = layer.Forward(input);

        Assert.Equal(input.Length, output.Length);

        // BatchNorm should transform the input - output should differ from input
        bool outputDiffers = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i] - input[i]) > 1e-15)
            {
                outputDiffers = true;
                break;
            }
        }
        Assert.True(outputDiffers, "BatchNorm should transform input values, not pass them through unchanged");

        // All outputs should be finite (no NaN or Infinity from normalization)
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(double.IsFinite(output[i]), $"BatchNorm output at index {i} should be finite, got {output[i]}");
        }
    }

    [Fact]
    public void ActivationLayer_ReLU_CorrectBehavior()
    {
        var activation = new ReLUActivation<double>();
        var layer = new ActivationLayer<double>(new[] { 4 }, (IActivationFunction<double>)activation);
        var input = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { -2.0, -1.0, 1.0, 2.0 }));

        var output = layer.Forward(input);

        Assert.Equal(0.0, output[0], Tolerance); // ReLU(-2) = 0
        Assert.Equal(0.0, output[1], Tolerance); // ReLU(-1) = 0
        Assert.Equal(1.0, output[2], Tolerance); // ReLU(1) = 1
        Assert.Equal(2.0, output[3], Tolerance); // ReLU(2) = 2
    }

    #endregion

    #region Cross-Type Consistency

    [Fact]
    public void Tensor_VectorConversion_RoundTrip()
    {
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var tensor = new Tensor<double>(new[] { 4 }, original);
        var backToVector = tensor.ToVector();

        Assert.Equal(original.Length, backToVector.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], backToVector[i]);
        }
    }

    [Fact]
    public void Tensor_MatrixConversion_RoundTrip()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var tensor = new Tensor<double>(new[] { 2, 2 }, matrix);

        Assert.Equal(4, tensor.Length);
        Assert.Equal(1.0, tensor[0, 0]);
        Assert.Equal(2.0, tensor[0, 1]);
        Assert.Equal(3.0, tensor[1, 0]);
        Assert.Equal(4.0, tensor[1, 1]);
    }

    #endregion

    #region Edge Cases and Stability

    [Fact]
    public void Tensor_EmptyShape_HandledGracefully()
    {
        var tensor = new Tensor<double>(new[] { 0 });
        Assert.Equal(0, tensor.Length);
    }

    [Fact]
    public void Vector_SingleElement_Operations()
    {
        var a = new Vector<double>(new[] { 42.0 });
        var b = new Vector<double>(new[] { 8.0 });

        var sum = a + b;
        Assert.Equal(50.0, sum[0], Tolerance);
        Assert.Equal(1, sum.Length);
    }

    [Fact]
    public void NumericOperations_ExtremeValues_NoOverflow()
    {
        var ops = MathHelper.GetNumericOperations<double>();

        // Large values: 1e100 + 1e100 = 2e100 (should be finite, not NaN or Infinity)
        var large = ops.FromDouble(1e100);
        var sum = ops.Add(large, large);
        var sumDouble = ops.ToDouble(sum);
        Assert.True(double.IsFinite(sumDouble), $"Sum of 1e100 + 1e100 should be finite, got {sumDouble}");
        Assert.Equal(2e100, sumDouble, 1e90); // Expected: 2e100

        // Small * large should stay finite and non-zero: 1e-300 * 1e100 = 1e-200
        var small = ops.FromDouble(1e-300);
        var product = ops.Multiply(small, ops.FromDouble(1e100));
        var productDouble = ops.ToDouble(product);
        Assert.True(double.IsFinite(productDouble), $"Product of 1e-300 * 1e100 should be finite, got {productDouble}");
        Assert.True(productDouble > 0.0, $"Product of 1e-300 * 1e100 should be positive, got {productDouble}");
        Assert.Equal(1e-200, productDouble, 1e-210); // Expected: 1e-200
    }

    [Fact]
    public void Tensor_LargeAllocation_DoesNotThrow()
    {
        // Allocate a reasonably large tensor to test memory paths
        var tensor = new Tensor<double>(new[] { 100, 100 });

        Assert.Equal(10000, tensor.Length);

        // Fill and verify
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i * 0.01;
        }

        Assert.Equal(0.0, tensor[0]);
        Assert.Equal(99.99, tensor[9999], 1e-6);
    }

    [Fact]
    public void MathHelper_NumericOperations_CrossTypeConsistency()
    {
        // Verify that double and float NumericOperations produce consistent results
        var doubleOps = MathHelper.GetNumericOperations<double>();
        var floatOps = MathHelper.GetNumericOperations<float>();

        // Both should compute 2+3=5 and 2*3=6
        Assert.Equal(5.0, doubleOps.Add(2.0, 3.0));
        Assert.Equal(5.0f, floatOps.Add(2.0f, 3.0f));
        Assert.Equal(6.0, doubleOps.Multiply(2.0, 3.0));
        Assert.Equal(6.0f, floatOps.Multiply(2.0f, 3.0f));

        // Sqrt should be consistent across types
        Assert.Equal(4.0, doubleOps.Sqrt(16.0), 1e-10);
        Assert.Equal(4.0f, floatOps.Sqrt(16.0f), 1e-5f);

        // Conversions: FromDouble on float should truncate correctly
        Assert.Equal(3.14f, floatOps.FromDouble(3.14), 1e-5f);
    }

    [Fact]
    public void Engine_VectorScalarMultiply_Correct()
    {
        var engine = AiDotNetEngine.Current;
        var v = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var scaled = engine.Multiply(v, 0.5);

        Assert.Equal(3, scaled.Length);
        Assert.Equal(1.0, scaled[0], Tolerance);
        Assert.Equal(2.0, scaled[1], Tolerance);
        Assert.Equal(3.0, scaled[2], Tolerance);
    }

    #endregion
}
