using AiDotNet.Extensions;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Extensions;

/// <summary>
/// Integration tests for extension method classes:
/// VectorExtensions, TensorExtensions, EnumerableExtensions, NumericTypeExtensions.
/// </summary>
public class ExtensionsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region VectorExtensions - Slice

    [Fact]
    public void VectorExtensions_Slice_ReturnsCorrectSubset()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var sliced = v.Slice(1, 3);
        Assert.Equal(3, sliced.Length);
        Assert.Equal(2.0, sliced[0], Tolerance);
        Assert.Equal(3.0, sliced[1], Tolerance);
        Assert.Equal(4.0, sliced[2], Tolerance);
    }

    [Fact]
    public void VectorExtensions_Slice_FullVector()
    {
        var v = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
        var sliced = v.Slice(0, 3);
        Assert.Equal(3, sliced.Length);
        Assert.Equal(10.0, sliced[0], Tolerance);
        Assert.Equal(30.0, sliced[2], Tolerance);
    }

    [Fact]
    public void VectorExtensions_Slice_SingleElement()
    {
        var v = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var sliced = v.Slice(2, 1);
        Assert.Equal(1, sliced.Length);
        Assert.Equal(3.0, sliced[0], Tolerance);
    }

    #endregion

    #region TensorExtensions - ConvertToMatrix

    [Fact]
    public void TensorExtensions_ConvertToMatrix_2D()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var matrix = tensor.ConvertToMatrix();
        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0], Tolerance);
        Assert.Equal(6.0, matrix[1, 2], Tolerance);
    }

    [Fact]
    public void TensorExtensions_ConvertToMatrix_1D()
    {
        var tensor = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 1, 2, 3, 4 }));
        var matrix = tensor.ConvertToMatrix();
        Assert.Equal(4, matrix.Rows);
        Assert.Equal(1, matrix.Columns);
    }

    [Fact]
    public void TensorExtensions_ConvertToMatrix_3D_Throws()
    {
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });
        Assert.Throws<ArgumentException>(() => tensor.ConvertToMatrix());
    }

    #endregion

    #region TensorExtensions - Unflatten

    [Fact]
    public void TensorExtensions_Unflatten_PreservesShape()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        var flat = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 });
        var result = tensor.Unflatten(flat);
        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(6.0, result[1, 2], Tolerance);
    }

    [Fact]
    public void TensorExtensions_Unflatten_WrongSize_Throws()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 });
        var flat = new Vector<double>(new double[] { 1, 2, 3 }); // Size 3, but tensor needs 6
        Assert.Throws<ArgumentException>(() => tensor.Unflatten(flat));
    }

    #endregion

    #region TensorExtensions - TensorEquals

    [Fact]
    public void TensorExtensions_TensorEquals_SameTensors_True()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        Assert.True(a.TensorEquals(b));
    }

    [Fact]
    public void TensorExtensions_TensorEquals_DifferentValues_False()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var b = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 4 }));
        Assert.False(a.TensorEquals(b));
    }

    [Fact]
    public void TensorExtensions_TensorEquals_DifferentShapes_False()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var b = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 1, 2 }));
        Assert.False(a.TensorEquals(b));
    }

    #endregion

    #region TensorExtensions - ConcatenateTensors

    [Fact]
    public void TensorExtensions_Concatenate_2D()
    {
        var a = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new double[] { 1, 2, 3, 4 }));
        var b = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 5, 6, 7, 8, 9, 10 }));
        var result = a.ConcatenateTensors(b);
        Assert.Equal(new[] { 2, 5 }, result.Shape);
    }

    [Fact]
    public void TensorExtensions_Concatenate_DifferentRanks_Throws()
    {
        var a = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
        var b = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 4, 5, 6 }));
        Assert.Throws<ArgumentException>(() => a.ConcatenateTensors(b));
    }

    #endregion

    #region TensorExtensions - ForEachPosition

    [Fact]
    public void TensorExtensions_ForEachPosition_VisitsAllElements()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        int count = 0;
        tensor.ForEachPosition((pos, val) => { count++; return true; });
        Assert.Equal(6, count);
    }

    [Fact]
    public void TensorExtensions_ForEachPosition_CanStopEarly()
    {
        var tensor = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        int count = 0;
        tensor.ForEachPosition((pos, val) =>
        {
            count++;
            return count < 3; // Stop after 3 elements
        });
        Assert.Equal(3, count);
    }

    #endregion

    #region TensorExtensions - CreateOnesTensor

    [Fact]
    public void TensorExtensions_CreateOnesTensor()
    {
        var tensor = TensorExtensions.CreateOnesTensor<double>(5);
        Assert.Equal(5, tensor.Length);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(1.0, tensor[i], Tolerance);
        }
    }

    #endregion

    #region TensorExtensions - HeStddev and XavierStddev

    [Fact]
    public void TensorExtensions_HeStddev_KnownValue()
    {
        // He stddev = sqrt(2/fanIn)
        double result = TensorExtensions.HeStddev(100);
        Assert.Equal(Math.Sqrt(2.0 / 100), result, Tolerance);
    }

    [Fact]
    public void TensorExtensions_XavierStddev_KnownValue()
    {
        // Xavier stddev = sqrt(2/(fanIn+fanOut))
        double result = TensorExtensions.XavierStddev(100, 200);
        Assert.Equal(Math.Sqrt(2.0 / 300), result, Tolerance);
    }

    [Fact]
    public void TensorExtensions_HeStddev_SymmetricInputOutput()
    {
        // For equal input and output, He and Xavier should differ
        double he = TensorExtensions.HeStddev(100);
        double xavier = TensorExtensions.XavierStddev(100, 100);
        Assert.True(he > xavier, "He stddev should be larger than Xavier for same fanIn");
    }

    #endregion

    #region TensorExtensions - CreateXavierInitializedTensor

    [Fact]
    public void TensorExtensions_CreateXavierInitializedTensor_CorrectShape()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var tensor = TensorExtensions.CreateXavierInitializedTensor<double>(new[] { 3, 4 }, 0.1, random);
        Assert.Equal(new[] { 3, 4 }, tensor.Shape);
        Assert.Equal(12, tensor.Length);
    }

    [Fact]
    public void TensorExtensions_CreateXavierInitializedTensor_NotAllZero()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var tensor = TensorExtensions.CreateXavierInitializedTensor<double>(new[] { 10 }, 1.0, random);
        bool anyNonZero = false;
        for (int i = 0; i < tensor.Length; i++)
        {
            if (Math.Abs(tensor[i]) > 1e-15)
            {
                anyNonZero = true;
                break;
            }
        }
        Assert.True(anyNonZero, "Xavier initialized tensor should have non-zero values");
    }

    [Fact]
    public void TensorExtensions_CreateXavierInitializedTensor_BoundedByStddev()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        double stddev = 0.5;
        var tensor = TensorExtensions.CreateXavierInitializedTensor<double>(new[] { 100 }, stddev, random);
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.True(Math.Abs(tensor[i]) <= stddev,
                $"Value {tensor[i]} exceeds stddev bound of {stddev}");
        }
    }

    #endregion

    #region NumericTypeExtensions - IsRealType

    [Fact]
    public void NumericTypeExtensions_IsRealType_Double()
    {
        Assert.True(NumericTypeExtensions.IsRealType<double>());
    }

    [Fact]
    public void NumericTypeExtensions_IsRealType_Float()
    {
        Assert.True(NumericTypeExtensions.IsRealType<float>());
    }

    [Fact]
    public void NumericTypeExtensions_IsRealType_Int()
    {
        Assert.True(NumericTypeExtensions.IsRealType<int>());
    }

    [Fact]
    public void NumericTypeExtensions_IsRealType_Long()
    {
        Assert.True(NumericTypeExtensions.IsRealType<long>());
    }

    [Fact]
    public void NumericTypeExtensions_IsRealType_Decimal()
    {
        Assert.True(NumericTypeExtensions.IsRealType<decimal>());
    }

    [Fact]
    public void NumericTypeExtensions_IsRealType_String_False()
    {
        Assert.False(NumericTypeExtensions.IsRealType<string>());
    }

    #endregion

    #region NumericTypeExtensions - IsComplexType

    [Fact]
    public void NumericTypeExtensions_IsComplexType_Complex()
    {
        Assert.True(NumericTypeExtensions.IsComplexType<Complex<double>>());
    }

    [Fact]
    public void NumericTypeExtensions_IsComplexType_Double_False()
    {
        Assert.False(NumericTypeExtensions.IsComplexType<double>());
    }

    #endregion

    #region EnumerableExtensions - RandomElement

    [Fact]
    public void EnumerableExtensions_RandomElement_ReturnsElementFromCollection()
    {
        var source = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var element = source.RandomElement();
        Assert.Contains(element, source);
    }

    [Fact]
    public void EnumerableExtensions_RandomElement_EmptyCollection_ReturnsZero()
    {
        var empty = Array.Empty<double>();
        var result = empty.RandomElement();
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void EnumerableExtensions_RandomElement_SingleElement()
    {
        var single = new double[] { 42.0 };
        var result = single.RandomElement();
        Assert.Equal(42.0, result, Tolerance);
    }

    #endregion
}
