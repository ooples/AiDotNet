using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Integration tests for VectorExtensions methods.
/// </summary>
public class VectorExtensionsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Slice Tests

    [Fact]
    public void Slice_MiddleElements_ReturnsCorrectSlice()
    {
        var vector = new Vector<double>([1, 2, 3, 4, 5]);
        var sliced = vector.Slice(1, 3);

        Assert.Equal(3, sliced.Length);
        Assert.Equal(2, sliced[0]);
        Assert.Equal(3, sliced[1]);
        Assert.Equal(4, sliced[2]);
    }

    [Fact]
    public void Slice_FromStart_ReturnsCorrectSlice()
    {
        var vector = new Vector<double>([10, 20, 30, 40]);
        var sliced = vector.Slice(0, 2);

        Assert.Equal(2, sliced.Length);
        Assert.Equal(10, sliced[0]);
        Assert.Equal(20, sliced[1]);
    }

    [Fact]
    public void Slice_ToEnd_ReturnsCorrectSlice()
    {
        var vector = new Vector<double>([1, 2, 3, 4, 5]);
        var sliced = vector.Slice(3, 2);

        Assert.Equal(2, sliced.Length);
        Assert.Equal(4, sliced[0]);
        Assert.Equal(5, sliced[1]);
    }

    #endregion

    #region Norm Tests

    [Fact]
    public void Norm_SimpleVector_ReturnsCorrectNorm()
    {
        var vector = new Vector<double>([3, 4]);
        var norm = vector.Norm();

        Assert.Equal(5, norm, Tolerance);
    }

    [Fact]
    public void Norm_ZeroVector_ReturnsZero()
    {
        var vector = new Vector<double>([0, 0, 0]);
        var norm = vector.Norm();

        Assert.Equal(0, norm, Tolerance);
    }

    [Fact]
    public void Norm_UnitVector_ReturnsOne()
    {
        var vector = new Vector<double>([1, 0, 0]);
        var norm = vector.Norm();

        Assert.Equal(1, norm, Tolerance);
    }

    #endregion

    #region ToVectorList/ToIntList Tests

    [Fact]
    public void ToVectorList_Indices_CreatesVectorsFromInts()
    {
        var indices = new[] { 1, 2, 3 };
        var vectorList = indices.ToVectorList<double>();

        Assert.Equal(3, vectorList.Count);
        Assert.Equal(1, vectorList[0][0]);
        Assert.Equal(2, vectorList[1][0]);
        Assert.Equal(3, vectorList[2][0]);
    }

    [Fact]
    public void ToIntList_Vectors_ConvertsBackToInts()
    {
        var vectors = new List<Vector<double>>
        {
            new([5.0]),
            new([10.0]),
            new([15.0])
        };
        var intList = vectors.ToIntList();

        Assert.Equal(3, intList.Count);
        Assert.Equal(5, intList[0]);
        Assert.Equal(10, intList[1]);
        Assert.Equal(15, intList[2]);
    }

    #endregion

    #region CreateDiagonal Tests

    [Fact]
    public void CreateDiagonal_SimpleVector_CreatesDiagonalMatrix()
    {
        var vector = new Vector<double>([1, 2, 3]);
        var diagonal = vector.CreateDiagonal();

        Assert.Equal(3, diagonal.Rows);
        Assert.Equal(3, diagonal.Columns);
        Assert.Equal(1, diagonal[0, 0]);
        Assert.Equal(2, diagonal[1, 1]);
        Assert.Equal(3, diagonal[2, 2]);
        Assert.Equal(0, diagonal[0, 1]);
        Assert.Equal(0, diagonal[1, 0]);
    }

    #endregion

    #region Argsort Tests

    [Fact]
    public void Argsort_UnsortedVector_ReturnsSortedIndices()
    {
        var vector = new Vector<double>([3, 1, 4, 1, 5]);
        var indices = vector.Argsort();

        Assert.Equal(1, indices[0]); // 1 at index 1
        Assert.Equal(3, indices[1]); // 1 at index 3
        Assert.Equal(0, indices[2]); // 3 at index 0
        Assert.Equal(2, indices[3]); // 4 at index 2
        Assert.Equal(4, indices[4]); // 5 at index 4
    }

    [Fact]
    public void Argsort_SortedVector_ReturnsSequentialIndices()
    {
        var vector = new Vector<double>([1, 2, 3, 4, 5]);
        var indices = vector.Argsort();

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(i, indices[i]);
        }
    }

    #endregion

    #region Repeat Tests

    [Fact]
    public void Repeat_Vector_RepeatsCorrectly()
    {
        var vector = new Vector<double>([1, 2]);
        var repeated = vector.Repeat(3);

        Assert.Equal(6, repeated.Length);
        Assert.Equal(1, repeated[0]);
        Assert.Equal(2, repeated[1]);
        Assert.Equal(1, repeated[2]);
        Assert.Equal(2, repeated[3]);
        Assert.Equal(1, repeated[4]);
        Assert.Equal(2, repeated[5]);
    }

    [Fact]
    public void Repeat_Once_ReturnsCopy()
    {
        var vector = new Vector<double>([5, 10, 15]);
        var repeated = vector.Repeat(1);

        Assert.Equal(3, repeated.Length);
        Assert.Equal(5, repeated[0]);
        Assert.Equal(10, repeated[1]);
        Assert.Equal(15, repeated[2]);
    }

    #endregion

    #region Add (Vector) Tests

    [Fact]
    public void Add_TwoVectors_ReturnsSum()
    {
        var a = new Vector<double>([1, 2, 3]);
        var b = new Vector<double>([4, 5, 6]);
        var sum = a.Add(b);

        Assert.Equal(5, sum[0]);
        Assert.Equal(7, sum[1]);
        Assert.Equal(9, sum[2]);
    }

    [Fact]
    public void Add_DifferentLengths_ThrowsException()
    {
        var a = new Vector<double>([1, 2, 3]);
        var b = new Vector<double>([1, 2]);

        Assert.Throws<ArgumentException>(() => a.Add(b));
    }

    #endregion

    #region Add (Scalar) Tests

    [Fact]
    public void Add_Scalar_AddsToEachElement()
    {
        var vector = new Vector<double>([1, 2, 3]);
        var result = vector.Add(5.0);

        Assert.Equal(6, result[0]);
        Assert.Equal(7, result[1]);
        Assert.Equal(8, result[2]);
    }

    #endregion

    #region PointwiseExp Tests

    [Fact]
    public void PointwiseExp_SimpleVector_ReturnsExponential()
    {
        var vector = new Vector<double>([0, 1, 2]);
        var exp = vector.PointwiseExp();

        Assert.Equal(1, exp[0], Tolerance);
        Assert.Equal(Math.E, exp[1], Tolerance);
        Assert.Equal(Math.E * Math.E, exp[2], Tolerance);
    }

    #endregion

    #region PointwiseLog Tests

    [Fact]
    public void PointwiseLog_SimpleVector_ReturnsLogarithm()
    {
        var vector = new Vector<double>([1, Math.E, Math.E * Math.E]);
        var log = vector.PointwiseLog();

        Assert.Equal(0, log[0], Tolerance);
        Assert.Equal(1, log[1], Tolerance);
        Assert.Equal(2, log[2], Tolerance);
    }

    #endregion

    #region Subtract Tests

    [Fact]
    public void Subtract_TwoVectors_ReturnsDifference()
    {
        var a = new Vector<double>([5, 8, 3]);
        var b = new Vector<double>([2, 3, 1]);
        var diff = a.Subtract(b);

        Assert.Equal(3, diff[0]);
        Assert.Equal(5, diff[1]);
        Assert.Equal(2, diff[2]);
    }

    [Fact]
    public void Subtract_Scalar_SubtractsFromEach()
    {
        var vector = new Vector<double>([10, 15, 20]);
        var result = vector.Subtract(5.0);

        Assert.Equal(5, result[0]);
        Assert.Equal(10, result[1]);
        Assert.Equal(15, result[2]);
    }

    #endregion

    #region DotProduct Tests

    [Fact]
    public void DotProduct_TwoVectors_ReturnsCorrectResult()
    {
        var a = new Vector<double>([1, 2, 3]);
        var b = new Vector<double>([4, 5, 6]);
        var dot = a.DotProduct(b);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Assert.Equal(32, dot, Tolerance);
    }

    [Fact]
    public void DotProduct_OrthogonalVectors_ReturnsZero()
    {
        var a = new Vector<double>([1, 0, 0]);
        var b = new Vector<double>([0, 1, 0]);
        var dot = a.DotProduct(b);

        Assert.Equal(0, dot, Tolerance);
    }

    #endregion

    #region Divide (Scalar) Tests

    [Fact]
    public void Divide_ByScalar_DividesEachElement()
    {
        var vector = new Vector<double>([10, 20, 30]);
        var result = vector.Divide(10.0);

        Assert.Equal(1, result[0]);
        Assert.Equal(2, result[1]);
        Assert.Equal(3, result[2]);
    }

    #endregion

    #region Multiply (Scalar) Tests

    [Fact]
    public void Multiply_ByScalar_MultipliesEachElement()
    {
        var vector = new Vector<double>([1, 2, 3]);
        var result = vector.Multiply(10.0);

        Assert.Equal(10, result[0]);
        Assert.Equal(20, result[1]);
        Assert.Equal(30, result[2]);
    }

    #endregion

    #region Multiply (Matrix) Tests

    [Fact]
    public void Multiply_VectorByMatrix_ReturnsCorrectResult()
    {
        var vector = new Vector<double>([1, 2, 3]);
        var matrix = new Matrix<double>(3, 2);
        matrix[0, 0] = 1; matrix[0, 1] = 4;
        matrix[1, 0] = 2; matrix[1, 1] = 5;
        matrix[2, 0] = 3; matrix[2, 1] = 6;

        var result = vector.Multiply(matrix);

        // result[0] = 1*1 + 2*2 + 3*3 = 14
        // result[1] = 1*4 + 2*5 + 3*6 = 32
        Assert.Equal(2, result.Length);
        Assert.Equal(14, result[0], Tolerance);
        Assert.Equal(32, result[1], Tolerance);
    }

    #endregion

    #region PointwiseMultiply Tests

    [Fact]
    public void PointwiseMultiply_TwoVectors_ReturnsHadamardProduct()
    {
        var a = new Vector<double>([1, 2, 3]);
        var b = new Vector<double>([4, 5, 6]);
        var result = a.PointwiseMultiply(b);

        Assert.Equal(4, result[0]);
        Assert.Equal(10, result[1]);
        Assert.Equal(18, result[2]);
    }

    #endregion

    #region PointwiseMultiplyInPlace Tests

    [Fact]
    public void PointwiseMultiplyInPlace_ModifiesLeftVector()
    {
        var a = new Vector<double>([1, 2, 3]);
        var b = new Vector<double>([4, 5, 6]);
        a.PointwiseMultiplyInPlace(b);

        Assert.Equal(4, a[0]);
        Assert.Equal(10, a[1]);
        Assert.Equal(18, a[2]);
    }

    #endregion

    #region OuterProduct Tests

    [Fact]
    public void OuterProduct_TwoVectors_ReturnsCorrectMatrix()
    {
        var a = new Vector<double>([1, 2]);
        var b = new Vector<double>([3, 4, 5]);
        var outer = a.OuterProduct(b);

        Assert.Equal(2, outer.Rows);
        Assert.Equal(3, outer.Columns);
        Assert.Equal(3, outer[0, 0]);  // 1*3
        Assert.Equal(4, outer[0, 1]);  // 1*4
        Assert.Equal(5, outer[0, 2]);  // 1*5
        Assert.Equal(6, outer[1, 0]);  // 2*3
        Assert.Equal(8, outer[1, 1]);  // 2*4
        Assert.Equal(10, outer[1, 2]); // 2*5
    }

    #endregion

    #region Magnitude Tests

    [Fact]
    public void Magnitude_SimpleVector_ReturnsCorrectMagnitude()
    {
        var vector = new Vector<double>([3, 4]);
        var mag = vector.Magnitude();

        Assert.Equal(5, mag, Tolerance);
    }

    #endregion

    #region PointwiseDivide Tests

    [Fact]
    public void PointwiseDivide_TwoVectors_DividesElementwise()
    {
        var a = new Vector<double>([10, 8, 6]);
        var b = new Vector<double>([2, 4, 3]);
        var result = a.PointwiseDivide(b);

        Assert.Equal(5, result[0]);
        Assert.Equal(2, result[1]);
        Assert.Equal(2, result[2]);
    }

    #endregion

    #region Max Tests

    [Fact]
    public void Max_Vector_ReturnsMaximumValue()
    {
        var vector = new Vector<double>([3, 7, 2, 5]);
        var max = vector.Max();

        Assert.Equal(7, max);
    }

    [Fact]
    public void Max_EmptyVector_ThrowsException()
    {
        var vector = new Vector<double>(0);

        Assert.Throws<ArgumentException>(() => vector.Max());
    }

    #endregion

    #region Min Tests

    [Fact]
    public void Min_Vector_ReturnsMinimumValue()
    {
        var vector = new Vector<double>([3, 7, 2, 5]);
        var min = vector.Min();

        Assert.Equal(2, min);
    }

    #endregion

    #region Average Tests

    [Fact]
    public void Average_Vector_ReturnsArithmeticMean()
    {
        var vector = new Vector<double>([2, 4, 6, 8]);
        var avg = vector.Average();

        Assert.Equal(5, avg, Tolerance);
    }

    #endregion

    #region Sum Tests

    [Fact]
    public void Sum_Vector_ReturnsTotalSum()
    {
        var vector = new Vector<double>([1, 2, 3, 4]);
        var sum = vector.Sum();

        Assert.Equal(10, sum, Tolerance);
    }

    #endregion

    #region PointwiseSign Tests

    [Fact]
    public void PointwiseSign_MixedVector_ReturnsCorrectSigns()
    {
        var vector = new Vector<double>([-5, 0, 3]);
        var signs = vector.PointwiseSign();

        Assert.Equal(-1, signs[0]);
        Assert.Equal(0, signs[1]);
        Assert.Equal(1, signs[2]);
    }

    #endregion

    #region AbsoluteMaximum Tests

    [Fact]
    public void AbsoluteMaximum_MixedVector_ReturnsLargestAbsoluteValue()
    {
        var vector = new Vector<double>([-10, 5, 3]);
        var absMax = vector.AbsoluteMaximum();

        Assert.Equal(10, absMax);
    }

    #endregion

    #region PointwiseAbs Tests

    [Fact]
    public void PointwiseAbs_MixedVector_ReturnsAbsoluteValues()
    {
        var vector = new Vector<double>([-5, 0, 3]);
        var abs = vector.PointwiseAbs();

        Assert.Equal(5, abs[0]);
        Assert.Equal(0, abs[1]);
        Assert.Equal(3, abs[2]);
    }

    #endregion

    #region PointwiseSqrt Tests

    [Fact]
    public void PointwiseSqrt_PerfectSquares_ReturnsCorrectRoots()
    {
        var vector = new Vector<double>([4, 9, 16]);
        var sqrt = vector.PointwiseSqrt();

        Assert.Equal(2, sqrt[0], Tolerance);
        Assert.Equal(3, sqrt[1], Tolerance);
        Assert.Equal(4, sqrt[2], Tolerance);
    }

    #endregion

    #region Maximum (scalar) Tests

    [Fact]
    public void Maximum_WithScalar_ReturnsElementwiseMaximum()
    {
        var vector = new Vector<double>([1, 5, 3]);
        var result = vector.Maximum(2.0);

        Assert.Equal(2, result[0]); // max(1, 2) = 2
        Assert.Equal(5, result[1]); // max(5, 2) = 5
        Assert.Equal(3, result[2]); // max(3, 2) = 3
    }

    #endregion

    #region Transform Tests

    [Fact]
    public void Transform_AppliesFunction()
    {
        var vector = new Vector<double>([1, 2, 3]);
        var result = vector.Transform(x => x * x);

        Assert.Equal(1, result[0]);
        Assert.Equal(4, result[1]);
        Assert.Equal(9, result[2]);
    }

    #endregion

    #region MaxIndex Tests

    [Fact]
    public void MaxIndex_Vector_ReturnsIndexOfMaximum()
    {
        var vector = new Vector<double>([3, 7, 2, 5]);
        var idx = vector.MaxIndex();

        Assert.Equal(1, idx);
    }

    #endregion

    #region MinIndex Tests

    [Fact]
    public void MinIndex_Vector_ReturnsIndexOfMinimum()
    {
        var vector = new Vector<double>([3, 7, 2, 5]);
        var idx = vector.MinIndex();

        Assert.Equal(2, idx);
    }

    #endregion

    #region SubVector Tests

    [Fact]
    public void SubVector_ByRange_ExtractsCorrectElements()
    {
        var vector = new Vector<double>([5, 10, 15, 20, 25]);
        var sub = vector.SubVector(1, 3);

        Assert.Equal(3, sub.Length);
        Assert.Equal(10, sub[0]);
        Assert.Equal(15, sub[1]);
        Assert.Equal(20, sub[2]);
    }

    [Fact]
    public void SubVector_ByIndices_ExtractsCorrectElements()
    {
        var vector = new Vector<double>([5, 10, 15, 20, 25]);
        var sub = vector.SubVector([0, 3, 4]);

        Assert.Equal(3, sub.Length);
        Assert.Equal(5, sub[0]);
        Assert.Equal(20, sub[1]);
        Assert.Equal(25, sub[2]);
    }

    #endregion

    #region ToDiagonalMatrix Tests

    [Fact]
    public void ToDiagonalMatrix_CreatesCorrectMatrix()
    {
        var vector = new Vector<double>([3, 5, 7]);
        var diag = vector.ToDiagonalMatrix();

        Assert.Equal(3, diag.Rows);
        Assert.Equal(3, diag.Columns);
        Assert.Equal(3, diag[0, 0]);
        Assert.Equal(5, diag[1, 1]);
        Assert.Equal(7, diag[2, 2]);
        Assert.Equal(0, diag[0, 1]);
    }

    #endregion

    #region ToRowMatrix Tests

    [Fact]
    public void ToRowMatrix_CreatesCorrectMatrix()
    {
        var vector = new Vector<double>([3, 5, 7]);
        var row = vector.ToRowMatrix();

        Assert.Equal(1, row.Rows);
        Assert.Equal(3, row.Columns);
        Assert.Equal(3, row[0, 0]);
        Assert.Equal(5, row[0, 1]);
        Assert.Equal(7, row[0, 2]);
    }

    #endregion

    #region ToColumnMatrix Tests

    [Fact]
    public void ToColumnMatrix_CreatesCorrectMatrix()
    {
        var vector = new Vector<double>([3, 5, 7]);
        var col = vector.ToColumnMatrix();

        Assert.Equal(3, col.Rows);
        Assert.Equal(1, col.Columns);
        Assert.Equal(3, col[0, 0]);
        Assert.Equal(5, col[1, 0]);
        Assert.Equal(7, col[2, 0]);
    }

    #endregion

    #region Extract Tests

    [Fact]
    public void Extract_FirstNElements_ReturnsCorrectSubset()
    {
        var vector = new Vector<double>([5, 10, 15, 20, 25]);
        var extracted = vector.Extract(3);

        Assert.Equal(3, extracted.Length);
        Assert.Equal(5, extracted[0]);
        Assert.Equal(10, extracted[1]);
        Assert.Equal(15, extracted[2]);
    }

    #endregion

    #region Reshape Tests

    [Fact]
    public void Reshape_VectorToMatrix_CorrectDimensions()
    {
        var vector = new Vector<double>([1, 2, 3, 4, 5, 6]);
        var matrix = vector.Reshape(2, 3);

        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[0, 1]);
        Assert.Equal(3, matrix[0, 2]);
        Assert.Equal(4, matrix[1, 0]);
        Assert.Equal(5, matrix[1, 1]);
        Assert.Equal(6, matrix[1, 2]);
    }

    [Fact]
    public void Reshape_IncompatibleSize_ThrowsException()
    {
        var vector = new Vector<double>([1, 2, 3, 4, 5]);

        Assert.Throws<ArgumentException>(() => vector.Reshape(2, 3));
    }

    #endregion

    #region StandardDeviation Tests

    [Fact]
    public void StandardDeviation_KnownData_ReturnsCorrectValue()
    {
        var vector = new Vector<double>([2, 4, 4, 4, 5, 5, 7, 9]);
        var stdDev = vector.StandardDeviation();

        // Mean = 5, sample variance = ((2-5)^2 + (4-5)^2 * 3 + (5-5)^2 * 2 + (7-5)^2 + (9-5)^2) / 7
        // = (9 + 3 + 0 + 4 + 16) / 7 = 32 / 7 = 4.571...
        // sample std dev = sqrt(32/7) = 2.138...
        Assert.Equal(2.138089935299395, stdDev, 1e-6);
    }

    [Fact]
    public void StandardDeviation_SingleElement_ReturnsZero()
    {
        var vector = new Vector<double>([5.0]);
        var stdDev = vector.StandardDeviation();

        Assert.Equal(0, stdDev);
    }

    #endregion

    #region Median Tests

    [Fact]
    public void Median_OddLength_ReturnsMiddleValue()
    {
        var vector = new Vector<double>([1, 3, 5, 7, 9]);
        var median = vector.Median();

        Assert.Equal(5, median);
    }

    [Fact]
    public void Median_EvenLength_ReturnsAverageOfMiddle()
    {
        var vector = new Vector<double>([1, 3, 5, 7]);
        var median = vector.Median();

        Assert.Equal(4, median, Tolerance); // (3 + 5) / 2 = 4
    }

    [Fact]
    public void Median_UnsortedVector_ReturnsCorrectMedian()
    {
        var vector = new Vector<double>([9, 1, 7, 3, 5]);
        var median = vector.Median();

        Assert.Equal(5, median);
    }

    #endregion

    #region EuclideanDistance Tests

    [Fact]
    public void EuclideanDistance_TwoVectors_ReturnsCorrectDistance()
    {
        var a = new Vector<double>([1, 2]);
        var b = new Vector<double>([4, 6]);
        var distance = a.EuclideanDistance(b);

        // sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = sqrt(25) = 5
        Assert.Equal(5, distance, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_SameVectors_ReturnsZero()
    {
        var a = new Vector<double>([1, 2, 3]);
        var b = new Vector<double>([1, 2, 3]);
        var distance = a.EuclideanDistance(b);

        Assert.Equal(0, distance, Tolerance);
    }

    #endregion

    #region Subvector Tests

    [Fact]
    public void Subvector_ByIndices_ExtractsCorrectElements()
    {
        var vector = new Vector<double>([10, 20, 30, 40, 50]);
        var sub = vector.Subvector([1, 3]);

        Assert.Equal(2, sub.Length);
        Assert.Equal(20, sub[0]);
        Assert.Equal(40, sub[1]);
    }

    #endregion

    #region Minimum Tests

    [Fact]
    public void Minimum_Vector_ReturnsSmallestValue()
    {
        var vector = new Vector<double>([5, 2, 8, 1]);
        var min = vector.Minimum();

        Assert.Equal(1, min);
    }

    #endregion

    #region Transform<T, TResult> Tests

    [Fact]
    public void Transform_ToDoubleFromInt_TransformsCorrectly()
    {
        var vector = new Vector<double>([1, 2, 3, 4, 5]);

        var result = vector.Transform<double, int>(x => (int)(x * 10));

        Assert.Equal(5, result.Length);
        Assert.Equal(10, result[0]);
        Assert.Equal(50, result[4]);
    }

    [Fact]
    public void Transform_WithTypeChange_PreservesLength()
    {
        var vector = new Vector<double>([1.5, 2.5, 3.5]);

        var result = vector.Transform<double, float>(x => (float)x);

        Assert.Equal(3, result.Length);
        Assert.Equal(1.5f, result[0], 0.001f);
        Assert.Equal(2.5f, result[1], 0.001f);
        Assert.Equal(3.5f, result[2], 0.001f);
    }

    #endregion

    #region ToRealVector Tests

    [Fact]
    public void ToRealVector_ExtractsRealPartFromComplex()
    {
        // Arrange - Create complex vector
        var complexVector = new Vector<Complex<double>>([
            new Complex<double>(1.0, 2.0),
            new Complex<double>(3.0, 4.0),
            new Complex<double>(5.0, 6.0)
        ]);

        // Act
        var realVector = complexVector.ToRealVector();

        // Assert - Should extract real parts only
        Assert.Equal(3, realVector.Length);
        Assert.Equal(1.0, realVector[0], Tolerance);
        Assert.Equal(3.0, realVector[1], Tolerance);
        Assert.Equal(5.0, realVector[2], Tolerance);
    }

    [Fact]
    public void ToRealVector_PureImaginary_ReturnsZeros()
    {
        // Arrange - Complex vector with zero real parts
        var complexVector = new Vector<Complex<double>>([
            new Complex<double>(0.0, 1.0),
            new Complex<double>(0.0, 2.0)
        ]);

        // Act
        var realVector = complexVector.ToRealVector();

        // Assert
        Assert.Equal(0.0, realVector[0], Tolerance);
        Assert.Equal(0.0, realVector[1], Tolerance);
    }

    #endregion
}
