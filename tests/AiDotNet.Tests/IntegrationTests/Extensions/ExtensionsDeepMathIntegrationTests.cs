using AiDotNet.Extensions;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Extensions;

/// <summary>
/// Deep math integration tests for VectorExtensions and MatrixExtensions:
/// vector arithmetic, norms, dot product, outer product, diagonal matrix,
/// pointwise operations, statistics, and linear algebra identities.
/// </summary>
public class ExtensionsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    // ============================
    // Vector Norm / Magnitude Tests
    // ============================

    [Fact]
    public void Norm_UnitVector_IsOne()
    {
        // ||e_i|| = 1 for any standard basis vector
        var v = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
        Assert.Equal(1.0, v.Norm(), Tolerance);
    }

    [Fact]
    public void Norm_345Triangle_Is5()
    {
        // ||[3, 4]|| = sqrt(9 + 16) = 5
        var v = new Vector<double>(new[] { 3.0, 4.0 });
        Assert.Equal(5.0, v.Norm(), Tolerance);
    }

    [Fact]
    public void Magnitude_EqualsNorm()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        Assert.Equal(v.Norm(), v.Magnitude(), Tolerance);
    }

    [Fact]
    public void Norm_ZeroVector_IsZero()
    {
        var v = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        Assert.Equal(0.0, v.Norm(), Tolerance);
    }

    [Fact]
    public void Norm_ScaledVector_ScalesLinearly()
    {
        // ||alpha * v|| = |alpha| * ||v||
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        double alpha = 3.0;
        var scaled = v.Multiply(alpha);
        Assert.Equal(alpha * v.Norm(), scaled.Norm(), Tolerance);
    }

    [Fact]
    public void Norm_HandComputed()
    {
        // ||[1, 2, 2]|| = sqrt(1 + 4 + 4) = 3
        var v = new Vector<double>(new[] { 1.0, 2.0, 2.0 });
        Assert.Equal(3.0, v.Norm(), Tolerance);
    }

    // ============================
    // Dot Product Tests
    // ============================

    [Fact]
    public void DotProduct_OrthogonalVectors_IsZero()
    {
        // e1 . e2 = 0
        var v1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
        Assert.Equal(0.0, v1.DotProduct(v2), Tolerance);
    }

    [Fact]
    public void DotProduct_HandComputed()
    {
        // [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        Assert.Equal(32.0, v1.DotProduct(v2), Tolerance);
    }

    [Fact]
    public void DotProduct_SelfDot_EqualsNormSquared()
    {
        // v . v = ||v||^2
        var v = new Vector<double>(new[] { 2.0, 3.0, 6.0 });
        double norm = v.Norm();
        Assert.Equal(norm * norm, v.DotProduct(v), Tolerance);
    }

    [Fact]
    public void DotProduct_IsCommutative()
    {
        var v1 = new Vector<double>(new[] { 1.0, -2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, -6.0 });
        Assert.Equal(v1.DotProduct(v2), v2.DotProduct(v1), Tolerance);
    }

    [Fact]
    public void DotProduct_IsLinear()
    {
        // (a*v1 + v2) . v3 = a*(v1.v3) + (v2.v3)
        var v1 = new Vector<double>(new[] { 1.0, 2.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0 });
        var v3 = new Vector<double>(new[] { 5.0, 6.0 });
        double a = 2.5;

        var lhs = v1.Multiply(a).Add(v2).DotProduct(v3);
        var rhs = a * v1.DotProduct(v3) + v2.DotProduct(v3);
        Assert.Equal(rhs, lhs, Tolerance);
    }

    // ============================
    // Vector Arithmetic Tests
    // ============================

    [Fact]
    public void Add_HandComputed()
    {
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var sum = v1.Add(v2);
        Assert.Equal(5.0, sum[0], Tolerance);
        Assert.Equal(7.0, sum[1], Tolerance);
        Assert.Equal(9.0, sum[2], Tolerance);
    }

    [Fact]
    public void Subtract_HandComputed()
    {
        var v1 = new Vector<double>(new[] { 5.0, 8.0, 3.0 });
        var v2 = new Vector<double>(new[] { 2.0, 3.0, 1.0 });
        var diff = v1.Subtract(v2);
        Assert.Equal(3.0, diff[0], Tolerance);
        Assert.Equal(5.0, diff[1], Tolerance);
        Assert.Equal(2.0, diff[2], Tolerance);
    }

    [Fact]
    public void Add_SubtractInverse()
    {
        // v + w - w = v
        var v = new Vector<double>(new[] { 1.5, -2.3, 4.7 });
        var w = new Vector<double>(new[] { 3.1, 0.8, -1.2 });
        var result = v.Add(w).Subtract(w);
        for (int i = 0; i < v.Length; i++)
            Assert.Equal(v[i], result[i], Tolerance);
    }

    [Fact]
    public void ScalarMultiply_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var scaled = v.Multiply(10.0);
        Assert.Equal(10.0, scaled[0], Tolerance);
        Assert.Equal(20.0, scaled[1], Tolerance);
        Assert.Equal(30.0, scaled[2], Tolerance);
    }

    [Fact]
    public void ScalarDivide_HandComputed()
    {
        var v = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var divided = v.Divide(10.0);
        Assert.Equal(1.0, divided[0], Tolerance);
        Assert.Equal(2.0, divided[1], Tolerance);
        Assert.Equal(3.0, divided[2], Tolerance);
    }

    [Fact]
    public void ScalarAdd_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var result = v.Add(5.0);
        Assert.Equal(6.0, result[0], Tolerance);
        Assert.Equal(7.0, result[1], Tolerance);
        Assert.Equal(8.0, result[2], Tolerance);
    }

    [Fact]
    public void ScalarSubtract_HandComputed()
    {
        var v = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var result = v.Subtract(5.0);
        Assert.Equal(5.0, result[0], Tolerance);
        Assert.Equal(15.0, result[1], Tolerance);
        Assert.Equal(25.0, result[2], Tolerance);
    }

    // ============================
    // Pointwise Operation Tests
    // ============================

    [Fact]
    public void PointwiseMultiply_HandComputed()
    {
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var product = v1.PointwiseMultiply(v2);
        Assert.Equal(4.0, product[0], Tolerance);
        Assert.Equal(10.0, product[1], Tolerance);
        Assert.Equal(18.0, product[2], Tolerance);
    }

    [Fact]
    public void PointwiseDivide_HandComputed()
    {
        var v1 = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var v2 = new Vector<double>(new[] { 2.0, 5.0, 10.0 });
        var quotient = v1.PointwiseDivide(v2);
        Assert.Equal(5.0, quotient[0], Tolerance);
        Assert.Equal(4.0, quotient[1], Tolerance);
        Assert.Equal(3.0, quotient[2], Tolerance);
    }

    [Fact]
    public void PointwiseExp_Log_AreInverse()
    {
        // log(exp(x)) = x for positive x
        var v = new Vector<double>(new[] { 0.5, 1.0, 2.0 });
        var expV = v.PointwiseExp();
        // PointwiseExp returns VectorBase<T>, convert back to Vector<T> for PointwiseLog
        var vecExpV = new Vector<double>(new[] { expV[0], expV[1], expV[2] });
        var logExpV = vecExpV.PointwiseLog();
        for (int i = 0; i < v.Length; i++)
            Assert.Equal(v[i], logExpV[i], Tolerance);
    }

    [Fact]
    public void PointwiseExp_HandComputed()
    {
        // exp(0) = 1, exp(1) = e
        var v = new Vector<double>(new[] { 0.0, 1.0 });
        var result = v.PointwiseExp();
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(Math.E, result[1], Tolerance);
    }

    [Fact]
    public void PointwiseLog_HandComputed()
    {
        // log(1) = 0, log(e) = 1
        var v = new Vector<double>(new[] { 1.0, Math.E });
        var result = v.PointwiseLog();
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(1.0, result[1], Tolerance);
    }

    [Fact]
    public void PointwiseAbs_HandComputed()
    {
        var v = new Vector<double>(new[] { -3.0, 0.0, 5.0, -7.0 });
        var abs = v.PointwiseAbs();
        Assert.Equal(3.0, abs[0], Tolerance);
        Assert.Equal(0.0, abs[1], Tolerance);
        Assert.Equal(5.0, abs[2], Tolerance);
        Assert.Equal(7.0, abs[3], Tolerance);
    }

    [Fact]
    public void PointwiseSqrt_HandComputed()
    {
        var v = new Vector<double>(new[] { 4.0, 9.0, 16.0, 25.0 });
        var sqrt = v.PointwiseSqrt();
        Assert.Equal(2.0, sqrt[0], Tolerance);
        Assert.Equal(3.0, sqrt[1], Tolerance);
        Assert.Equal(4.0, sqrt[2], Tolerance);
        Assert.Equal(5.0, sqrt[3], Tolerance);
    }

    [Fact]
    public void PointwiseSign_HandComputed()
    {
        var v = new Vector<double>(new[] { -3.0, 0.0, 5.0 });
        var signs = v.PointwiseSign();
        Assert.Equal(-1.0, signs[0], Tolerance);
        Assert.Equal(0.0, signs[1], Tolerance);
        Assert.Equal(1.0, signs[2], Tolerance);
    }

    // ============================
    // Outer Product Tests
    // ============================

    [Fact]
    public void OuterProduct_HandComputed()
    {
        // [1,2] outer [3,4,5] = [[3,4,5],[6,8,10]]
        var v1 = new Vector<double>(new[] { 1.0, 2.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0, 5.0 });
        var outer = v1.OuterProduct(v2);

        Assert.Equal(2, outer.Rows);
        Assert.Equal(3, outer.Columns);
        Assert.Equal(3.0, outer[0, 0], Tolerance);
        Assert.Equal(4.0, outer[0, 1], Tolerance);
        Assert.Equal(5.0, outer[0, 2], Tolerance);
        Assert.Equal(6.0, outer[1, 0], Tolerance);
        Assert.Equal(8.0, outer[1, 1], Tolerance);
        Assert.Equal(10.0, outer[1, 2], Tolerance);
    }

    [Fact]
    public void OuterProduct_Rank1Matrix()
    {
        // Outer product always produces a rank-1 matrix
        // For rank-1: every 2x2 minor has determinant 0
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0 });
        var outer = v1.OuterProduct(v2);

        // Check 2x2 minor [0,1]x[0,1]: det = (1*5 - 4*2) = -3... wait
        // Actually det = a00*a11 - a01*a10 = (1*4)*(2*5) pattern
        // row0 = [4, 5], row1 = [8, 10]
        // det = 4*10 - 5*8 = 40 - 40 = 0 (rank 1!)
        double det = outer[0, 0] * outer[1, 1] - outer[0, 1] * outer[1, 0];
        Assert.Equal(0.0, det, Tolerance);
    }

    // ============================
    // Diagonal Matrix Tests
    // ============================

    [Fact]
    public void CreateDiagonal_HandComputed()
    {
        var v = new Vector<double>(new[] { 2.0, 5.0, 7.0 });
        var diag = v.CreateDiagonal();

        Assert.Equal(3, diag.Rows);
        Assert.Equal(3, diag.Columns);
        Assert.Equal(2.0, diag[0, 0], Tolerance);
        Assert.Equal(5.0, diag[1, 1], Tolerance);
        Assert.Equal(7.0, diag[2, 2], Tolerance);
        // Off-diagonal is zero
        Assert.Equal(0.0, diag[0, 1], Tolerance);
        Assert.Equal(0.0, diag[1, 0], Tolerance);
        Assert.Equal(0.0, diag[0, 2], Tolerance);
    }

    [Fact]
    public void ToDiagonalMatrix_EqualsCreateDiagonal()
    {
        var v = new Vector<double>(new[] { 1.0, 3.0, 5.0 });
        var d1 = v.CreateDiagonal();
        var d2 = v.ToDiagonalMatrix();

        for (int i = 0; i < d1.Rows; i++)
            for (int j = 0; j < d1.Columns; j++)
                Assert.Equal(d1[i, j], d2[i, j], Tolerance);
    }

    // ============================
    // Statistics Tests
    // ============================

    [Fact]
    public void Sum_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        Assert.Equal(15.0, v.Sum(), Tolerance);
    }

    [Fact]
    public void Average_HandComputed()
    {
        var v = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0, 10.0 });
        Assert.Equal(6.0, v.Average(), Tolerance);
    }

    [Fact]
    public void Max_Min_HandComputed()
    {
        var v = new Vector<double>(new[] { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0 });
        Assert.Equal(9.0, v.Max(), Tolerance);
        Assert.Equal(1.0, v.Min(), Tolerance);
    }

    [Fact]
    public void MaxIndex_MinIndex_HandComputed()
    {
        var v = new Vector<double>(new[] { 3.0, 1.0, 5.0, 2.0 });
        Assert.Equal(2, v.MaxIndex()); // 5.0 at index 2
        Assert.Equal(1, v.MinIndex()); // 1.0 at index 1
    }

    [Fact]
    public void AbsoluteMaximum_HandComputed()
    {
        var v = new Vector<double>(new[] { -7.0, 3.0, -2.0, 5.0 });
        Assert.Equal(7.0, v.AbsoluteMaximum(), Tolerance);
    }

    [Fact]
    public void StandardDeviation_HandComputed()
    {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 40/8 = 5
        // Sum of squared deviations = 9+1+1+1+0+0+4+16 = 32
        // Sample variance (Bessel's correction, n-1) = 32/7
        // Sample std = sqrt(32/7)
        var v = new Vector<double>(new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 });
        double std = v.StandardDeviation();
        double expected = Math.Sqrt(32.0 / 7.0);
        Assert.Equal(expected, std, Tolerance);
    }

    [Fact]
    public void Median_OddLength()
    {
        var v = new Vector<double>(new[] { 5.0, 1.0, 3.0 });
        Assert.Equal(3.0, v.Median(), Tolerance);
    }

    [Fact]
    public void Median_EvenLength()
    {
        // Sorted: 1, 2, 3, 4 -> median = (2+3)/2 = 2.5
        var v = new Vector<double>(new[] { 3.0, 1.0, 4.0, 2.0 });
        Assert.Equal(2.5, v.Median(), Tolerance);
    }

    // ============================
    // Euclidean Distance Tests
    // ============================

    [Fact]
    public void EuclideanDistance_SameVector_IsZero()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        Assert.Equal(0.0, v.EuclideanDistance(v), Tolerance);
    }

    [Fact]
    public void EuclideanDistance_HandComputed()
    {
        // d([1,0], [0,1]) = sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        var v1 = new Vector<double>(new[] { 1.0, 0.0 });
        var v2 = new Vector<double>(new[] { 0.0, 1.0 });
        Assert.Equal(Math.Sqrt(2.0), v1.EuclideanDistance(v2), Tolerance);
    }

    [Fact]
    public void EuclideanDistance_345Triangle()
    {
        // d([0,0], [3,4]) = 5
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0 });
        Assert.Equal(5.0, v1.EuclideanDistance(v2), Tolerance);
    }

    [Fact]
    public void EuclideanDistance_IsSymmetric()
    {
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        Assert.Equal(v1.EuclideanDistance(v2), v2.EuclideanDistance(v1), Tolerance);
    }

    [Fact]
    public void EuclideanDistance_EqualsSubtractNorm()
    {
        // d(v1, v2) = ||v1 - v2||
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        Assert.Equal(v1.Subtract(v2).Norm(), v1.EuclideanDistance(v2), Tolerance);
    }

    // ============================
    // Slice / SubVector Tests
    // ============================

    [Fact]
    public void Slice_HandComputed()
    {
        var v = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var sliced = v.Slice(1, 3);
        Assert.Equal(3, sliced.Length);
        Assert.Equal(20.0, sliced[0], Tolerance);
        Assert.Equal(30.0, sliced[1], Tolerance);
        Assert.Equal(40.0, sliced[2], Tolerance);
    }

    [Fact]
    public void SubVector_ByIndices()
    {
        var v = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var sub = v.SubVector(new[] { 0, 2, 4 });
        Assert.Equal(3, sub.Length);
        Assert.Equal(10.0, sub[0], Tolerance);
        Assert.Equal(30.0, sub[1], Tolerance);
        Assert.Equal(50.0, sub[2], Tolerance);
    }

    [Fact]
    public void Extract_GetsFirstNElements()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var extracted = v.Extract(3);
        Assert.Equal(3, extracted.Length);
        Assert.Equal(1.0, extracted[0], Tolerance);
        Assert.Equal(2.0, extracted[1], Tolerance);
        Assert.Equal(3.0, extracted[2], Tolerance);
    }

    // ============================
    // Argsort Tests
    // ============================

    [Fact]
    public void Argsort_HandComputed()
    {
        var v = new Vector<double>(new[] { 3.0, 1.0, 4.0, 1.0, 5.0 });
        var indices = v.Argsort();
        // Sorted order: 1.0(idx1), 1.0(idx3), 3.0(idx0), 4.0(idx2), 5.0(idx4)
        Assert.Equal(1, indices[0]);
        Assert.Equal(3, indices[1]);
        Assert.Equal(0, indices[2]);
        Assert.Equal(2, indices[3]);
        Assert.Equal(4, indices[4]);
    }

    [Fact]
    public void Argsort_AlreadySorted()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var indices = v.Argsort();
        for (int i = 0; i < indices.Length; i++)
            Assert.Equal(i, indices[i]);
    }

    // ============================
    // Repeat Tests
    // ============================

    [Fact]
    public void Repeat_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0 });
        var repeated = v.Repeat(3);
        Assert.Equal(6, repeated.Length);
        Assert.Equal(1.0, repeated[0], Tolerance);
        Assert.Equal(2.0, repeated[1], Tolerance);
        Assert.Equal(1.0, repeated[2], Tolerance);
        Assert.Equal(2.0, repeated[3], Tolerance);
        Assert.Equal(1.0, repeated[4], Tolerance);
        Assert.Equal(2.0, repeated[5], Tolerance);
    }

    // ============================
    // Matrix Shape Conversions
    // ============================

    [Fact]
    public void ToRowMatrix_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var rowMat = v.ToRowMatrix();
        Assert.Equal(1, rowMat.Rows);
        Assert.Equal(3, rowMat.Columns);
        Assert.Equal(1.0, rowMat[0, 0], Tolerance);
        Assert.Equal(2.0, rowMat[0, 1], Tolerance);
        Assert.Equal(3.0, rowMat[0, 2], Tolerance);
    }

    [Fact]
    public void ToColumnMatrix_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var colMat = v.ToColumnMatrix();
        Assert.Equal(3, colMat.Rows);
        Assert.Equal(1, colMat.Columns);
        Assert.Equal(1.0, colMat[0, 0], Tolerance);
        Assert.Equal(2.0, colMat[1, 0], Tolerance);
        Assert.Equal(3.0, colMat[2, 0], Tolerance);
    }

    [Fact]
    public void Reshape_HandComputed()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        var mat = v.Reshape(2, 3);
        Assert.Equal(2, mat.Rows);
        Assert.Equal(3, mat.Columns);
        // Row-major: [1,2,3; 4,5,6]
        Assert.Equal(1.0, mat[0, 0], Tolerance);
        Assert.Equal(2.0, mat[0, 1], Tolerance);
        Assert.Equal(3.0, mat[0, 2], Tolerance);
        Assert.Equal(4.0, mat[1, 0], Tolerance);
        Assert.Equal(5.0, mat[1, 1], Tolerance);
        Assert.Equal(6.0, mat[1, 2], Tolerance);
    }

    // ============================
    // Vector-Matrix Multiplication
    // ============================

    [Fact]
    public void VectorMatrixMultiply_HandComputed()
    {
        // [1,2,3] * [[1,4],[2,5],[3,6]] = [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var m = new Matrix<double>(3, 2);
        m[0, 0] = 1.0; m[0, 1] = 4.0;
        m[1, 0] = 2.0; m[1, 1] = 5.0;
        m[2, 0] = 3.0; m[2, 1] = 6.0;

        var result = v.Multiply(m);
        Assert.Equal(2, result.Length);
        Assert.Equal(14.0, result[0], Tolerance);
        Assert.Equal(32.0, result[1], Tolerance);
    }

    [Fact]
    public void VectorMatrixMultiply_IdentityMatrix()
    {
        // v * I = v (for square identity)
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var identity = new Matrix<double>(3, 3);
        identity[0, 0] = 1.0; identity[1, 1] = 1.0; identity[2, 2] = 1.0;

        var result = v.Multiply(identity);
        for (int i = 0; i < v.Length; i++)
            Assert.Equal(v[i], result[i], Tolerance);
    }

    // ============================
    // Transform Tests
    // ============================

    [Fact]
    public void Transform_SquareEachElement()
    {
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var squared = v.Transform(x => x * x);
        Assert.Equal(1.0, squared[0], Tolerance);
        Assert.Equal(4.0, squared[1], Tolerance);
        Assert.Equal(9.0, squared[2], Tolerance);
        Assert.Equal(16.0, squared[3], Tolerance);
    }

    // ============================
    // Maximum (clamp) Tests
    // ============================

    [Fact]
    public void Maximum_ClampsToScalar()
    {
        var v = new Vector<double>(new[] { -3.0, -1.0, 0.0, 2.0, 5.0 });
        var clamped = v.Maximum(0.0);
        Assert.Equal(0.0, clamped[0], Tolerance);
        Assert.Equal(0.0, clamped[1], Tolerance);
        Assert.Equal(0.0, clamped[2], Tolerance);
        Assert.Equal(2.0, clamped[3], Tolerance);
        Assert.Equal(5.0, clamped[4], Tolerance);
    }

    // ============================
    // Conversion Tests
    // ============================

    [Fact]
    public void ToVectorList_ToIntList_Roundtrip()
    {
        var indices = new[] { 0, 5, 10, 15 };
        var vectorList = indices.ToVectorList<double>();
        Assert.Equal(4, vectorList.Count);
        Assert.Equal(0.0, vectorList[0][0], Tolerance);
        Assert.Equal(5.0, vectorList[1][0], Tolerance);

        var intList = vectorList.ToIntList<double>();
        Assert.Equal(indices, intList.ToArray());
    }

    // ============================
    // PointwiseMultiplyInPlace Tests
    // ============================

    [Fact]
    public void PointwiseMultiplyInPlace_ModifiesVector()
    {
        var v1 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
        var v2 = new Vector<double>(new[] { 5.0, 6.0, 7.0 });
        v1.PointwiseMultiplyInPlace(v2);
        Assert.Equal(10.0, v1[0], Tolerance);
        Assert.Equal(18.0, v1[1], Tolerance);
        Assert.Equal(28.0, v1[2], Tolerance);
    }

    // ============================
    // Mathematical Identities / Properties
    // ============================

    [Fact]
    public void CauchySchwarzInequality()
    {
        // |v . w| <= ||v|| * ||w||
        var v = new Vector<double>(new[] { 1.0, -2.0, 3.0 });
        var w = new Vector<double>(new[] { 4.0, 5.0, -6.0 });
        double dot = Math.Abs(v.DotProduct(w));
        double product = v.Norm() * w.Norm();
        Assert.True(dot <= product + Tolerance);
    }

    [Fact]
    public void TriangleInequality()
    {
        // ||v + w|| <= ||v|| + ||w||
        var v = new Vector<double>(new[] { 1.0, -2.0, 3.0 });
        var w = new Vector<double>(new[] { 4.0, 5.0, -6.0 });
        double sumNorm = v.Add(w).Norm();
        double normSum = v.Norm() + w.Norm();
        Assert.True(sumNorm <= normSum + Tolerance);
    }

    [Fact]
    public void PythagoreanTheorem_OrthogonalVectors()
    {
        // If v . w = 0, then ||v + w||^2 = ||v||^2 + ||w||^2
        var v = new Vector<double>(new[] { 3.0, 0.0, 0.0 });
        var w = new Vector<double>(new[] { 0.0, 4.0, 0.0 });
        Assert.Equal(0.0, v.DotProduct(w), Tolerance); // verify orthogonal

        var sum = v.Add(w);
        double sumNormSq = sum.DotProduct(sum);
        double vNormSq = v.DotProduct(v);
        double wNormSq = w.DotProduct(w);
        Assert.Equal(vNormSq + wNormSq, sumNormSq, Tolerance);
    }

    [Fact]
    public void OuterProduct_DotProduct_Relationship()
    {
        // For vectors u, v: u^T v = trace(v * u^T) where v*u^T is outer product
        var u = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        double dot = u.DotProduct(v);

        var outer = v.OuterProduct(u);
        double trace = 0;
        for (int i = 0; i < outer.Rows && i < outer.Columns; i++)
            trace += outer[i, i];

        Assert.Equal(dot, trace, Tolerance);
    }

    [Fact]
    public void DiagonalMatrix_VectorMultiply_IsPointwiseMultiply()
    {
        // D * v = diag(d) . v (pointwise)
        var d = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
        var v = new Vector<double>(new[] { 5.0, 6.0, 7.0 });
        var diagMatrix = d.CreateDiagonal();

        // Vector * Matrix (v * D) should give pointwise multiply
        var result = v.Multiply(diagMatrix);
        var expected = v.PointwiseMultiply(d);

        for (int i = 0; i < v.Length; i++)
            Assert.Equal(expected[i], result[i], Tolerance);
    }
}
