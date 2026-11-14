using System;
using System.Threading.Tasks;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Reference behaviour for <see cref="VectorModel{T}"/> and the <see cref="ModelBase{T, TInput, TOutput}"/>
/// serialization it inherits, pinned where the engine-op conversion (#2490) fixed a wrong result.
/// </summary>
public class VectorModelReferenceTests
{
    [Fact]
    public void SerializeDeserialize_Half_RoundTripsCoefficients()
    {
        // Half is not IConvertible, so the former Convert.ToDouble serialization threw InvalidCastException.
        var coefficients = new Vector<Half>(new[] { (Half)1.5, (Half)(-2.25), (Half)0.125 });
        using var source = new VectorModel<Half>(coefficients);

        byte[] payload = source.Serialize();
        using var restored = new VectorModel<Half>(new Vector<Half>(3));
        restored.Deserialize(payload);

        Assert.Equal(3, restored.Coefficients.Length);
        Assert.Equal((Half)1.5, restored.Coefficients[0]);
        Assert.Equal((Half)(-2.25), restored.Coefficients[1]);
        Assert.Equal((Half)0.125, restored.Coefficients[2]);
    }

    [Fact]
    public async Task GetPartialDependenceAsync_OnePointGrid_IsTheGridStart()
    {
        // The grid spans [-1, 1]; a one-point grid is its start, -1 (the former i / (n - 1) step gave 0 / 0 = NaN).
        using var model = new VectorModel<double>(new Vector<double>(new[] { 2.0, 3.0 }));

        var pd = await model.GetPartialDependenceAsync(new Vector<int>(new[] { 1 }), gridResolution: 1);

        Assert.Equal(-1.0, pd.GridValues[1][0]);
        Assert.Equal(1, pd.PartialDependenceValues.Columns);
        Assert.Equal(-3.0, pd.PartialDependenceValues[0, 0], 1e-12);
    }

    [Fact]
    public void Train_Half_SumsBeyondHalfRange_FitsTheLine()
    {
        // 64 rows alternating x = 1024 and 1026 with y = 2x. The column sums (65600 and 131200) overflow Half
        // (65504), so summing before dividing by the count made the means infinite; the fit is slope 2, intercept 0.
        const int rows = 64;
        var x = new Matrix<Half>(rows, 1);
        var y = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            double value = i % 2 == 0 ? 1024 : 1026;
            x[i, 0] = (Half)value;
            y[i] = (Half)(2 * value);
        }

        using var model = new VectorModel<Half>(new Vector<Half>(1));
        model.Train(x, y);

        Assert.Equal(2.0, (double)model.Coefficients[0], 1e-3);
        Assert.Equal(0.0, (double)model.Intercept, 1e-3);
    }

    [Fact]
    public void ComputeGradients_Half_SumsBeyondHalfRange_AreFinite()
    {
        // 64 rows of x = 2048 predicting 2048 against a target of 2028. The MSE gradient is 2 * 20 / 64 = 0.625 per
        // row, so X^T g = 81920 overflows Half before the 1/n; dividing first gives 1280 and 0.625.
        const int rows = 64;
        var x = new Matrix<Half>(rows, 1);
        var target = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            x[i, 0] = (Half)2048;
            target[i] = (Half)2028;
        }

        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)1 }));

        var gradients = model.ComputeGradients(x, target);

        Assert.Equal(2, gradients.Length);
        Assert.Equal(1280.0, (double)gradients[0], 1.0);
        Assert.Equal(0.625, (double)gradients[1], 1e-3);
    }

    [Fact]
    public void Train_Half_SingularValueAtHalfPrecision_IsTruncated()
    {
        // The second column equals the first up to 2^-8 in two rows, so X^T X is singular in Half and training
        // takes the SVD path. X's singular values are 2.83 and 2.76e-3; the second is about one Half epsilon
        // relative to the first, below the numpy/LAPACK cutoff max(m, n) * eps * sigma_max = 0.011. Truncating it
        // gives the rank-1 minimum-norm fit (0.998, 1.000). A 1e-12 tolerance rounded to 0 in Half and kept it.
        const double d = 1.0 / 256;
        var x = new Matrix<Half>(new[,]
        {
            { (Half)1, (Half)1 },
            { (Half)(-1), (Half)(-1) },
            { (Half)1, (Half)(1 + d) },
            { (Half)(-1), (Half)(-1 - d) }
        });
        var y = new Vector<Half>(new[] { (Half)2, (Half)(-2), (Half)2, (Half)(-2) });

        using var model = new VectorModel<Half>(new Vector<Half>(2));
        model.Train(x, y);

        Assert.Equal(0.998, (double)model.Coefficients[0], 0.02);
        Assert.Equal(1.0, (double)model.Coefficients[1], 0.02);
    }
}
