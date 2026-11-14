using System;
using System.Threading.Tasks;
using AiDotNet.DecompositionMethods.MatrixDecomposition;
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
    public void ComputeGradients_Half_SmallGradientsSurviveTheAverage()
    {
        // 8192 rows of x = 1 predicting 0 against a target of -1. The MSE gradient is 2 * 1 / 8192 = 2^-12 per row,
        // and (1/n) X^T g = 2^-12 as well. Dividing each row's 2^-12 by 8192 first gives 2^-25, which is 0 in Half,
        // so the whole gradient came back as 0.
        const int rows = 8192;
        var x = new Matrix<Half>(rows, 1);
        var target = new Vector<Half>(rows);
        for (int i = 0; i < rows; i++)
        {
            x[i, 0] = (Half)1;
            target[i] = (Half)(-1);
        }

        using var model = new VectorModel<Half>(new Vector<Half>(new[] { (Half)0 }));

        var gradients = model.ComputeGradients(x, target);

        double expected = 1.0 / 4096;
        Assert.Equal(2, gradients.Length);
        Assert.Equal(expected, (double)gradients[0], expected * 0.02);
        Assert.Equal(expected, (double)gradients[1], expected * 0.02);
    }

    [Fact]
    public void Train_Half_SingularValueBelowHalfPrecision_IsTruncated()
    {
        // X = [diag(1, 1e-4); -diag(1, 1e-4)] has orthogonal, already-centred columns, so its singular values are
        // known exactly: sqrt(2) and sqrt(2) * 1e-4. X^T X = diag(2, 2e-8) is singular in Half (2e-8 rounds to
        // 0), so training takes the SVD path. The numpy/LAPACK cutoff max(m, n) * eps * sigma_max = 4 * 2^-10 *
        // sqrt(2) = 5.5e-3 drops sigma_2, so the fit is the rank-1 minimum-norm solution (1, 0). A 1e-12
        // tolerance rounds to 0 in Half and keeps sigma_2, which fits y = x1 + x2 exactly as (1, 1).
        var x = new Matrix<Half>(new[,]
        {
            { (Half)1, (Half)0 },
            { (Half)0, (Half)1e-4 },
            { (Half)(-1), (Half)0 },
            { (Half)0, (Half)(-1e-4) }
        });
        var y = new Vector<Half>(new[] { (Half)1, (Half)1e-4, (Half)(-1), (Half)(-1e-4) });

        // Precondition: the engine resolves sigma_2 as non-zero and below the cutoff, so the result depends
        // only on the tolerance, not on SVD rounding.
        var singular = new SvdDecomposition<Half>(x).S;
        double sigma1 = Math.Max((double)singular[0], (double)singular[1]);
        double sigma2 = Math.Min((double)singular[0], (double)singular[1]);
        Assert.True(sigma2 > 0, "sigma_2 must be non-zero for the fixture to discriminate.");
        Assert.True(sigma2 < 4 * (1.0 / 1024) * sigma1, "sigma_2 must be below the eps(T) cutoff.");

        using var model = new VectorModel<Half>(new Vector<Half>(2));
        model.Train(x, y);

        Assert.Equal(1.0, (double)model.Coefficients[0], 1e-2);
        Assert.Equal(0.0, (double)model.Coefficients[1], 1e-2);
    }
}
