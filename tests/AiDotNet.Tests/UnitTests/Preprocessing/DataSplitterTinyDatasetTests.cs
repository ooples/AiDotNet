using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Pins the tiny-dataset breakage in <see cref="DataSplitter"/> that surfaced
/// through <c>AiModelBuilder.BuildAsync</c>: integer <c>floor()</c> of the 0.7 /
/// 0.15 ratios starves a partition to zero rows on small inputs. For n=1 the
/// TRAINING set comes out empty (model never trains, lazy layers stay unseeded);
/// for any n &lt; 7 the validation set is empty. The split must always produce a
/// usable training set, a partition total that sums back to n, and must never
/// throw — regardless of how few rows the caller supplies.
/// </summary>
public class DataSplitterTinyDatasetTests
{
    private const int NumFeatures = 3;

    private static (Matrix<double> x, Vector<double> y) MakeMatrix(int n)
    {
        var x = new Matrix<double>(n, NumFeatures);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int f = 0; f < NumFeatures; f++)
                x[i, f] = i * NumFeatures + f;
            y[i] = i;
        }
        return (x, y);
    }

    private static (Tensor<double> x, Tensor<double> y) MakeTensor(int n)
    {
        var x = new Tensor<double>([n, NumFeatures]);
        var y = new Tensor<double>([n, 1]);
        for (int i = 0; i < n; i++)
        {
            for (int f = 0; f < NumFeatures; f++)
                x[i, f] = i * NumFeatures + f;
            y[i, 0] = i;
        }
        return (x, y);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(10)]
    public void Split_Matrix_TinyDataset_KeepsTrainNonEmpty_AndSumsToN(int n)
    {
        var (x, y) = MakeMatrix(n);

        var (xTrain, yTrain, xVal, yVal, xTest, yTest) =
            DataSplitter.Split<double, Matrix<double>, Vector<double>>(x, y);

        Assert.True(xTrain.Rows >= 1, $"n={n}: training set is empty — model cannot train.");
        Assert.Equal(xTrain.Rows, yTrain.Length);
        Assert.Equal(xVal.Rows, yVal.Length);
        Assert.Equal(xTest.Rows, yTest.Length);
        Assert.True(xVal.Rows >= 0 && xTest.Rows >= 0);
        Assert.Equal(n, xTrain.Rows + xVal.Rows + xTest.Rows);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(10)]
    public void Split_Tensor_TinyDataset_KeepsTrainNonEmpty_AndSumsToN(int n)
    {
        var (x, y) = MakeTensor(n);

        var (xTrain, yTrain, xVal, yVal, xTest, yTest) =
            DataSplitter.Split<double, Tensor<double>, Tensor<double>>(x, y);

        Assert.True(xTrain.Shape[0] >= 1, $"n={n}: training set is empty — model cannot train.");
        Assert.Equal(xTrain.Shape[0], yTrain.Shape[0]);
        Assert.Equal(xVal.Shape[0], yVal.Shape[0]);
        Assert.Equal(xTest.Shape[0], yTest.Shape[0]);
        Assert.Equal(n, xTrain.Shape[0] + xVal.Shape[0] + xTest.Shape[0]);
    }

    // With enough rows to afford one each, every partition must be non-empty so
    // validation-based model selection and the held-out test metric are real.
    [Theory]
    [InlineData(3)]
    [InlineData(8)]
    [InlineData(20)]
    public void Split_Matrix_AffordableDataset_FillsEveryPartition(int n)
    {
        var (x, y) = MakeMatrix(n);

        var (xTrain, _, xVal, _, xTest, _) =
            DataSplitter.Split<double, Matrix<double>, Vector<double>>(x, y);

        Assert.True(xTrain.Rows >= 1, $"n={n}: empty training set.");
        Assert.True(xVal.Rows >= 1, $"n={n}: empty validation set — model selection has nothing to score.");
        Assert.True(xTest.Rows >= 1, $"n={n}: empty test set — no held-out evaluation.");
        Assert.Equal(n, xTrain.Rows + xVal.Rows + xTest.Rows);
    }
}
