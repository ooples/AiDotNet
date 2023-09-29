using AiDotNet.Enums;

namespace AiDotNet.Models;

public class MultipleRegressionOptions : RegressionOptions
{
    public MatrixDecomposition MatrixDecomposition { get; set; } = MatrixDecomposition.Cholesky;

    public bool UseIntercept { get; set; } = false;

    public MatrixLayout MatrixLayout { get; set; } = MatrixLayout.ColumnArrays;
}