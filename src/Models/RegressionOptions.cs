namespace AiDotNet.Models;

public class RegressionOptions
{
    public MatrixDecomposition DecompositionMethod { get; set; } = MatrixDecomposition.Normal;
    public bool UseIntercept { get; set; } = true;
}