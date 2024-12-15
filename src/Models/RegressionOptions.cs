namespace AiDotNet.Models;

public class RegressionOptions<T>
{
    public IMatrixDecomposition<T>? DecompositionMethod { get; set; }
    public bool UseIntercept { get; set; } = true;
}