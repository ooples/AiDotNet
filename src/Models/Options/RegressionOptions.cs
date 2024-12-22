namespace AiDotNet.Models.Options;

public class RegressionOptions<T> : ModelOptions
{
    public IMatrixDecomposition<T>? DecompositionMethod { get; set; }
    public bool UseIntercept { get; set; } = true;
}