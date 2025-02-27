namespace AiDotNet.Models.Options;

public class OrthogonalRegressionOptions<T> : RegressionOptions<T>
{
    public double Tolerance { get; set; } = 1e-6;
    public int MaxIterations { get; set; } = 100;
    public bool ScaleVariables { get; set; } = true;
}