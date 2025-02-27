namespace AiDotNet.Models.Options;

public class KNearestNeighborsOptions : NonLinearRegressionOptions
{
    public int K { get; set; } = 5;
}