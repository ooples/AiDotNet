namespace AiDotNet.Models.Options;

public class RadialBasisFunctionOptions : NonLinearRegressionOptions
{
    public int NumberOfCenters { get; set; } = 10;
    public int? Seed { get; set; } = null;
}