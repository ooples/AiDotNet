namespace AiDotNet.Models;

public class RegularizationOptions
{
    public RegularizationType Type { get; set; } = RegularizationType.None;
    public double Strength { get; set; } = 0.0;
    public double L1Ratio { get; set; } = 0.5;
}