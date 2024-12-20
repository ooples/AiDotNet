namespace AiDotNet.Models;

public class AutocorrelationFitDetectorOptions
{
    public double StrongPositiveAutocorrelationThreshold { get; set; } = 1.0;
    public double StrongNegativeAutocorrelationThreshold { get; set; } = 3.0;
    public double NoAutocorrelationLowerBound { get; set; } = 1.5;
    public double NoAutocorrelationUpperBound { get; set; } = 2.5;
}