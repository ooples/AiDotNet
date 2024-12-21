namespace AiDotNet.Models;

public class CalibratedProbabilityFitDetectorOptions
{
    public int NumCalibrationBins { get; set; } = 10;
    public double GoodFitThreshold { get; set; } = 0.05;
    public double OverfitThreshold { get; set; } = 0.15;
    public double MaxCalibrationError { get; set; } = 1.0;
}