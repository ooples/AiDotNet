namespace AiDotNet.Models.Options;

public class PrecisionRecallCurveFitDetectorOptions
{
    public double AreaUnderCurveThreshold { get; set; } = 0.7;
    public double F1ScoreThreshold { get; set; } = 0.6;
    public double AucWeight { get; set; } = 0.6;
    public double F1ScoreWeight { get; set; } = 0.4;
}