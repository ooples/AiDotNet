namespace AiDotNet.Models;

public class VIFFitDetectorOptions
{
    public double SevereMulticollinearityThreshold { get; set; } = 10.0;
    public double ModerateMulticollinearityThreshold { get; set; } = 5.0;
    public double GoodFitThreshold { get; set; } = 0.7;
    public MetricType PrimaryMetric { get; set; } = MetricType.R2;
}