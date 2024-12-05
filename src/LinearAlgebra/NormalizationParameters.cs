namespace AiDotNet.LinearAlgebra;

public record NormalizationParameters
{
    public NormalizationMethod Method { get; set; }
    public double Min { get; set; }
    public double Max { get; set; }
    public double Mean { get; set; }
    public double StdDev { get; set; }
    public double Scale { get; set; }
    public double Shift { get; set; }
    public List<double> Bins { get; set; } = [];
    public double Median { get; set; }
    public double IQR { get; set; }
    public double P { get; set; }
}
