global using AiDotNet.WindowFunctions;

namespace AiDotNet.Models.Options
{
    public class SpectralAnalysisOptions<T> : TimeSeriesRegressionOptions<T>
    {
        public int NFFT { get; set; } = 512; // Default value, typically a power of 2
        public bool UseWindowFunction { get; set; } = true;
        public IWindowFunction<T> WindowFunction { get; set; } = new HanningWindow<T>();
        public int OverlapPercentage { get; set; } = 50; // Default 50% overlap
    }
}