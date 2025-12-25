namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public sealed class OperatorBenchmarkResult
    {
        public string OperatorName { get; set; } = string.Empty;
        public int SpatialPoints { get; set; }
        public int SampleCount { get; set; }
        public double Mse { get; set; }
        public double L2Error { get; set; }
        public double RelativeL2Error { get; set; }
        public double MaxError { get; set; }
    }
}
