namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public sealed class PdeBenchmarkResult
    {
        public string EquationName { get; set; } = string.Empty;
        public int SpatialPoints { get; set; }
        public int TimeSteps { get; set; }
        public double FinalTime { get; set; }
        public double L2Error { get; set; }
        public double MaxError { get; set; }
    }
}
