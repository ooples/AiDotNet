namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public sealed class OperatorDataset2D
    {
        public string OperatorName { get; set; } = string.Empty;
        public int GridSize { get; set; }
        public int SampleCount { get; set; }
        public double[,,] Inputs { get; set; } = new double[0, 0, 0];
        public double[,,] Outputs { get; set; } = new double[0, 0, 0];
    }
}
