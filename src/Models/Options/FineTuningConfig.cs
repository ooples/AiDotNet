namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Fine-tuning configuration
    /// </summary>
    public class FineTuningConfig
    {
        public int Epochs { get; set; } = 3;
        public double LearningRate { get; set; } = 1e-5;
        public int BatchSize { get; set; } = 8;
        public double WeightDecay { get; set; } = 0.01;
        public int WarmupSteps { get; set; } = 500;
        public string OptimizerType { get; set; } = "AdamW";
        public bool UseMixedPrecision { get; set; } = true;
        public int GradientAccumulationSteps { get; set; } = 1;
        public double ValidationSplit { get; set; } = 0.1;
    }
}