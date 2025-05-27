namespace StockTradingTest.Models
{
    public class ModelMetrics
    {
        public double MeanAbsoluteError { get; set; }
        public double MeanSquaredError { get; set; }
        public double RootMeanSquaredError { get; set; }
        public double R2Score { get; set; }
        public double DirectionalAccuracy { get; set; }
        public TimeSpan TrainingTime { get; set; }
        public TimeSpan InferenceTime { get; set; }
        public int TrainingDataPoints { get; set; }
        public int TestingDataPoints { get; set; }
    }
}