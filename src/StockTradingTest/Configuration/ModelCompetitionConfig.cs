namespace StockTradingTest.Configuration
{
    public class ModelCompetitionConfig
    {
        public int NumberOfRounds { get; set; } = 3;
        public double EliminationPercentage { get; set; } = 0.5;
        public double ValidationDataSplit { get; set; } = 0.2;
        public double TestDataSplit { get; set; } = 0.2;
        public int LookbackPeriod { get; set; } = 60;
        public int PredictionHorizon { get; set; } = 5;
    }
}