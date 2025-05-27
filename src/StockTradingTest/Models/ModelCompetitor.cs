using AiDotNet.Interfaces;

namespace StockTradingTest.Models
{
    public class ModelCompetitor
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Name { get; set; } = string.Empty;
        public IFullModel<double, Vector<double>, Vector<double>> Model { get; set; } = null!;
        public ModelType Type { get; set; }
        public ModelMetrics TrainingMetrics { get; set; } = new ModelMetrics();
        public ModelMetrics ValidationMetrics { get; set; } = new ModelMetrics();
        public List<TournamentResult> TournamentResults { get; set; } = new List<TournamentResult>();
        public bool IsEliminated { get; set; } = false;

        public double AverageProfitPercent => TournamentResults.Count > 0 
            ? TournamentResults.Average(r => (double)r.FinalProfitLossPercent) 
            : 0;

        public double Sharpe => TournamentResults.Count > 0 
            ? CalculateSharpe() 
            : 0;

        private double CalculateSharpe()
        {
            var returns = TournamentResults.Select(r => (double)r.FinalProfitLossPercent).ToArray();
            var meanReturn = returns.Average();
            var stdDev = Math.Sqrt(returns.Select(r => Math.Pow(r - meanReturn, 2)).Sum() / returns.Length);
            return stdDev == 0 ? 0 : meanReturn / stdDev;
        }
    }

    public enum ModelType
    {
        NeuralNetwork,
        RandomForest,
        GradientBoosting,
        SupportVectorMachine,
        GaussianProcess,
        ARIMA,
        LSTM,
        Transformer
    }
}