using AiDotNet;
using AiDotNet.Factories;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.TimeSeries;

namespace AiDotNetTestConsole.Examples;

/// <summary>
/// Demonstrates how to use time series models in AiDotNet for forecasting stock prices.
/// </summary>
public class TimeSeriesExample
{
    public void RunExample()
    {
        Console.WriteLine("Time Series Example - Stock Price Forecasting");
        Console.WriteLine("===========================================\n");

        try
        {
            // Sample time series data (e.g., daily stock prices)
            double[] dates = [.. Enumerable.Range(0, 100).Select(i => (double)i)];
            double[] prices = new double[100];

            // Generate synthetic stock price data with trend and seasonality
            for (int i = 0; i < 100; i++)
            {
                // Trend component
                double trend = 100 + 0.5 * i;

                // Seasonal component (weekly pattern)
                double seasonal = 5 * Math.Sin(2 * Math.PI * i / 7);

                // Random noise
                double noise = new Random(42 + i).NextDouble() * 10 - 5;

                prices[i] = trend + seasonal + noise;
            }

            // Print a few sample data points
            Console.WriteLine("Sample data points:");
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine($"Day {i}: ${prices[i]:F2}");
            }
            Console.WriteLine("...");

            // Convert to Matrix/Vector format
            var timeFeatures = new Matrix<double>([.. dates.Select(d => new[] { d })]);
            var priceVector = new Vector<double>(prices);

            Console.WriteLine("Data prepared. Starting model training...");

            var model = TimeSeriesModelFactory<double, Matrix<double>, Vector<double>>.CreateProphetModel(timeFeatures, priceVector, 30);
            var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

            // Configure optimizer with appropriate settings for trend preservation
            var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 500,
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, adamOptions);

            // Build the time series model
            var modelResult = modelBuilder
                .ConfigureOptimizer(optimizer)
                .Build(timeFeatures, priceVector);

            Console.WriteLine("Model trained successfully!");

            // Generate future dates for forecasting
            var futureDates = Enumerable.Range(100, 30).Select(i => (double)i).ToArray();
            var futureFeatures = new Matrix<double>([.. futureDates.Select(d => new[] { d })]);

            // Make forecast
            var forecast = modelBuilder.Predict(futureFeatures, modelResult);

            Console.WriteLine("\nForecast for the next 30 days:");
            for (int i = 0; i < Math.Min(10, forecast.Length); i++)
            {
                Console.WriteLine($"Day {futureDates[i]}: ${forecast[i]:F2}");
            }
            Console.WriteLine("...");

            // Save model for later use
            string modelPath = "stock_forecast_model.bin";
            modelBuilder.SaveModel(modelResult, modelPath);
            Console.WriteLine($"\nModel saved to {modelPath}");

            // Load the model back
            var loadedModel = modelBuilder.LoadModel(modelPath);
            var forecast2 = modelBuilder.Predict(futureFeatures, loadedModel);

            Console.WriteLine("\nForecast using loaded model (first 5 days):");
            for (int i = 0; i < Math.Min(5, forecast2.Length); i++)
            {
                Console.WriteLine($"Day {futureDates[i]}: ${forecast2[i]:F2}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    // Helper method to generate evenly spaced changepoints
    private List<double> GetChangepoints(int count, double start, double end)
    {
        var changepoints = new List<double>();
        double step = (end - start) / (count + 1);

        for (int i = 1; i <= count; i++)
        {
            changepoints.Add(start + i * step);
        }

        return changepoints;
    }
}