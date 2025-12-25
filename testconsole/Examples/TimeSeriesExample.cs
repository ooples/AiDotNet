using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TimeSeries;

namespace AiDotNetTestConsole.Examples;

/// <summary>
/// Demonstrates how to use time series models in AiDotNet for forecasting stock prices.
/// </summary>
public class TimeSeriesExample
{
    public async Task RunExample()
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
                double noise = RandomHelper.CreateSeededRandom(42 + i).NextDouble() * 10 - 5;

                prices[i] = trend + seasonal + noise;
            }

            // Convert to Matrix/Vector format
            var timeFeatures = new Matrix<double>(dates.Select(d => new[] { d }).ToArray());
            var priceVector = new Vector<double>(prices);

            Console.WriteLine("Data prepared. Starting model training...");

            // Create and configure the model builder
            var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

            // Configure optimizer
            var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 1000
            };
            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, adamOptions);

            // Configure time series model (e.g., Prophet-like model)
            var timeSeriesOptions = new ProphetOptions<double, Matrix<double>, Vector<double>>
            {
                SeasonalPeriods = [7],  // Weekly seasonality (using List<int> instead of double[])
                ChangePointPriorScale = 0.05,           // Control flexibility of the trend (this is the correct property)
                ForecastHorizon = 30                    // Forecast 30 days ahead
            };

            // Build the time series model
            var model = await modelBuilder
                .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(timeFeatures, priceVector))
                .ConfigureOptimizer(optimizer)
                .ConfigureModel(new ProphetModel<double, Matrix<double>, Vector<double>>(timeSeriesOptions))
                .BuildAsync();

            Console.WriteLine("Model trained successfully!");

            // Generate future dates for forecasting
            var futureDates = Enumerable.Range(100, 30).Select(i => (double)i).ToArray();
            var futureFeatures = new Matrix<double>([.. futureDates.Select(d => new[] { d })]);

            // Make forecast
            var forecast = modelBuilder.Predict(futureFeatures, model);

            Console.WriteLine("\nForecast for the next 30 days:");
            for (int i = 0; i < Math.Min(10, forecast.Length); i++)
            {
                Console.WriteLine($"Day {futureDates[i]}: ${forecast[i]:F2}");
            }
            Console.WriteLine("...");

            // Save model for later use
            string modelPath = "stock_forecast_model.bin";
            modelBuilder.SaveModel(model, modelPath);
            Console.WriteLine($"\nModel saved to {modelPath}");

            // Load the model back
            var loadedModel = modelBuilder.LoadModel(modelPath);
            var forecast2 = modelBuilder.Predict(futureFeatures, loadedModel);

            Console.WriteLine("\nForecast using loaded model (first 5 days):");
            for (int i = 0; i < Math.Min(5, forecast2.Length); i++)
            {
                Console.WriteLine($"Day {futureDates[i]}: ${forecast2[i]:F2}");
            }

            // Visualize components if available
            if (model.Model is ProphetModel<double, Matrix<double>, Vector<double>> prophetModel)
            {
                Console.WriteLine("\nModel Components Analysis available in ProphetModel");
                Console.WriteLine("(Implementation would show trend, seasonality components)");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
