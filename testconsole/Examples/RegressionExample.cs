using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;

namespace AiDotNetTestConsole.Examples;

/// <summary>
/// Demonstrates how to use regression models in AiDotNet for predicting house prices.
/// </summary>
public class RegressionExample
{
    public async Task RunExample()
    {
        Console.WriteLine("Multiple Regression Example - House Price Prediction");
        Console.WriteLine("==================================================\n");

        try
        {
            // Sample data: house features (size, bedrooms, age) and prices
            double[,] houseFeaturesArray = new double[,]
            {
                { 1500, 3, 10 },  // 1500 sq ft, 3 bedrooms, 10 years old
                { 2200, 4, 5 },   // 2200 sq ft, 4 bedrooms, 5 years old
                { 1200, 2, 15 },  // 1200 sq ft, 2 bedrooms, 15 years old
                { 3000, 5, 2 },   // 3000 sq ft, 5 bedrooms, 2 years old
                { 1800, 3, 8 },   // 1800 sq ft, 3 bedrooms, 8 years old
                { 2500, 4, 3 },   // 2500 sq ft, 4 bedrooms, 3 years old
                { 1600, 3, 12 },  // 1600 sq ft, 3 bedrooms, 12 years old
                { 2800, 5, 4 }    // 2800 sq ft, 5 bedrooms, 4 years old
            };
            double[] housePricesArray = new double[]
            {
                250000,  // $250,000
                350000,  // $350,000
                180000,  // $180,000
                450000,  // $450,000
                275000,  // $275,000
                380000,  // $380,000
                240000,  // $240,000
                425000   // $425,000
            };

            // Convert raw arrays to Matrix and Vector
            var houseFeatures = new Matrix<double>(houseFeaturesArray);
            var housePrices = new Vector<double>(housePricesArray);

            Console.WriteLine("Data prepared. Starting model training...");

            // Create and configure the model builder
            var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

            // Configure Adam optimizer for training
            var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 1000,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8
            };

            var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, adamOptions);

            // Use MultipleRegression since we have multiple input features
            var regressionOptions = new RegressionOptions<double>
            {
                UseIntercept = true // Include intercept term (bias)
            };

            // Build a multiple regression model
            var model = await modelBuilder
                .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(houseFeatures, housePrices))
                .ConfigureOptimizer(optimizer)
                .ConfigureModel(new MultipleRegression<double>(regressionOptions))
                .BuildAsync();

            Console.WriteLine("Model trained successfully!");

            // Test with new data
            var newHouseArray = new double[,] { { 2000, 4, 7 } };  // 2000 sq ft, 4 bedrooms, 7 years old
            var newHouse = new Matrix<double>(newHouseArray);

            // Make prediction
            var predictedPrice = modelBuilder.Predict(newHouse, model);

            Console.WriteLine($"Predicted price for the test house: ${predictedPrice[0]:N0}");

            // Save model for later use
            string modelPath = "house_price_model.bin";
            modelBuilder.SaveModel(model, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");

            // Load the model back
            var loadedModel = modelBuilder.LoadModel(modelPath);
            var predictedPrice2 = modelBuilder.Predict(newHouse, loadedModel);

            Console.WriteLine($"Predicted price using loaded model: ${predictedPrice2[0]:N0}");

            // Print model coefficients if available
            if (model.Model != null)
            {
                Console.WriteLine("\nModel Details:");

                if (model.Model.GetType().GetProperty("Coefficients") != null)
                {
                    var coefficients = model.Model.GetType().GetProperty("Coefficients")?.GetValue(model.Model);
                    Console.WriteLine($"Coefficients: {coefficients}");
                }

                if (model.Model.GetType().GetProperty("Intercept") != null)
                {
                    var intercept = model.Model.GetType().GetProperty("Intercept")?.GetValue(model.Model);
                    Console.WriteLine($"Intercept: {intercept}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
