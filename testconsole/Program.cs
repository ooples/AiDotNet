using AiDotNet.Examples;
using AiDotNet.Prototypes;
using AiDotNetTestConsole.Examples;

namespace AiDotNetTestConsole;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("AiDotNet Examples");
        Console.WriteLine("================");
        Console.WriteLine("1. Neural Network Example (Classification and Regression)");
        Console.WriteLine("2. Multiple Regression Example (House Price Prediction)");
        Console.WriteLine("3. Time Series Example (Stock Price Forecasting)");
        Console.WriteLine("4. Enhanced Regression Example (Real Estate Analysis)");
        Console.WriteLine("5. Enhanced Neural Network Example (Customer Churn Prediction)");
        Console.WriteLine("6. Enhanced Time Series Example (Energy Demand Forecasting)");
        Console.WriteLine("7. Phase A GPU Acceleration Integration Tests");
        Console.WriteLine("8. DeconvolutionalLayer Test");
        Console.WriteLine("0. Exit");
        Console.WriteLine();
        Console.Write("Select an example to run (0-8): ");

        if (int.TryParse(Console.ReadLine(), out int choice))
        {
            Console.Clear();

            switch (choice)
            {
                case 0:
                    Console.WriteLine("Exiting...");
                    break;
                case 1:
                    var nnExample = new NeuralNetworkExample();
                    nnExample.RunExample();
                    break;
                case 2:
                    var regExample = new RegressionExample();
                    await regExample.RunExample();
                    break;
                case 3:
                    var tsExample = new TimeSeriesExample();
                    await tsExample.RunExample();
                    break;
                case 4:
                    var enhancedRegExample = new EnhancedRegressionExample();
                    await enhancedRegExample.RunExample();
                    break;
                case 5:
                    var enhancedNNExample = new EnhancedNeuralNetworkExample();
                    enhancedNNExample.RunExample();
                    break;
                case 6:
                    var enhancedTSExample = new EnhancedTimeSeriesExample();
                    await enhancedTSExample.RunExample();
                    break;
                case 7:
                    Console.WriteLine("Running Phase A GPU Acceleration Integration Tests...");
                    Console.WriteLine();
                    PrototypeIntegrationTests.RunAll();
                    break;
                case 8:
                    DeconvTest.Run();
                    break;
                default:
                    Console.WriteLine("Invalid choice. Please select a number between 0 and 8.");
                    break;
            }
        }
        else
        {
            Console.WriteLine("Invalid input. Please enter a number.");
        }

        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
}
