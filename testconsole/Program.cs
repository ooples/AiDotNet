using AiDotNet.Prototypes;
using AiDotNetTestConsole.Examples;
using AiDotNetTestConsole.Examples.MetaLearning;

namespace AiDotNetTestConsole;

class Program
{
    // Bottleneck-profiling entry points: each key matches a CLI arg
    // passed under dotnet-trace to collect CPU samples against a
    // single model at research-paper defaults. Adding a new profile
    // now means one line here instead of a new `if (args[0] == …)`
    // block below.
    private static readonly Dictionary<string, Action> ProfileModes = new()
    {
        ["chronosbolt-profile"] = ChronosBoltProfile.Run,
        ["timemoe-profile"]     = TimeMoEProfile.Run,
        ["timesfm-profile"]     = TimesFMProfile.Run,
        ["moment-profile"]      = MOMENTProfile.Run,
        ["lstmvae-profile"]     = LSTMVAEProfile.Run,
        ["deepant-profile"]     = DeepANTProfile.Run,
        ["nbeats-profile"]      = NBEATSProfile.Run,
        ["autoformer-profile"]  = AutoformerProfile.Run,
        ["resnet50-profile"]    = ResNet50Profile.Run,
        ["clone-diag"]          = CloneDiag.Run,
        ["ngboost-profile"]     = NGBoostProfile.Run,
        ["svc-profile"]         = SVCProfile.Run,
        ["vec-inspect"]         = VecInspect.Run,
    };

    static async Task Main(string[] args)
    {
        if (args.Length > 0 && ProfileModes.TryGetValue(args[0], out var profile))
        {
            profile();
            return;
        }

        int choice = -1;

        // Check for command-line argument
        if (args.Length > 0 && int.TryParse(args[0], out int argChoice))
        {
            choice = argChoice;
            Console.WriteLine($"Running option {choice} from command line...");
        }
        else
        {
            Console.WriteLine("AiDotNet Examples");
            Console.WriteLine("================");
            Console.WriteLine("1. Neural Network Example (Classification and Regression)");
            Console.WriteLine("2. Multiple Regression Example (House Price Prediction)");
            Console.WriteLine("3. Time Series Example (Stock Price Forecasting)");
            Console.WriteLine("4. Phase A GPU Acceleration Integration Tests");
            Console.WriteLine("5. DeconvolutionalLayer Test");
            Console.WriteLine("6. Meta-Learning Examples (MAML, ProtoNets, CNP, Data Infrastructure)");
            // Note: GPU tuning tests have been removed as they require internal Tensors types
            // that are not exposed in the NuGet package. These are developer-only tools.
            Console.WriteLine("0. Exit");
            Console.WriteLine();
            Console.Write("Select an example to run (0-6): ");

            int.TryParse(Console.ReadLine(), out choice);
        }

        if (choice >= 0)
        {
            // Only clear console in interactive mode
            if (args.Length == 0)
            {
                try { Console.Clear(); } catch { /* Ignore clear failures */ }
            }

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
                    Console.WriteLine("Running Phase A GPU Acceleration Integration Tests...");
                    Console.WriteLine();
                    PrototypeIntegrationTests.RunAll();
                    break;
                case 5:
                    DeconvTest.Run();
                    break;
                case 6:
                    var metaExample = new MetaLearningExample();
                    metaExample.RunExample();
                    break;
                default:
                    Console.WriteLine("Invalid choice. Please select a number between 0 and 6.");
                    break;
            }
        }
        else
        {
            Console.WriteLine("Invalid input. Please enter a number.");
        }

        // Only wait for key press in interactive mode
        if (args.Length == 0)
        {
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}
