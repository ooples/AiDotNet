using AiDotNetTestConsole.Examples;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using TestConsoleApp.Examples;

namespace AiDotNetTestConsole;

class Program
{
    static void Main(string[] args)
    {
        // Check if command line arguments were provided
        if (args.Length > 0)
        {
            // Command line mode
            if (args[0].ToLower() == "all")
            {
                RunAllExamples();
                return;
            }
            else if (int.TryParse(args[0], out int exampleNumber))
            {
                RunSpecificExample(exampleNumber);
                return;
            }
            else
            {
                Console.WriteLine($"Invalid argument: {args[0]}");
                Console.WriteLine("Usage: dotnet run [example-number | all]");
                Console.WriteLine("Example: dotnet run 1");
                Console.WriteLine("         dotnet run all");
                return;
            }
        }

        // Interactive mode (original behavior)
        while (true)
        {
            Console.Clear();
            Console.WriteLine("AiDotNet Examples");
            Console.WriteLine("================");
            Console.WriteLine("1. Neural Network Example (Classification and Regression)");
            Console.WriteLine("2. Multiple Regression Example (House Price Prediction)");
            Console.WriteLine("3. Time Series Example (Stock Price Forecasting)");
            Console.WriteLine("4. Enhanced Regression Example (Real Estate Analysis)");
            Console.WriteLine("5. Enhanced Neural Network Example (Customer Churn Prediction)");
            Console.WriteLine("6. Enhanced Time Series Example (Energy Demand Forecasting)");
            Console.WriteLine("7. Transformer Positional Encoding Techniques Demo");
            Console.WriteLine("8. Reasoning Model Examples (Chain-of-Thought, Self-Consistency)");
            Console.WriteLine("9. Ensemble Model Example");
            Console.WriteLine("10. Online Learning Example");
            Console.WriteLine("11. Transfer Learning Example");
            Console.WriteLine("0. Exit");
            Console.WriteLine();
            Console.Write("Select an example to run (0-11): ");

            if (int.TryParse(Console.ReadLine(), out int choice))
            {
                Console.Clear();

                if (choice == 0)
                {
                    Console.WriteLine("Exiting...");
                    return;
                }

                RunSpecificExample(choice);

                if (choice != 7 && choice != 8)
                {
                    Console.WriteLine("\nPress any key to return to the main menu...");
                    Console.ReadKey();
                }
            }
            else
            {
                Console.WriteLine("Invalid input. Please enter a number.");
                Console.WriteLine("\nPress any key to continue...");
                Console.ReadKey();
            }
        }
    }

    static void RunSpecificExample(int exampleNumber)
    {
        try
        {
            switch (exampleNumber)
            {
                case 1:
                    var nnExample = new NeuralNetworkExample();
                    nnExample.RunExample();
                    break;
                case 2:
                    var regExample = new RegressionExample();
                    regExample.RunExample();
                    break;
                case 3:
                    var tsExample = new TimeSeriesExample();
                    tsExample.RunExample();
                    break;
                case 4:
                    var enhancedRegExample = new EnhancedRegressionExample();
                    enhancedRegExample.RunExample();
                    break;
                case 5:
                    var enhancedNNExample = new EnhancedNeuralNetworkExample();
                    enhancedNNExample.RunExample();
                    break;
                case 6:
                    var enhancedTSExample = new EnhancedTimeSeriesExample();
                    enhancedTSExample.RunExample();
                    break;
                case 7:
                    RunPositionalEncodingDemo();
                    break;
                case 8:
                    AiDotNet.Examples.ReasoningModelExample.RunAllExamples();
                    break;
                case 9:
                    EnsembleExample.Run();
                    break;
                case 10:
                    OnlineLearningExample.Run();
                    break;
                case 11:
                    AiDotNet.Examples.TransferLearningExample.Main(new string[] { });
                    break;
                default:
                    Console.WriteLine($"Invalid choice: {exampleNumber}. Please select a number between 0 and 11.");
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError running example {exampleNumber}: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }

    static void RunAllExamples()
    {
        Console.WriteLine("Running all AiDotNet examples...");
        Console.WriteLine(new string('=', 50));
        Console.WriteLine();

        for (int i = 1; i <= 11; i++)
        {
            Console.WriteLine($"\n--- Example {i} ---");
            try
            {
                RunSpecificExample(i);
                Console.WriteLine($"\nExample {i} completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nExample {i} failed: {ex.Message}");
            }
            Console.WriteLine();
        }
        
        Console.WriteLine(new string('=', 50));
        Console.WriteLine("All examples completed!");
    }

    private static void RunPositionalEncodingDemo()
    {
        Console.WriteLine("=================================================");
        Console.WriteLine("  Transformer Positional Encoding Techniques Demo");
        Console.WriteLine("=================================================");
        Console.WriteLine();
        
        while (true)
        {
            Console.WriteLine("Choose a demo to run:");
            Console.WriteLine("1. Create language understanding model (BERT-like)");
            Console.WriteLine("2. Create language generation model (GPT-like)");
            Console.WriteLine("3. Create translation model (Transformer-like)");
            Console.WriteLine("4. Create large language model with ALiBi");
            Console.WriteLine("5. Compare positional encoding techniques");
            Console.WriteLine("6. View positional encoding recommendations");
            Console.WriteLine("7. Return to main menu");
            Console.Write("\nEnter your choice (1-7): ");
            
            string? choice = Console.ReadLine();
            Console.WriteLine();
            
            switch (choice)
            {
                case "1":
                    DemonstrateBert();
                    break;
                case "2":
                    DemonstrateGpt();
                    break;
                case "3":
                    DemonstrateTranslation();
                    break;
                case "4":
                    DemonstrateLlm();
                    break;
                case "5":
                    CompareEncodings();
                    break;
                case "6":
                    TransformerExamples.PositionalEncodingRecommendations();
                    break;
                case "7":
                    return;
                default:
                    Console.WriteLine("Invalid choice. Please try again.");
                    break;
            }
            
            if (choice != "7")
            {
                Console.WriteLine("\nPress Enter to continue...");
                Console.ReadLine();
                Console.Clear();
            }
        }
    }
    
    private static void DemonstrateBert()
    {
        Console.WriteLine("Creating a BERT-like model with Learned positional encoding");
        Console.WriteLine("------------------------------------------------------------");
        
        var bertModel = TransformerExamples.CreateLanguageUnderstandingTransformer(
            vocabularySize: 30000,
            maxSequenceLength: 512,
            encodingType: PositionalEncodingType.Learned);
        
        PrintModelInfo(bertModel);
    }
    
    private static void DemonstrateGpt()
    {
        Console.WriteLine("Creating a GPT-like model with Rotary positional encoding");
        Console.WriteLine("----------------------------------------------------------");
        
        var gptModel = TransformerExamples.CreateLanguageGenerationTransformer(
            vocabularySize: 50000,
            maxSequenceLength: 1024,
            encodingType: PositionalEncodingType.Rotary);
        
        PrintModelInfo(gptModel);
    }
    
    private static void DemonstrateTranslation()
    {
        Console.WriteLine("Creating a Translation model with T5-style relative bias");
        Console.WriteLine("-------------------------------------------------------");
        
        var translationModel = TransformerExamples.CreateSequenceToSequenceTransformer(
            vocabularySize: 40000,
            maxSequenceLength: 256,
            encodingType: PositionalEncodingType.T5RelativeBias);
        
        PrintModelInfo(translationModel);
    }
    
    private static void DemonstrateLlm()
    {
        Console.WriteLine("Creating a Large Language Model with ALiBi positional encoding");
        Console.WriteLine("-------------------------------------------------------------");
        
        var llmModel = TransformerExamples.CreateLargeLanguageModel(
            vocabularySize: 50000,
            maxSequenceLength: 2048,
            encodingType: PositionalEncodingType.ALiBi);
        
        PrintModelInfo(llmModel);
        
        Console.WriteLine("\nDemonstrating custom ALiBi configuration:");
        var customParameters = new Dictionary<string, object> { { "slope", -0.2 } };
        TransformerExamples.DemonstratePositionalEncodingConfigurations();
    }
    
    private static void CompareEncodings()
    {
        Console.WriteLine("Comparing Different Positional Encoding Techniques");
        Console.WriteLine("------------------------------------------------");
        
        var baseConfig = new Dictionary<string, string>
        {
            ["Model Dimension"] = "512",
            ["Heads"] = "8",
            ["Layers"] = "6 (encoder) + 6 (decoder)",
            ["Sequence Length"] = "1024",
            ["Parameters"] = "~125M"
        };
        
        var comparisons = new Dictionary<PositionalEncodingType, Dictionary<string, string>>
        {
            [PositionalEncodingType.Sinusoidal] = new Dictionary<string, string>
            {
                ["Key advantage"] = "No learned parameters, can extrapolate to unseen positions",
                ["Best for"] = "Small to medium-sized models with fixed context windows",
                ["Popular in"] = "Original Transformer, smaller LMs",
                ["Extrapolation"] = "Moderate"
            },
            
            [PositionalEncodingType.Learned] = new Dictionary<string, string>
            {
                ["Key advantage"] = "Flexible representation of position, learned from data",
                ["Best for"] = "Tasks where sequence length is fixed",
                ["Popular in"] = "BERT, RoBERTa, earlier transformer models",
                ["Extrapolation"] = "Poor (limited to trained sequence length)"
            },
            
            [PositionalEncodingType.Relative] = new Dictionary<string, string>
            {
                ["Key advantage"] = "Models relationships between positions rather than absolute positions",
                ["Best for"] = "Tasks with complex positional relationships",
                ["Popular in"] = "Transformer-XL, Music Transformer",
                ["Extrapolation"] = "Good"
            },
            
            [PositionalEncodingType.Rotary] = new Dictionary<string, string>
            {
                ["Key advantage"] = "Combines absolute and relative position information",
                ["Best for"] = "Large language models, general purpose usage",
                ["Popular in"] = "GPT-Neo-X, PaLM, Llama",
                ["Extrapolation"] = "Very good"
            },
            
            [PositionalEncodingType.ALiBi] = new Dictionary<string, string>
            {
                ["Key advantage"] = "Linear bias in attention, excellent for extrapolation",
                ["Best for"] = "Models that need to handle sequences much longer than training",
                ["Popular in"] = "Bloom, some recent LLMs",
                ["Extrapolation"] = "Excellent"
            },
            
            [PositionalEncodingType.T5RelativeBias] = new Dictionary<string, string>
            {
                ["Key advantage"] = "Bucketed relative positions, efficient for text-to-text tasks",
                ["Best for"] = "Translation, summarization, other seq2seq tasks",
                ["Popular in"] = "T5, mT5",
                ["Extrapolation"] = "Good"
            }
        };
        
        // Print the base model configuration
        Console.WriteLine("Base Model Configuration:");
        foreach (var item in baseConfig)
        {
            Console.WriteLine($"  {item.Key}: {item.Value}");
        }
        
        // Print comparison for each encoding type
        foreach (PositionalEncodingType encodingType in Enum.GetValues(typeof(PositionalEncodingType)))
        {
            if (encodingType == PositionalEncodingType.None)
                continue;
                
            Console.WriteLine($"\n{encodingType} Positional Encoding:");
            
            if (comparisons.TryGetValue(encodingType, out var details))
            {
                foreach (var item in details)
                {
                    Console.WriteLine($"  {item.Key}: {item.Value}");
                }
            }
        }
    }
    
    private static void PrintModelInfo<T>(Transformer<T> model)
    {
        var metadata = model.GetModelMetaData();
        
        Console.WriteLine($"\nModel Information:");
        Console.WriteLine($"  Type: {metadata.ModelType}");
        
        // Print selected key information
        foreach (var key in new[] { 
            "NumHeads", "NumEncoderLayers", "NumDecoderLayers", 
            "PositionalEncodingType", "VocabularySize", "MaxSequenceLength", 
            "LayerCount", "ParameterCount" 
        })
        {
            if (metadata.AdditionalInfo.TryGetValue(key, out var value))
            {
                Console.WriteLine($"  {key}: {value}");
            }
        }
    }
}