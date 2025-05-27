namespace AiDotNet.Examples;

using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;

/// <summary>
/// A simple CLI program demonstrating different positional encoding techniques for transformers.
/// </summary>
public static class PositionalEncodingDemoProgram
{
    /// <summary>
    /// Main entry point for the positional encoding demo program.
    /// </summary>
    public static void Main(string[] args)
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
            Console.WriteLine("7. Exit");
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
            
            Console.WriteLine("\nPress Enter to continue...");
            Console.ReadLine();
            Console.Clear();
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