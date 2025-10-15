using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Factories;
using AiDotNet.FoundationModels.Tokenizers;
using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;

namespace AiDotNetTestConsole.Examples
{
    /// <summary>
    /// Example demonstrating how to use the Transformer neural network with tokenizers for NLP tasks
    /// </summary>
    public class TransformerWithTokenizerExample
    {
        public static async Task RunAllExamples()
        {
            Console.WriteLine("=== Transformer with Tokenizer Examples ===\n");
            
            await Example1_BasicTextClassification();
            await Example2_CharacterLevelLanguageModel();
            await Example3_TextGeneration();
            await Example4_TokenizerComparison();
            await Example5_CustomVocabularyTraining();
        }

        /// <summary>
        /// Example 1: Basic text classification using Transformer with WordPiece tokenizer
        /// </summary>
        private static async Task Example1_BasicTextClassification()
        {
            Console.WriteLine("Example 1: Text Classification with Transformer\n");

            // Create a WordPiece tokenizer
            var tokenizer = TokenizerFactory.CreateTokenizer(TokenizerType.WordPiece);
            await tokenizer.InitializeAsync();

            // Create transformer configuration
            var config = new TransformerArchitecture<double>
            {
                VocabSize = tokenizer.VocabularySize,
                HiddenSize = 256,
                NumLayers = 4,
                NumHeads = 8,
                IntermediateSize = 1024,
                MaxPositionEmbeddings = 512,
                DropoutRate = 0.1
            };

            // Create transformer neural network
            var transformer = new Transformer<double>(config);

            // Create tokenizer adapter
            var textModel = new TokenizerAdapter<double>(
                transformer, 
                tokenizer,
                embeddingDim: config.HiddenSize,
                usePositionalEncoding: true
            );

            // Training data for sentiment classification
            var trainingData = new List<(string input, string label)>
            {
                ("This movie is absolutely fantastic!", "positive"),
                ("I love this product, it works great.", "positive"),
                ("Terrible experience, would not recommend.", "negative"),
                ("Worst purchase I've ever made.", "negative"),
                ("The service was okay, nothing special.", "neutral"),
                ("Average quality, average price.", "neutral")
            };

            // Train the model
            Console.WriteLine("Training sentiment classifier...");
            foreach (var (input, label) in trainingData)
            {
                var result = await textModel.Train(input, label);
                Console.WriteLine($"Training on: '{input}' -> {label}");
            }

            // Test predictions
            Console.WriteLine("\nTesting predictions:");
            var testInputs = new[]
            {
                "This is amazing!",
                "I hate this so much.",
                "It's fine, I guess."
            };

            foreach (var input in testInputs)
            {
                var prediction = await textModel.Predict(input);
                Console.WriteLine($"Input: '{input}' -> Predicted: '{prediction}'");
            }
            
            Console.WriteLine();
        }

        /// <summary>
        /// Example 2: Character-level language model for specialized domains
        /// </summary>
        private static async Task Example2_CharacterLevelLanguageModel()
        {
            Console.WriteLine("Example 2: Character-Level Language Model\n");

            // Create character-level tokenizer for DNA sequences
            var dnaTokenizer = CharacterLevelTokenizer.CreateDNATokenizer();
            await dnaTokenizer.InitializeAsync();

            Console.WriteLine($"DNA Tokenizer vocabulary size: {dnaTokenizer.VocabularySize}");

            // Create a smaller transformer for character-level modeling
            var config = new TransformerArchitecture<double>
            {
                VocabSize = dnaTokenizer.VocabularySize,
                HiddenSize = 128,
                NumLayers = 2,
                NumHeads = 4,
                IntermediateSize = 512,
                MaxPositionEmbeddings = 256
            };

            var transformer = new Transformer<double>(config);
            var dnaModel = new TokenizerAdapter<double>(transformer, dnaTokenizer, config.HiddenSize);

            // Train on DNA sequences
            var dnaSequences = new[]
            {
                ("ATCGATCG", "TAGCTAGC"), // Complement pairs
                ("GGCCAATT", "CCGGTTAA"),
                ("ACGTACGT", "TGCATGCA")
            };

            Console.WriteLine("Training DNA complement predictor...");
            foreach (var (input, complement) in dnaSequences)
            {
                await dnaModel.Train(input, complement);
                Console.WriteLine($"Training: {input} -> {complement}");
            }

            // Test DNA complement prediction
            Console.WriteLine("\nTesting DNA complement prediction:");
            var testDNA = "ATCG";
            var predictedComplement = await dnaModel.Predict(testDNA);
            Console.WriteLine($"Input DNA: {testDNA} -> Predicted complement: {predictedComplement}");
            
            Console.WriteLine();
        }

        /// <summary>
        /// Example 3: Text generation with different tokenizers
        /// </summary>
        private static async Task Example3_TextGeneration()
        {
            Console.WriteLine("Example 3: Text Generation with Byte-Level BPE\n");

            // Create Byte-level BPE tokenizer (GPT-2 style)
            var tokenizer = new ByteLevelBPETokenizer(vocabSize: 1000);
            await tokenizer.InitializeAsync();

            // Create transformer for text generation
            var config = new TransformerArchitecture<double>
            {
                VocabSize = tokenizer.VocabularySize,
                HiddenSize = 384,
                NumLayers = 6,
                NumHeads = 6,
                IntermediateSize = 1536,
                MaxPositionEmbeddings = 1024
            };

            var transformer = new Transformer<double>(config);
            var textGenerator = new TokenizerAdapter<double>(transformer, tokenizer, config.HiddenSize);

            // Train on simple patterns
            var trainingTexts = new[]
            {
                ("Once upon a time", "there was a brave knight"),
                ("In a galaxy far far away", "the force was strong"),
                ("The quick brown fox", "jumps over the lazy dog"),
                ("To be or not to be", "that is the question")
            };

            Console.WriteLine("Training text generator...");
            foreach (var (prompt, completion) in trainingTexts)
            {
                await textGenerator.Train(prompt, completion);
            }

            // Generate text
            Console.WriteLine("\nGenerating text:");
            var prompts = new[] { "Once upon a", "The quick", "To be or" };
            
            foreach (var prompt in prompts)
            {
                var generated = await textGenerator.GenerateText(prompt, maxLength: 20, temperature: 0.8);
                Console.WriteLine($"Prompt: '{prompt}' -> Generated: '{generated}'");
            }
            
            Console.WriteLine();
        }

        /// <summary>
        /// Example 4: Comparing different tokenizers
        /// </summary>
        private static async Task Example4_TokenizerComparison()
        {
            Console.WriteLine("Example 4: Tokenizer Comparison\n");

            var text = "Hello world! This is a test of different tokenizers.";
            
            // Compare different tokenizers
            var tokenizers = new Dictionary<string, ITokenizer>
            {
                ["WordPiece"] = TokenizerFactory.CreateTokenizer(TokenizerType.WordPiece),
                ["BPE"] = TokenizerFactory.CreateTokenizer(TokenizerType.BPE),
                ["Character"] = new CharacterLevelTokenizer(),
                ["Unigram"] = new UnigramTokenizer(vocabSize: 1000),
                ["ByteLevelBPE"] = new ByteLevelBPETokenizer(vocabSize: 1000)
            };

            Console.WriteLine($"Original text: \"{text}\"\n");

            foreach (var (name, tokenizer) in tokenizers)
            {
                await tokenizer.InitializeAsync();
                
                var tokens = await tokenizer.TokenizeAsync(text);
                var encoded = await tokenizer.EncodeAsync(text);
                var decoded = await tokenizer.DecodeAsync(encoded);
                
                Console.WriteLine($"{name} Tokenizer:");
                Console.WriteLine($"  Tokens: [{string.Join(", ", tokens)}]");
                Console.WriteLine($"  Token count: {tokens.Count}");
                Console.WriteLine($"  Token IDs: [{string.Join(", ", encoded.Take(10))}...]");
                Console.WriteLine($"  Decoded: \"{decoded}\"");
                Console.WriteLine($"  Vocabulary size: {tokenizer.VocabularySize}");
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Example 5: Training custom vocabulary
        /// </summary>
        private static async Task Example5_CustomVocabularyTraining()
        {
            Console.WriteLine("Example 5: Custom Vocabulary Training\n");

            // Create corpus for training
            var corpus = new List<string>
            {
                "The transformer architecture is revolutionary for NLP.",
                "Attention is all you need for sequence modeling.",
                "BERT uses bidirectional transformers for pre-training.",
                "GPT models are autoregressive language models.",
                "Tokenization is crucial for transformer models.",
                "Self-attention allows modeling long-range dependencies."
            };

            // Train a Unigram tokenizer
            var unigramTokenizer = new UnigramTokenizer(vocabSize: 100);
            
            Console.WriteLine("Training Unigram tokenizer on NLP corpus...");
            await unigramTokenizer.TrainAsync(corpus, maxIterations: 5);
            await unigramTokenizer.InitializeAsync();

            // Test the trained tokenizer
            Console.WriteLine("\nTesting trained tokenizer:");
            foreach (var text in corpus.Take(3))
            {
                var tokens = await unigramTokenizer.TokenizeAsync(text);
                Console.WriteLine($"\nText: \"{text}\"");
                Console.WriteLine($"Tokens: [{string.Join(", ", tokens)}]");
            }

            // Create and train a model with custom tokenizer
            var config = new TransformerArchitecture<double>
            {
                VocabSize = unigramTokenizer.VocabularySize,
                HiddenSize = 128,
                NumLayers = 2,
                NumHeads = 4,
                IntermediateSize = 512
            };

            var transformer = new Transformer<double>(config);
            var nlpModel = new TokenizerAdapter<double>(transformer, unigramTokenizer, config.HiddenSize);

            Console.WriteLine("\n\nTraining model with custom vocabulary...");
            // Train on domain-specific task
            var trainingPairs = new[]
            {
                ("transformer", "architecture"),
                ("attention", "mechanism"),
                ("tokenization", "process")
            };

            foreach (var (input, output) in trainingPairs)
            {
                await nlpModel.Train(input, output);
            }

            Console.WriteLine("\nModel trained with custom vocabulary!");
        }
    }
}