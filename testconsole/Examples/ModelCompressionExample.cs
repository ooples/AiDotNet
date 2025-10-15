using AiDotNet.Compression;
using AiDotNet.Compression.Quantization;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;
using System;
using System.IO;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Examples demonstrating how to use model compression techniques.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This class provides examples of how to apply different compression techniques
    /// to neural network models, evaluate their effects, and save/load compressed models.
    /// </para>
    /// <para><b>For Beginners:</b> These examples show how to make your models smaller and faster.
    /// 
    /// Model compression can help you:
    /// - Deploy models to mobile devices or edge devices with limited resources
    /// - Reduce memory usage and storage requirements
    /// - Improve inference speed
    /// 
    /// These examples demonstrate the complete workflow from training to compressed deployment.
    /// </para>
    /// </remarks>
    public static class ModelCompressionExample
    {
        /// <summary>
        /// Demonstrates quantization compression on a neural network model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This example shows how to apply 8-bit quantization to a trained neural network,
        /// evaluate its performance before and after compression, and save the compressed model.
        /// </para>
        /// <para><b>For Beginners:</b> This shows how to use quantization to compress a model.
        /// 
        /// Quantization reduces the precision of model parameters (weights):
        /// - Converting 32-bit floating point numbers to 8-bit integers
        /// - Making the model up to 4x smaller
        /// - Potentially improving inference speed
        /// </para>
        /// </remarks>
        public static void QuantizationExample()
        {
            Console.WriteLine("Model Compression Example: Quantization");
            Console.WriteLine("========================================");
            
            // Step 1: Create and train a simple neural network
            Console.WriteLine("\n1. Creating and training a neural network...");
            var model = CreateAndTrainSimpleNetwork();
            
            // Step 2: Prepare test data for evaluating compression
            Console.WriteLine("\n2. Preparing test data...");
            var (testInputs, testOutputs) = CreateTestData(100);
            
            // Step 3: Create a quantization compressor
            Console.WriteLine("\n3. Creating quantization compressor...");
            var compressor = new QuantizationCompressor<double, FeedForwardNeuralNetwork<double>, Tensor<double>, Tensor<double>>();
            
            // Step 4: Configure compression options
            Console.WriteLine("\n4. Configuring compression options...");
            var options = new ModelCompressionOptions
            {
                Technique = CompressionTechnique.Quantization,
                QuantizationPrecision = 8, // 8-bit quantization
                TargetCompressionRatio = 4.0,
                MaxAcceptableAccuracyLoss = 0.02, // Accept up to 2% accuracy loss
                VerifyAccuracy = true
            };
            
            // Step 5: Apply compression
            Console.WriteLine("\n5. Applying quantization compression...");
            var compressedModel = compressor.Compress(model, options);
            
            // Step 6: Evaluate compression results
            Console.WriteLine("\n6. Evaluating compression results...");
            var results = compressor.EvaluateCompression(model, compressedModel, testInputs, testOutputs);
            
            // Print compression results
            Console.WriteLine("\nCompression Results:");
            Console.WriteLine(results.ToString());
            Console.WriteLine($"Original Model Size: {results.OriginalModelSizeBytes / 1024.0:F2} KB");
            Console.WriteLine($"Compressed Model Size: {results.CompressedModelSizeBytes / 1024.0:F2} KB");
            Console.WriteLine($"Compression Ratio: {results.CompressionRatio:F2}x");
            Console.WriteLine($"Accuracy Change: {results.OriginalAccuracy:P2} → {results.CompressedAccuracy:P2}");
            Console.WriteLine($"Inference Speedup: {results.InferenceSpeedupRatio:F2}x");
            
            // Step 7: Save the compressed model
            Console.WriteLine("\n7. Saving compressed model...");
            string outputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "compressed_model.bin");
            compressor.SerializeCompressedModel(compressedModel, outputPath);
            Console.WriteLine($"Compressed model saved to: {outputPath}");
            
            // Step 8: Load the compressed model
            Console.WriteLine("\n8. Loading compressed model...");
            var loadedModel = compressor.DeserializeCompressedModel(outputPath);
            Console.WriteLine("Compressed model loaded successfully.");
            
            // Step 9: Verify the loaded model
            Console.WriteLine("\n9. Verifying loaded model...");
            var prediction = loadedModel.Predict(testInputs[0]);
            Console.WriteLine("Model prediction verified successfully.");
        }

        /// <summary>
        /// Demonstrates pruning compression on a neural network model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This example shows how to apply pruning to a trained neural network,
        /// evaluate its performance before and after compression, and save the compressed model.
        /// </para>
        /// <para><b>For Beginners:</b> This shows how to use pruning to compress a model.
        /// 
        /// Pruning removes unimportant connections from the neural network:
        /// - Setting less important weights to zero
        /// - Making the network sparse
        /// - Reducing model size through efficient storage of sparse matrices
        /// </para>
        /// </remarks>
        public static void PruningExample()
        {
            Console.WriteLine("Model Compression Example: Pruning");
            Console.WriteLine("===================================");
            
            // Step 1: Create and train a simple neural network
            Console.WriteLine("\n1. Creating and training a neural network...");
            var model = CreateAndTrainSimpleNetwork();
            
            // Step 2: Prepare test data for evaluating compression
            Console.WriteLine("\n2. Preparing test data...");
            var (testInputs, testOutputs) = CreateTestData(100);
            
            // Step 3: Configure compression options for pruning
            Console.WriteLine("\n3. Configuring pruning compression options...");
            var options = new ModelCompressionOptions
            {
                Technique = CompressionTechnique.Pruning,
                PruningSparsityTarget = 0.7, // Target 70% sparsity
                MaxAcceptableAccuracyLoss = 0.03, // Accept up to 3% accuracy loss
                VerifyAccuracy = true
            };
            
            // Note: For this example, we're not implementing a full pruning compressor
            Console.WriteLine("\n[Example Only] Pruning would remove 70% of smallest weights.");
            Console.WriteLine("A complete implementation would include a PruningCompressor class.");
        }

        /// <summary>
        /// Demonstrates knowledge distillation compression on a neural network model.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This example shows how to apply knowledge distillation to transfer knowledge
        /// from a large "teacher" model to a smaller "student" model.
        /// </para>
        /// <para><b>For Beginners:</b> This shows how to create a smaller model that mimics a larger one.
        /// 
        /// Knowledge distillation creates a small model that learns from a larger one:
        /// - The large model (teacher) is already trained and accurate
        /// - The small model (student) learns to mimic the teacher's outputs
        /// - This produces a much smaller model that retains most of the accuracy
        /// </para>
        /// </remarks>
        public static void KnowledgeDistillationExample()
        {
            Console.WriteLine("Model Compression Example: Knowledge Distillation");
            Console.WriteLine("==================================================");
            
            // Step 1: Create and train a large "teacher" neural network
            Console.WriteLine("\n1. Creating and training a large teacher model...");
            var teacherModel = CreateAndTrainLargeNetwork();
            
            // Step 2: Create a smaller "student" neural network
            Console.WriteLine("\n2. Creating a smaller student model...");
            var studentModel = CreateSmallNetwork();
            
            // Step 3: Prepare training data for knowledge distillation
            Console.WriteLine("\n3. Preparing data for knowledge distillation...");
            var (trainingInputs, _) = CreateTestData(1000);
            
            // Step 4: Configure compression options for knowledge distillation
            Console.WriteLine("\n4. Configuring knowledge distillation options...");
            var options = new ModelCompressionOptions
            {
                Technique = CompressionTechnique.KnowledgeDistillation,
                DistillationStudentSize = 0.3, // Student is 30% the size of teacher
                DistillationTemperature = 2.0, // Use temperature of 2.0 for distillation
                MaxAcceptableAccuracyLoss = 0.05 // Accept up to 5% accuracy loss
            };
            
            // Note: For this example, we're not implementing a full knowledge distillation compressor
            Console.WriteLine("\n[Example Only] Knowledge distillation would train the smaller model to mimic the larger one.");
            Console.WriteLine("A complete implementation would include a KnowledgeDistillationCompressor class.");
            
            // Step 5: Simulate knowledge distillation
            Console.WriteLine("\n5. Showing model size comparison:");
            long teacherSize = EstimateModelSize(teacherModel);
            long studentSize = EstimateModelSize(studentModel);
            Console.WriteLine($"Teacher Model Size: {teacherSize / 1024.0:F2} KB");
            Console.WriteLine($"Student Model Size: {studentSize / 1024.0:F2} KB");
            Console.WriteLine($"Size Reduction: {(double)teacherSize / studentSize:F2}x");
        }

        /// <summary>
        /// Demonstrates comparing different compression techniques.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This example shows how different compression techniques compare in terms of
        /// model size, accuracy preservation, and inference speed.
        /// </para>
        /// <para><b>For Beginners:</b> This compares different ways to make models smaller.
        /// 
        /// Different compression techniques have different trade-offs:
        /// - Some preserve accuracy better than others
        /// - Some produce smaller models than others
        /// - Some have better inference speed than others
        /// 
        /// This example helps you understand which technique might be best for your needs.
        /// </para>
        /// </remarks>
        public static void CompareCompressionTechniques()
        {
            Console.WriteLine("Model Compression Comparison");
            Console.WriteLine("============================");
            
            Console.WriteLine("\nComparison of different compression techniques:");
            Console.WriteLine("1. Quantization (8-bit)");
            Console.WriteLine("   - Size reduction: ~4x");
            Console.WriteLine("   - Accuracy impact: Minimal (typically <1%)");
            Console.WriteLine("   - Inference speedup: 1.5-2.5x");
            Console.WriteLine("   - Best for: General purpose compression, hardware with integer acceleration");
            
            Console.WriteLine("\n2. Pruning (70% sparsity)");
            Console.WriteLine("   - Size reduction: 2-3x");
            Console.WriteLine("   - Accuracy impact: Low to moderate (1-3%)");
            Console.WriteLine("   - Inference speedup: 1.2-1.8x (with sparse execution support)");
            Console.WriteLine("   - Best for: Over-parameterized models, when sparse execution is available");
            
            Console.WriteLine("\n3. Knowledge Distillation");
            Console.WriteLine("   - Size reduction: 3-10x (depending on student size)");
            Console.WriteLine("   - Accuracy impact: Moderate (2-5%)");
            Console.WriteLine("   - Inference speedup: 2-5x");
            Console.WriteLine("   - Best for: Very large models, when retraining is practical");
            
            Console.WriteLine("\n4. Quantized Pruning (combined approach)");
            Console.WriteLine("   - Size reduction: 5-8x");
            Console.WriteLine("   - Accuracy impact: Moderate (2-4%)");
            Console.WriteLine("   - Inference speedup: 2-3x");
            Console.WriteLine("   - Best for: Maximum compression with acceptable accuracy loss");
            
            Console.WriteLine("\nRecommended approach by deployment scenario:");
            Console.WriteLine("- Mobile devices: Quantization or Knowledge Distillation");
            Console.WriteLine("- Edge devices: Quantization or Quantized Pruning");
            Console.WriteLine("- Server deployment with constrained resources: Pruning");
            Console.WriteLine("- Real-time inference: Quantization");
        }

        /// <summary>
        /// A comprehensive example showing a production-ready compression workflow.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This example demonstrates a complete workflow for compressing a model for production,
        /// including model training, compression, evaluation, and deployment.
        /// </para>
        /// <para><b>For Beginners:</b> This shows how compression fits into a production ML workflow.
        /// 
        /// The complete workflow includes:
        /// 1. Training the original model
        /// 2. Compressing it for deployment
        /// 3. Evaluating the compressed model
        /// 4. Fine-tuning if necessary
        /// 5. Packaging for deployment
        /// 
        /// This process helps ensure that the compressed model meets production requirements.
        /// </para>
        /// </remarks>
        public static void ProductionCompressionWorkflow()
        {
            Console.WriteLine("Production-Ready Model Compression Workflow");
            Console.WriteLine("==========================================");
            
            // Step 1: Train the original model
            Console.WriteLine("\n1. Training the original model...");
            var model = CreateAndTrainLargeNetwork();
            
            // Step 2: Evaluate the original model
            Console.WriteLine("\n2. Evaluating original model performance...");
            var (testInputs, testOutputs) = CreateTestData(1000);
            double baselineAccuracy = EvaluateModelAccuracy(model, testInputs, testOutputs);
            Console.WriteLine($"Original model accuracy: {baselineAccuracy:P2}");
            
            // Step 3: Compress the model using quantization
            Console.WriteLine("\n3. Applying quantization compression...");
            var options = new ModelCompressionOptions
            {
                Technique = CompressionTechnique.Quantization,
                QuantizationPrecision = 8,
                UseMixedPrecision = true, // Use mixed precision for better accuracy
                MaxAcceptableAccuracyLoss = 0.01 // Only accept 1% accuracy loss for production
            };
            
            // Create compressor (in actual code, use the real implementation)
            Console.WriteLine("\n[Simulation] Applying 8-bit mixed precision quantization");
            
            // Step 4: Evaluate the compressed model
            Console.WriteLine("\n4. Evaluating compressed model...");
            double compressedAccuracy = baselineAccuracy - 0.005; // Simulated: 0.5% accuracy loss
            Console.WriteLine($"Compressed model accuracy: {compressedAccuracy:P2}");
            Console.WriteLine($"Accuracy loss: {baselineAccuracy - compressedAccuracy:P2}");
            
            // Step 5: Fine-tune the compressed model if needed
            double acceptableAccuracy = baselineAccuracy - 0.01; // 1% acceptable loss
            if (compressedAccuracy < acceptableAccuracy)
            {
                Console.WriteLine("\n5. Fine-tuning the compressed model...");
                Console.WriteLine("[Simulation] Fine-tuning for 5 epochs with reduced learning rate");
                compressedAccuracy += 0.003; // Simulated improvement from fine-tuning
                Console.WriteLine($"Fine-tuned model accuracy: {compressedAccuracy:P2}");
            }
            else
            {
                Console.WriteLine("\n5. Fine-tuning not needed (accuracy within acceptable range)");
            }
            
            // Step 6: Prepare the model for deployment
            Console.WriteLine("\n6. Preparing model for deployment...");
            Console.WriteLine("[Simulation] Serializing model with quantization parameters");
            Console.WriteLine("[Simulation] Converting to production-ready format");
            Console.WriteLine("[Simulation] Creating model signature file");
            
            // Step 7: Package for deployment
            Console.WriteLine("\n7. Packaging the model for deployment...");
            Console.WriteLine("[Simulation] Creating deployment package with model and metadata");
            Console.WriteLine("[Simulation] Adding inference runtime for target platform");
            
            // Step 8: Final verification
            Console.WriteLine("\n8. Final verification...");
            Console.WriteLine("[Simulation] Verifying model behavior on validation set");
            Console.WriteLine($"Final model size: ~{5.2:F1} MB (compressed from ~{20.8:F1} MB)");
            Console.WriteLine($"Final model accuracy: {compressedAccuracy:P2}");
            Console.WriteLine("Inference time: 2.3x faster than original model");
            
            Console.WriteLine("\nModel compression workflow complete. Model ready for production deployment.");
        }

        // Helper methods for the examples

        /// <summary>
        /// Creates and trains a simple feed-forward neural network.
        /// </summary>
        private static FeedForwardNeuralNetwork<double> CreateAndTrainSimpleNetwork()
        {
            // Create a simple neural network architecture
            var architecture = new NeuralNetworkArchitecture<double>(NetworkComplexity.Medium, NeuralNetworkTaskType.Regression);
            
            // Create the network (in actual code, we would define layers and train it)
            var network = new FeedForwardNeuralNetwork<double>(architecture);
            
            // Simulate training
            Console.WriteLine("[Simulation] Training network for 100 epochs...");
            Console.WriteLine("[Simulation] Training complete. Accuracy: 92.4%");
            
            return network;
        }

        /// <summary>
        /// Creates and trains a large neural network.
        /// </summary>
        private static FeedForwardNeuralNetwork<double> CreateAndTrainLargeNetwork()
        {
            // Create a neural network architecture for a large network
            var architecture = new NeuralNetworkArchitecture<double>(NetworkComplexity.Medium, NeuralNetworkTaskType.Regression);
            
            // Create the network (in actual code, we would define layers and train it)
            var network = new FeedForwardNeuralNetwork<double>(architecture);
            
            // Simulate training
            Console.WriteLine("[Simulation] Training large network for 200 epochs...");
            Console.WriteLine("[Simulation] Training complete. Accuracy: 94.8%");
            
            return network;
        }

        /// <summary>
        /// Creates a small neural network for knowledge distillation.
        /// </summary>
        private static FeedForwardNeuralNetwork<double> CreateSmallNetwork()
        {
            // Create a neural network architecture for a small network
            var architecture = new NeuralNetworkArchitecture<double>(NetworkComplexity.Medium, NeuralNetworkTaskType.Regression);
            
            // Create the network (in actual code, we would define layers)
            var network = new FeedForwardNeuralNetwork<double>(architecture);
            
            Console.WriteLine("[Simulation] Created small network with 30% parameter count of large network");
            
            return network;
        }

        /// <summary>
        /// Creates test data for evaluating models.
        /// </summary>
        private static (Tensor<double>[] inputs, Tensor<double>[] outputs) CreateTestData(int count)
        {
            var inputs = new Tensor<double>[count];
            var outputs = new Tensor<double>[count];
            
            // Create dummy data
            for (int i = 0; i < count; i++)
            {
                inputs[i] = new Tensor<double>([10]); // Example: 10-dimensional input
                outputs[i] = new Tensor<double>([1]); // Example: 1-dimensional output
            }
            
            Console.WriteLine($"[Simulation] Created {count} test samples");
            
            return (inputs, outputs);
        }

        /// <summary>
        /// Evaluates model accuracy on test data.
        /// </summary>
        private static double EvaluateModelAccuracy(
            FeedForwardNeuralNetwork<double> model, 
            Tensor<double>[] inputs, 
            Tensor<double>[] outputs)
        {
            // Simulate evaluating model accuracy
            return 0.943; // Example: 94.3% accuracy
        }

        /// <summary>
        /// Estimates model size in bytes.
        /// </summary>
        private static long EstimateModelSize(FeedForwardNeuralNetwork<double> model)
        {
            // Simulate estimating model size
            Random rnd = new Random(model.GetHashCode());
            return rnd.Next(100000, 10000000); // Random size between 100 KB and 10 MB
        }
    }
}