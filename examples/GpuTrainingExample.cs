using AiDotNet.Autodiff;
using AiDotNet.Gpu;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Examples;

/// <summary>
/// Demonstrates end-to-end GPU-accelerated neural network training.
/// </summary>
/// <remarks>
/// <para>
/// This example shows how to train a simple two-layer neural network using GPU acceleration.
/// It demonstrates:
/// - Setting up GPU execution context
/// - Creating trainable parameters
/// - Forward pass with GPU operations
/// - Loss computation and backpropagation
/// - Parameter updates with gradient descent
/// - Automatic CPU/GPU placement
/// </para>
/// <para><b>For Beginners:</b> This is a complete neural network training example!
///
/// The network structure:
/// - Input layer: 784 features (28x28 image)
/// - Hidden layer: 128 neurons with ReLU activation
/// - Output layer: 10 neurons (classification into 10 classes)
///
/// Training process:
/// 1. Forward pass: Input → Hidden → Output
/// 2. Compute loss: How wrong is the prediction?
/// 3. Backward pass: Compute gradients for all parameters
/// 4. Update parameters: Adjust weights to reduce loss
///
/// GPU acceleration makes this 10-100x faster for large datasets!
/// </para>
/// </remarks>
public class GpuTrainingExample
{
    public static void RunExample()
    {
        Console.WriteLine("=== GPU-Accelerated Neural Network Training ===\n");

        // Step 1: Initialize GPU backend
        using var backend = new IlgpuBackend<float>();
        backend.Initialize();

        if (!backend.IsAvailable)
        {
            Console.WriteLine("GPU not available. This example requires GPU support.");
            return;
        }

        Console.WriteLine($"GPU Device: {backend.DeviceName}");
        Console.WriteLine($"Total GPU Memory: {backend.TotalMemory / (1024 * 1024 * 1024):F2} GB");
        Console.WriteLine($"Free GPU Memory: {backend.FreeMemory / (1024 * 1024 * 1024):F2} GB\n");

        // Step 2: Create execution context with automatic placement
        using var context = new ExecutionContext(backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuThreshold = 50_000  // Use GPU for tensors with >50K elements
        };

        // Step 3: Initialize network parameters
        Console.WriteLine("Initializing network parameters...");

        const int inputSize = 784;   // 28x28 images flattened
        const int hiddenSize = 128;  // Hidden layer neurons
        const int outputSize = 10;   // 10 classes (digits 0-9)
        const float learningRate = 0.01f;

        // Weights and biases for layer 1 (input → hidden)
        var w1 = InitializeWeights(inputSize, hiddenSize);
        var b1 = InitializeBias(hiddenSize);

        // Weights and biases for layer 2 (hidden → output)
        var w2 = InitializeWeights(hiddenSize, outputSize);
        var b2 = InitializeBias(outputSize);

        Console.WriteLine($"W1 shape: [{string.Join("x", w1.Shape)}]");
        Console.WriteLine($"W2 shape: [{string.Join("x", w2.Shape)}]\n");

        // Step 4: Create synthetic training data
        Console.WriteLine("Creating synthetic training data...");
        const int batchSize = 32;
        var inputBatch = CreateRandomBatch(batchSize, inputSize);
        var targetBatch = CreateRandomTargets(batchSize, outputSize);

        Console.WriteLine($"Input batch shape: [{string.Join("x", inputBatch.Shape)}]");
        Console.WriteLine($"Target batch shape: [{string.Join("x", targetBatch.Shape)}]\n");

        // Step 5: Training loop
        Console.WriteLine("Starting training...\n");
        const int epochs = 10;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Reset statistics for this epoch
            context.ResetStatistics();

            using var tape = new GradientTape<float>();

            // Create computation nodes for parameters
            using var w1Node = GpuTensorOperations<float>.Variable(w1, context, "W1", requiresGradient: true);
            using var b1Node = GpuTensorOperations<float>.Variable(b1, context, "b1", requiresGradient: true);
            using var w2Node = GpuTensorOperations<float>.Variable(w2, context, "W2", requiresGradient: true);
            using var b2Node = GpuTensorOperations<float>.Variable(b2, context, "b2", requiresGradient: true);

            // Create computation node for input
            using var inputNode = GpuTensorOperations<float>.Constant(inputBatch, context, "input");

            // Watch parameters (we want to compute gradients for these)
            tape.Watch(w1Node);
            tape.Watch(b1Node);
            tape.Watch(w2Node);
            tape.Watch(b2Node);

            // ===== FORWARD PASS =====

            // Layer 1: hidden = ReLU(input · W1 + b1)
            using var layer1Matmul = GpuTensorOperations<float>.MatMul(inputNode, w1Node, context);
            using var layer1PreActivation = GpuTensorOperations<float>.Add(layer1Matmul, b1Node, context);
            using var hidden = GpuTensorOperations<float>.ReLU(layer1PreActivation, context);

            // Layer 2: output = hidden · W2 + b2
            using var layer2Matmul = GpuTensorOperations<float>.MatMul(hidden, w2Node, context);
            using var output = GpuTensorOperations<float>.Add(layer2Matmul, b2Node, context);

            // Compute loss (simplified MSE for demonstration)
            using var targetNode = GpuTensorOperations<float>.Constant(targetBatch, context, "target");
            using var error = GpuTensorOperations<float>.Subtract(output, targetNode, context);
            using var loss = GpuTensorOperations<float>.ElementwiseMultiply(error, error, context);

            // ===== BACKWARD PASS =====
            var gradients = tape.Gradient(loss, new[] { w1Node, b1Node, w2Node, b2Node });

            // ===== PARAMETER UPDATE =====
            // Update: param = param - learningRate * gradient
            if (gradients.ContainsKey(w1Node) && gradients[w1Node] != null)
            {
                w1 = UpdateParameter(w1, gradients[w1Node]!, learningRate);
            }
            if (gradients.ContainsKey(b1Node) && gradients[b1Node] != null)
            {
                b1 = UpdateParameter(b1, gradients[b1Node]!, learningRate);
            }
            if (gradients.ContainsKey(w2Node) && gradients[w2Node] != null)
            {
                w2 = UpdateParameter(w2, gradients[w2Node]!, learningRate);
            }
            if (gradients.ContainsKey(b2Node) && gradients[b2Node] != null)
            {
                b2 = UpdateParameter(b2, gradients[b2Node]!, learningRate);
            }

            // Calculate average loss
            float avgLoss = CalculateAverageLoss(loss.Value);

            // Print epoch statistics
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}:");
            Console.WriteLine($"  Loss: {avgLoss:F6}");
            Console.WriteLine($"  GPU Operations: {context.Statistics.GpuOperations}");
            Console.WriteLine($"  CPU Operations: {context.Statistics.CpuOperations}");
            Console.WriteLine($"  GPU Usage: {context.Statistics.GpuPercentage:F1}%");
            Console.WriteLine();
        }

        Console.WriteLine("Training completed!");
        Console.WriteLine("\n=== Summary ===");
        Console.WriteLine($"Final GPU Usage: {context.Statistics.GpuPercentage:F1}%");
        Console.WriteLine($"Total Operations: {context.Statistics.TotalOperations}");

        Console.WriteLine("\nGPU acceleration enabled automatic speedup for large tensor operations!");
        Console.WriteLine("Matrix multiplications and large activations were accelerated on GPU,");
        Console.WriteLine("while small operations remained on CPU to avoid transfer overhead.");
    }

    private static Tensor<float> InitializeWeights(int inputDim, int outputDim)
    {
        var weights = new Tensor<float>(new[] { inputDim, outputDim });
        var random = new Random(42);

        // Xavier initialization: scale = sqrt(2 / (inputDim + outputDim))
        float scale = (float)Math.Sqrt(2.0 / (inputDim + outputDim));

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        }

        return weights;
    }

    private static Tensor<float> InitializeBias(int size)
    {
        var bias = new Tensor<float>(new[] { 1, size });

        // Initialize biases to zero
        for (int i = 0; i < bias.Length; i++)
        {
            bias[i] = 0.0f;
        }

        return bias;
    }

    private static Tensor<float> CreateRandomBatch(int batchSize, int features)
    {
        var batch = new Tensor<float>(new[] { batchSize, features });
        var random = new Random(42);

        for (int i = 0; i < batch.Length; i++)
        {
            batch[i] = (float)(random.NextDouble() * 2 - 1);  // Range [-1, 1]
        }

        return batch;
    }

    private static Tensor<float> CreateRandomTargets(int batchSize, int numClasses)
    {
        var targets = new Tensor<float>(new[] { batchSize, numClasses });
        var random = new Random(42);

        // Create one-hot encoded targets
        for (int i = 0; i < batchSize; i++)
        {
            int targetClass = random.Next(numClasses);
            targets[new[] { i, targetClass }] = 1.0f;
        }

        return targets;
    }

    private static Tensor<float> UpdateParameter(Tensor<float> param, Tensor<float> gradient, float learningRate)
    {
        var updated = new Tensor<float>(param.Shape);

        for (int i = 0; i < param.Length; i++)
        {
            updated[i] = param[i] - learningRate * gradient[i];
        }

        return updated;
    }

    private static float CalculateAverageLoss(Tensor<float> lossTensor)
    {
        float sum = 0.0f;
        for (int i = 0; i < lossTensor.Length; i++)
        {
            sum += lossTensor[i];
        }
        return sum / lossTensor.Length;
    }

    /// <summary>
    /// Entry point for running the example standalone.
    /// </summary>
    public static void Main(string[] args)
    {
        try
        {
            RunExample();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
