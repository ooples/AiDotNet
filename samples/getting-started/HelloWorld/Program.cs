using AiDotNet;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

Console.WriteLine("=== AiDotNet Hello World ===");
Console.WriteLine("Training a neural network to learn the XOR function\n");

// XOR training data
// XOR truth table: 0^0=0, 0^1=1, 1^0=1, 1^1=0
var inputs = new double[,]
{
    { 0, 0 },
    { 0, 1 },
    { 1, 0 },
    { 1, 1 }
};

var outputs = new double[,]
{
    { 0 },
    { 1 },
    { 1 },
    { 0 }
};

// Convert to tensors
var features = new Tensor<double>(new int[] { 4, 2 });
var labels = new Tensor<double>(new int[] { 4, 1 });

for (int i = 0; i < 4; i++)
{
    for (int j = 0; j < 2; j++)
        features[new int[] { i, j }] = inputs[i, j];
    labels[new int[] { i, 0 }] = outputs[i, 0];
}

// Build and train the model using the facade pattern
Console.WriteLine("Building neural network using AiModelBuilder...");
Console.WriteLine("  Architecture: 2 inputs -> 8 hidden (ReLU) -> 1 output (Sigmoid)\n");

try
{
    // Create the neural network architecture
    var architecture = new NeuralNetworkArchitecture<double>(
        inputFeatures: 2,
        numClasses: 1,
        complexity: NetworkComplexity.Simple
    );

    var neuralNetwork = new NeuralNetwork<double>(architecture);

    // Use the AiModelBuilder facade to build and train the model
    var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
        .ConfigureModel(neuralNetwork);

    Console.WriteLine("Training...");
    var result = await builder.BuildAsync(features, labels);

    // Make predictions using the result object (facade pattern)
    Console.WriteLine("\nPredictions:");
    Console.WriteLine("─────────────────────────────────────");

    var predictions = result.Predict(features);

    string[] inputLabels = { "[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]" };
    double[] expected = { 0, 1, 1, 0 };

    for (int i = 0; i < 4; i++)
    {
        double prediction = predictions[new int[] { i, 0 }];
        double exp = expected[i];
        string status = Math.Abs(prediction - exp) < 0.5 ? "✓" : "✗";

        Console.WriteLine($"  {inputLabels[i]} => {prediction:F2} (expected: {exp}) {status}");
    }

    // Display metrics through the result object
    if (result.OptimizationResult != null)
    {
        Console.WriteLine($"\nFinal Loss: {result.OptimizationResult.BestFitness:F4}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the facade pattern API for neural networks.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");
