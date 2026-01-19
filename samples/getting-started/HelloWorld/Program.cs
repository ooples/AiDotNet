using AiDotNet;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

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

// Create and configure the neural network
Console.WriteLine("Building neural network...");
Console.WriteLine("  Architecture: 2 inputs -> 8 hidden (ReLU) -> 1 output (Sigmoid)\n");

var architecture = new NeuralNetworkArchitecture<double>(
    inputFeatures: 2,
    numClasses: 1,
    complexity: NetworkComplexity.Simple
);

var neuralNetwork = new NeuralNetwork<double>(architecture);

// Training loop
Console.WriteLine("Training...");
const int epochs = 1000;
const int reportInterval = 200;

for (int epoch = 0; epoch < epochs; epoch++)
{
    neuralNetwork.Train(features, labels);

    if (epoch % reportInterval == 0)
    {
        double loss = neuralNetwork.GetLastLoss();
        Console.WriteLine($"  Epoch {epoch,4}: Loss = {loss:F4}");
    }
}

double finalLoss = neuralNetwork.GetLastLoss();
Console.WriteLine($"  Epoch {epochs,4}: Loss = {finalLoss:F4}");
Console.WriteLine("\nTraining complete!\n");

// Make predictions
Console.WriteLine("Predictions:");
Console.WriteLine("─────────────────────────────────────");

var predictions = neuralNetwork.Predict(features);

string[] inputLabels = { "[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]" };
double[] expected = { 0, 1, 1, 0 };

for (int i = 0; i < 4; i++)
{
    double prediction = predictions[new int[] { i, 0 }];
    double exp = expected[i];
    string status = Math.Abs(prediction - exp) < 0.5 ? "✓" : "✗";

    Console.WriteLine($"  {inputLabels[i]} => {prediction:F2} (expected: {exp}) {status}");
}

Console.WriteLine("\n=== Sample Complete ===");
