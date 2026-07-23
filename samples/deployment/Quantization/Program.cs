// AiDotNet — Model Quantization
//
// Post-training quantization through the AiModelBuilder facade:
// ConfigureQuantization(new QuantizationConfig { Mode = ... }) quantizes the
// model's weights during BuildAsync. The returned AiModelResult exposes a
// QuantizationInfo (original vs quantized size, bit width) and predicts as usual
// via result.Predict().

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Deployment.Configuration;   // QuantizationConfig
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("==========================================");
Console.WriteLine("  Model Quantization Demo                 ");
Console.WriteLine("==========================================\n");

const int inputSize = 64;
const int numClasses = 10;
const int numSamples = 256;

// ── Synthetic training data ────────────────────────────────────────────────
var rng = new Random(42);
var trainX = new Tensor<double>(new[] { numSamples, inputSize });
var trainY = new Tensor<double>(new[] { numSamples, numClasses });
for (int i = 0; i < numSamples; i++)
{
    for (int j = 0; j < inputSize; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % numClasses }] = 1.0;
}
var probe = new Tensor<double>(new[] { 1, inputSize });
for (int j = 0; j < inputSize; j++) probe[new[] { 0, j }] = rng.NextDouble();

Console.WriteLine($"Model: MLP, input {inputSize} -> {numClasses} classes");
Console.WriteLine("Quantizing through AiModelBuilder.ConfigureQuantization ...\n");
Console.WriteLine($"{"Mode",-10} {"Quantized",-10} {"Bits",-6} {"Orig bytes",-12} {"Quant bytes",-12} {"Ratio",-8}");
Console.WriteLine(new string('-', 64));

foreach (var mode in new[] { QuantizationMode.Int8, QuantizationMode.Float16 })
{
    // Each pass builds + trains a fresh model, then quantizes it via the facade.
    var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
        inputFeatures: inputSize, numClasses: numClasses, complexity: NetworkComplexity.Simple));
    try
    {
        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
            .ConfigureQuantization(new QuantizationConfig { Mode = mode })
            .BuildAsync();

        // Inference still flows through the result object.
        _ = result.Predict(probe);

        if (result.QuantizationInfo is { } qi)
        {
            double ratio = qi.QuantizedSizeBytes > 0
                ? (double)qi.OriginalSizeBytes / qi.QuantizedSizeBytes
                : 0;
            Console.WriteLine(
                $"{mode,-10} {qi.IsQuantized,-10} {qi.BitWidth,-6} {qi.OriginalSizeBytes,-12:N0} {qi.QuantizedSizeBytes,-12:N0} {ratio,-7:F2}x");
        }
        else
        {
            Console.WriteLine($"{mode,-10} (no quantization info returned)");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"{mode,-10} (reported: {ex.Message})");
    }
}

Console.WriteLine(@"
Quantization shrinks model weights for deployment with minimal accuracy loss:
  - INT8     ~4x smaller than FP32, good accuracy retention.
  - Float16  ~2x smaller, near-lossless.
For advanced schemes (GPTQ, AWQ) configure a QuantizationStrategy on the config.
");

Console.WriteLine("=== Sample Complete ===");
