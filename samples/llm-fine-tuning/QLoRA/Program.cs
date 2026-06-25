// AiDotNet — QLoRA Fine-Tuning
//
// QLoRA = a quantized, frozen base model + trainable LoRA adapters, all wired
// through the AiModelBuilder facade: ConfigureQuantization quantizes the base
// weights (shrinking the memory footprint) and ConfigureLoRA trains only the
// small low-rank matrices on top. The result is an AiModelResult you predict
// through.

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Deployment.Configuration;   // QuantizationConfig
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet QLoRA Fine-Tuning ===");
Console.WriteLine("Quantized base model + Low-Rank Adaptation\n");

// Configuration
const int inputSize = 32;
const int outputSize = 4;
const int loraRank = 4;
const double loraAlpha = 4.0;

Console.WriteLine("Configuration:");
Console.WriteLine($"  - Input Size: {inputSize}");
Console.WriteLine($"  - Output Size: {outputSize}");
Console.WriteLine($"  - LoRA Rank: {loraRank}");
Console.WriteLine($"  - Quantization: INT8 base weights\n");

// ── 1. Base model (the architecture builds a small MLP) ────────────────────
var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: inputSize, numClasses: outputSize, complexity: NetworkComplexity.Simple));

// ── 2. QLoRA = quantization config + LoRA config ───────────────────────────
var quantConfig = new QuantizationConfig { Mode = QuantizationMode.Int8 };
var loraConfig = new DefaultLoRAConfiguration<double>(
    rank: loraRank, alpha: loraAlpha, freezeBaseLayer: true);

// ── 3. Tiny synthetic memorization task ────────────────────────────────────
var rng = new Random(42);
const int n = 32;
var trainX = new Tensor<double>(new[] { n, inputSize });
var trainY = new Tensor<double>(new[] { n, outputSize });
for (int i = 0; i < n; i++)
{
    for (int j = 0; j < inputSize; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % outputSize }] = 1.0;
}

// ── 4. Fine-tune through the facade ────────────────────────────────────────
Console.WriteLine("Fine-tuning through AiModelBuilder (ConfigureQuantization + ConfigureLoRA) ...");
try
{
    var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
        .ConfigureModel(model)
        .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
        .ConfigureQuantization(quantConfig)
        .ConfigureLoRA(loraConfig)
        .BuildAsync();

    Console.WriteLine("  Fine-tuning complete.");
    if (result.TotalTrainableParameters is long trainable)
        Console.WriteLine($"  Trainable parameters (LoRA only): {trainable:N0}");
    if (result.QuantizationInfo is { } qi)
    {
        Console.WriteLine($"  Quantized: {qi.IsQuantized}, Mode: {qi.Mode}, BitWidth: {qi.BitWidth}");
        if (qi.OriginalSizeBytes > 0)
            Console.WriteLine($"  Base size: {qi.OriginalSizeBytes:N0} -> {qi.QuantizedSizeBytes:N0} bytes");
    }
    var prediction = result.Predict(trainX);
    Console.WriteLine($"  Prediction shape: [{string.Join(", ", prediction.Shape)}]");
}
catch (Exception ex)
{
    // Surface failures so the samples CI catches broken quantization/LoRA wiring.
    Console.Error.WriteLine($"  QLoRA sample failed: {ex.Message}");
    throw;
}

// ── Why QLoRA: memory footprint of the base weights by precision ───────────
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Memory Footprint by Base-Weight Precision");
Console.WriteLine(new string('=', 60));
Console.WriteLine("\n| Precision | Bits/Weight | Relative Size | Notes                  |");
Console.WriteLine("|-----------|-------------|---------------|------------------------|");
Console.WriteLine("| FP32      |          32 |         1.00x | Full precision base    |");
Console.WriteLine("| FP16      |          16 |         0.50x | Half precision         |");
Console.WriteLine("| INT8      |           8 |         0.25x | QLoRA (this sample)    |");
Console.WriteLine("| INT4      |           4 |         0.125x| Aggressive quantization|");

Console.WriteLine(@"
QLoRA keeps the large base model frozen in low precision (here INT8) and trains
only the small LoRA adapters in full precision, so fine-tuning fits in a
fraction of the memory a full fine-tune would need.
");

Console.WriteLine("=== Sample Complete ===");
