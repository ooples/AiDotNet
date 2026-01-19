using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;

Console.WriteLine("=== AiDotNet QLoRA Fine-Tuning ===");
Console.WriteLine("4-bit Quantized LoRA for Memory-Efficient Fine-Tuning\n");

// Configuration
const int inputSize = 256;
const int hiddenSize = 512;
const int outputSize = 10;
const int loraRank = 8;
const double loraAlpha = 8.0;

Console.WriteLine("Configuration:");
Console.WriteLine($"  - Input Size: {inputSize}");
Console.WriteLine($"  - Hidden Size: {hiddenSize}");
Console.WriteLine($"  - Output Size: {outputSize}");
Console.WriteLine($"  - LoRA Rank: {loraRank}");
Console.WriteLine($"  - LoRA Alpha: {loraAlpha}");
Console.WriteLine();

// Create base model layers
Console.WriteLine("Creating base model layers...");

var baseLayer1 = new DenseLayer<double>(inputSize, hiddenSize, (IActivationFunction<double>)new ReLUActivation<double>());
var baseLayer2 = new DenseLayer<double>(hiddenSize, hiddenSize, (IActivationFunction<double>)new ReLUActivation<double>());
var baseLayer3 = new DenseLayer<double>(hiddenSize, outputSize, (IVectorActivationFunction<double>)new SoftmaxActivation<double>());

// Calculate memory usage for different configurations
Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("Memory Analysis: Standard vs QLoRA");
Console.WriteLine(new string('=', 70));

// Standard precision (16-bit/2 bytes per parameter)
int layer1Params = inputSize * hiddenSize + hiddenSize;
int layer2Params = hiddenSize * hiddenSize + hiddenSize;
int layer3Params = hiddenSize * outputSize + outputSize;
int totalBaseParams = layer1Params + layer2Params + layer3Params;

long memory16bit = totalBaseParams * 2L;
long memory32bit = totalBaseParams * 4L;

Console.WriteLine("\nBase Model Memory (without LoRA):");
Console.WriteLine($"  Total Parameters: {totalBaseParams:N0}");
Console.WriteLine($"  Memory (FP32): {FormatBytes(memory32bit)}");
Console.WriteLine($"  Memory (FP16): {FormatBytes(memory16bit)}");

// QLoRA memory calculation
int loraLayer1Params = loraRank * (inputSize + hiddenSize);
int loraLayer2Params = loraRank * (hiddenSize + hiddenSize);
int loraLayer3Params = loraRank * (hiddenSize + outputSize);
int totalLoraParams = loraLayer1Params + loraLayer2Params + loraLayer3Params;

long memoryBase4bit = (long)(totalBaseParams * 0.5);
long memoryQuantConstants = (long)(memoryBase4bit * 0.03);
long memoryLoraFP32 = totalLoraParams * 4L;
long memoryQlora = memoryBase4bit + memoryQuantConstants + memoryLoraFP32;

Console.WriteLine("\nQLoRA Memory Breakdown:");
Console.WriteLine($"  Base weights (4-bit): {FormatBytes(memoryBase4bit)}");
Console.WriteLine($"  Quantization constants: {FormatBytes(memoryQuantConstants)}");
Console.WriteLine($"  LoRA adapters (FP32): {FormatBytes(memoryLoraFP32)}");
Console.WriteLine($"  Total QLoRA memory: {FormatBytes(memoryQlora)}");

double savings = (1.0 - (double)memoryQlora / memory16bit) * 100;
Console.WriteLine($"\nMemory Savings vs FP16: {savings:F1}% ({(double)memory16bit / memoryQlora:F1}x reduction)");

// QLoRA configuration explanation
Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("QLoRA Quantization Configuration");
Console.WriteLine(new string('=', 70));

Console.WriteLine("\nQuantization Types:");
Console.WriteLine("  - INT4: Uniform 4-bit integer (-8 to 7)");
Console.WriteLine("  - NF4: 4-bit Normal Float (optimal for normally distributed weights)");
Console.WriteLine();

Console.WriteLine("NF4 Quantization Levels (16 values optimized for normal distribution):");
var nf4Values = new double[]
{
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
};

Console.WriteLine("  Index | NF4 Value | Description");
Console.WriteLine("  ------|-----------|-------------");
for (int i = 0; i < nf4Values.Length; i++)
{
    string desc = i switch
    {
        0 => "Minimum",
        7 => "Zero",
        15 => "Maximum",
        _ => ""
    };
    Console.WriteLine($"  {i,5} | {nf4Values[i],9:F4} | {desc}");
}

Console.WriteLine("\nNotice: Values are NOT evenly spaced - more resolution near zero");
Console.WriteLine("        where most neural network weights are concentrated.");

// Create QLoRA adapters
Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("Creating QLoRA Adapters");
Console.WriteLine(new string('=', 70));

try
{
    Console.WriteLine("\nInitializing QLoRA adapter for Layer 1...");
    Console.WriteLine($"  - Input size: {inputSize}");
    Console.WriteLine($"  - Output size: {hiddenSize}");
    Console.WriteLine($"  - LoRA rank: {loraRank}");
    Console.WriteLine($"  - Quantization: NF4");
    Console.WriteLine($"  - Double quantization: Enabled");
    Console.WriteLine($"  - Block size: 64");

    var qloraLayer1 = new QLoRAAdapter<double>(
        baseLayer: baseLayer1,
        rank: loraRank,
        alpha: loraAlpha,
        quantizationType: QLoRAAdapter<double>.QuantizationType.NF4,
        useDoubleQuantization: true,
        quantizationBlockSize: 64,
        freezeBaseLayer: true);

    Console.WriteLine("  [OK] Layer 1 QLoRA adapter created");

    Console.WriteLine("\nInitializing QLoRA adapter for Layer 2...");
    var qloraLayer2 = new QLoRAAdapter<double>(
        baseLayer: baseLayer2,
        rank: loraRank,
        alpha: loraAlpha,
        quantizationType: QLoRAAdapter<double>.QuantizationType.NF4,
        useDoubleQuantization: true,
        quantizationBlockSize: 64,
        freezeBaseLayer: true);

    Console.WriteLine("  [OK] Layer 2 QLoRA adapter created");

    Console.WriteLine("\nInitializing QLoRA adapter for Layer 3...");
    var qloraLayer3 = new QLoRAAdapter<double>(
        baseLayer: baseLayer3,
        rank: loraRank,
        alpha: loraAlpha,
        quantizationType: QLoRAAdapter<double>.QuantizationType.NF4,
        useDoubleQuantization: true,
        quantizationBlockSize: 64,
        freezeBaseLayer: true);

    Console.WriteLine("  [OK] Layer 3 QLoRA adapter created");

    // Display adapter properties
    Console.WriteLine("\nQLoRA Adapter Properties:");
    Console.WriteLine($"  Quantization Type: {qloraLayer1.Quantization}");
    Console.WriteLine($"  Double Quantization: {qloraLayer1.UsesDoubleQuantization}");
    Console.WriteLine($"  Block Size: {qloraLayer1.BlockSize}");

    // Merging demonstration
    Console.WriteLine("\n" + new string('=', 70));
    Console.WriteLine("QLoRA Adapter Merging");
    Console.WriteLine(new string('=', 70));

    Console.WriteLine("\nMerging QLoRA adapters back to base model...");
    Console.WriteLine("  1. Dequantize base weights (4-bit -> FP32)");
    Console.WriteLine("  2. Compute LoRA contribution (B * A * alpha/rank)");
    Console.WriteLine("  3. Add LoRA contribution to dequantized weights");
    Console.WriteLine("  4. (Optional) Re-quantize for deployment");

    try
    {
        var mergedLayer1 = qloraLayer1.MergeToOriginalLayer();
        Console.WriteLine("  [OK] Layer 1 merged successfully");

        var mergedLayer2 = qloraLayer2.MergeToOriginalLayer();
        Console.WriteLine("  [OK] Layer 2 merged successfully");

        var mergedLayer3 = qloraLayer3.MergeToOriginalLayer();
        Console.WriteLine("  [OK] Layer 3 merged successfully");

        Console.WriteLine("\nMerged model ready for deployment!");
        Console.WriteLine("  - Can be re-quantized to 4-bit for efficient inference");
        Console.WriteLine("  - Or kept in FP16/FP32 for maximum accuracy");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  Merge demonstration note: {ex.Message}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"\nNote: QLoRA requires Dense/FullyConnected layers with 1D input/output.");
    Console.WriteLine($"This sample demonstrates the API patterns and memory analysis.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

// Memory comparison visualization
Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("Memory Comparison Visualization");
Console.WriteLine(new string('=', 70));

var memoryConfigs = new[]
{
    ("Full FP32 Fine-Tuning", memory32bit, totalBaseParams),
    ("Full FP16 Fine-Tuning", memory16bit, totalBaseParams),
    ("Standard LoRA (FP16 base)", memory16bit + totalLoraParams * 4L, totalLoraParams),
    ("QLoRA (4-bit base)", memoryQlora, totalLoraParams)
};

long maxMemory = memoryConfigs.Max(c => c.Item2);
int barWidth = 40;

Console.WriteLine();
foreach (var (name, memory, trainable) in memoryConfigs)
{
    int barLength = (int)((double)memory / maxMemory * barWidth);
    string bar = new string('#', barLength) + new string('.', barWidth - barLength);
    Console.WriteLine($"  {name,-25} [{bar}] {FormatBytes(memory)}");
    Console.WriteLine($"  {"",-25} Trainable: {trainable:N0} params\n");
}

// QLoRA vs Standard LoRA comparison table
Console.WriteLine(new string('=', 70));
Console.WriteLine("QLoRA vs Standard LoRA Comparison");
Console.WriteLine(new string('=', 70));

Console.WriteLine("\n| Aspect                  | Standard LoRA    | QLoRA            |");
Console.WriteLine("|-------------------------|------------------|------------------|");
Console.WriteLine($"| Base weight precision   | FP16 (2 bytes)   | INT4 (0.5 bytes) |");
Console.WriteLine($"| LoRA adapter precision  | FP32             | FP32             |");
Console.WriteLine($"| Memory for base         | {FormatBytes(memory16bit),-16} | {FormatBytes(memoryBase4bit),-16} |");
Console.WriteLine($"| Memory for LoRA         | {FormatBytes(totalLoraParams * 4L),-16} | {FormatBytes(memoryLoraFP32),-16} |");
Console.WriteLine($"| Total memory            | {FormatBytes(memory16bit + totalLoraParams * 4L),-16} | {FormatBytes(memoryQlora),-16} |");
Console.WriteLine($"| Trainable parameters    | {totalLoraParams:N0,-16} | {totalLoraParams:N0,-16} |");
Console.WriteLine($"| Forward pass overhead   | None             | Dequantization   |");
Console.WriteLine($"| Best for                | Speed priority   | Memory limited   |");

// Real-world scaling example
Console.WriteLine("\n" + new string('=', 70));
Console.WriteLine("Real-World Scaling: 7B Parameter Model Example");
Console.WriteLine(new string('=', 70));

long params7B = 7_000_000_000L;
long memory7B_FP32 = params7B * 4L;
long memory7B_FP16 = params7B * 2L;
long memory7B_4bit = (long)(params7B * 0.5);

long loraParams7B = (long)(params7B * 0.01);
long loraMemory7B = loraParams7B * 4L;

Console.WriteLine("\n7B Parameter Model Memory Requirements:");
Console.WriteLine($"  Full Fine-Tuning (FP32):    {FormatBytes(memory7B_FP32)} (need ~56GB GPU)");
Console.WriteLine($"  Full Fine-Tuning (FP16):    {FormatBytes(memory7B_FP16)} (need ~28GB GPU)");
Console.WriteLine($"  Standard LoRA (FP16 base):  {FormatBytes(memory7B_FP16 + loraMemory7B)} (need ~28GB GPU)");
Console.WriteLine($"  QLoRA (4-bit base):         {FormatBytes(memory7B_4bit + loraMemory7B)} (need ~7GB GPU)");

Console.WriteLine("\nWith QLoRA, you can fine-tune a 7B model on a consumer GPU!");
Console.WriteLine("  - NVIDIA RTX 3080 (10GB): Can train 7B models");
Console.WriteLine("  - NVIDIA RTX 4090 (24GB): Can train 13B-30B models");
Console.WriteLine("  - Single A100 (80GB): Can train 65B+ models");

Console.WriteLine("\n=== Sample Complete ===");

static string FormatBytes(long bytes)
{
    string[] sizes = { "B", "KB", "MB", "GB", "TB" };
    int order = 0;
    double size = bytes;

    while (size >= 1024 && order < sizes.Length - 1)
    {
        order++;
        size /= 1024;
    }

    return $"{size:F2} {sizes[order]}";
}
