using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;

Console.WriteLine("=== AiDotNet LoRA Fine-Tuning ===");
Console.WriteLine("Parameter-Efficient Fine-Tuning with Low-Rank Adaptation\n");

// Configuration
const int inputSize = 128;
const int hiddenSize = 256;
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

// Create base model (simulating a pre-trained model)
Console.WriteLine("Creating base model (simulating pre-trained weights)...");

var baseLayer1 = new DenseLayer<double>(inputSize, hiddenSize, (IActivationFunction<double>)new ReLUActivation<double>());
var baseLayer2 = new DenseLayer<double>(hiddenSize, hiddenSize, (IActivationFunction<double>)new ReLUActivation<double>());
var baseLayer3 = new DenseLayer<double>(hiddenSize, outputSize, (IVectorActivationFunction<double>)new SoftmaxActivation<double>());

// Calculate base model parameters
int baseParams = (inputSize * hiddenSize + hiddenSize) +
                 (hiddenSize * hiddenSize + hiddenSize) +
                 (hiddenSize * outputSize + outputSize);

Console.WriteLine($"  Base model parameters: {baseParams:N0}");

// Create LoRA configuration
Console.WriteLine("\nApplying LoRA adapters...");

var loraConfig = new DefaultLoRAConfiguration<double>(
    rank: loraRank,
    alpha: loraAlpha,
    freezeBaseLayer: true);

Console.WriteLine($"  - Rank: {loraConfig.Rank}");
Console.WriteLine($"  - Alpha: {loraConfig.Alpha}");
Console.WriteLine($"  - Freeze Base Layers: {loraConfig.FreezeBaseLayer}");

// Wrap layers with LoRA adapters
var loraLayer1 = loraConfig.ApplyLoRA(baseLayer1);
var loraLayer2 = loraConfig.ApplyLoRA(baseLayer2);
var loraLayer3 = loraConfig.ApplyLoRA(baseLayer3);

// Calculate LoRA parameters
int loraParamsPerLayer1 = loraRank * (inputSize + hiddenSize);
int loraParamsPerLayer2 = loraRank * (hiddenSize + hiddenSize);
int loraParamsPerLayer3 = loraRank * (hiddenSize + outputSize);
int totalLoraParams = loraParamsPerLayer1 + loraParamsPerLayer2 + loraParamsPerLayer3;

Console.WriteLine("\nLoRA Adapter Configuration:");
Console.WriteLine($"  Layer 1: {loraParamsPerLayer1:N0} trainable parameters");
Console.WriteLine($"  Layer 2: {loraParamsPerLayer2:N0} trainable parameters");
Console.WriteLine($"  Layer 3: {loraParamsPerLayer3:N0} trainable parameters");
Console.WriteLine($"  Total LoRA parameters: {totalLoraParams:N0}");
Console.WriteLine($"  Parameter reduction: {(1.0 - (double)totalLoraParams / baseParams) * 100:F1}%");
Console.WriteLine($"  Compression ratio: {(double)baseParams / totalLoraParams:F1}x");

// Demonstrate different LoRA variants
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Available LoRA Variants in AiDotNet");
Console.WriteLine(new string('=', 60));

Console.WriteLine(@"
AiDotNet includes 32 LoRA adapter variants:

| Variant             | Best For                          | Key Benefit              |
|---------------------|-----------------------------------|--------------------------|
| StandardLoRAAdapter | General purpose                   | Simple, effective        |
| QLoRAAdapter        | Memory-constrained training       | 4-bit quantization (75%) |
| DoRAAdapter         | Improved accuracy                 | Weight decomposition     |
| AdaLoRAAdapter      | Dynamic allocation                | Adaptive rank            |
| VeRAAdapter         | Extreme efficiency                | 10x fewer params         |
| LoRAPlusAdapter     | Faster convergence                | Dual learning rates      |
| LoHaAdapter         | CNN fine-tuning                   | Hadamard products        |
| LoKrAdapter         | High compression                  | Kronecker (57x)          |
| DyLoRAAdapter       | Dynamic rank training             | Flexible rank            |
| RoSAAdapter         | Distribution shifts               | Robust adaptation        |
");

// Training simulation
Console.WriteLine(new string('=', 60));
Console.WriteLine("Simulated Training Progress");
Console.WriteLine(new string('=', 60));

var random = new Random(42);
double currentLoss = 2.3026;
var trainingLosses = new List<double>();

Console.WriteLine("\nEpoch  | Train Loss | Loss Delta | Status");
Console.WriteLine(new string('-', 50));

for (int epoch = 1; epoch <= 50; epoch++)
{
    double delta = currentLoss * (0.03 + random.NextDouble() * 0.02);
    currentLoss -= delta;
    trainingLosses.Add(currentLoss);

    if (epoch == 1 || epoch % 10 == 0 || epoch == 50)
    {
        string status = epoch < 20 ? "Warming up" :
                       epoch < 40 ? "Converging" : "Stabilizing";
        Console.WriteLine($"{epoch,5}  | {currentLoss,10:F4} | {-delta,10:F4} | {status}");
    }
}

Console.WriteLine(new string('-', 50));
Console.WriteLine($"Final Loss: {currentLoss:F4}");
Console.WriteLine($"Improvement: {((2.3026 - currentLoss) / 2.3026 * 100):F1}%");

// LoRA Rank comparison
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("LoRA Rank Comparison");
Console.WriteLine(new string('=', 60));

Console.WriteLine("\n| Rank | Trainable Params | Compression | Memory Savings |");
Console.WriteLine("|------|------------------|-------------|----------------|");

foreach (int rank in new[] { 1, 4, 8, 16, 32, 64 })
{
    int loraParams = rank * (inputSize + hiddenSize) +
                     rank * (hiddenSize + hiddenSize) +
                     rank * (hiddenSize + outputSize);
    double compression = (double)baseParams / loraParams;
    double savings = (1.0 - (double)loraParams / baseParams) * 100;

    Console.WriteLine($"|  {rank,3} | {loraParams,16:N0} | {compression,10:F1}x | {savings,13:F1}% |");
}

// LoRA merging demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("LoRA Adapter Merging");
Console.WriteLine(new string('=', 60));

Console.WriteLine(@"
After training, LoRA adapters can be merged back into the base model:

  Training:    y = W_0 * x + (B * A) * x * (alpha/rank)
                   -----     -----------
                   frozen    trainable

  After Merge: y = W_merged * x
               where W_merged = W_0 + B * A * (alpha/rank)

Benefits of merging:
  - No inference overhead (single matrix multiply)
  - Smaller model size (no separate adapter weights)
  - Same quality as with adapters
  - Can be re-quantized for deployment
");

try
{
    if (loraLayer1 is ILoRAAdapter<double> adapter1)
    {
        Console.WriteLine("Merging Layer 1...");
        var merged1 = adapter1.MergeToOriginalLayer();
        Console.WriteLine($"  [OK] Layer 1 merged successfully");
        Console.WriteLine($"       Type: {merged1.GetType().Name}");
    }

    if (loraLayer2 is ILoRAAdapter<double> adapter2)
    {
        Console.WriteLine("Merging Layer 2...");
        var merged2 = adapter2.MergeToOriginalLayer();
        Console.WriteLine($"  [OK] Layer 2 merged successfully");
    }

    if (loraLayer3 is ILoRAAdapter<double> adapter3)
    {
        Console.WriteLine("Merging Layer 3...");
        var merged3 = adapter3.MergeToOriginalLayer();
        Console.WriteLine($"  [OK] Layer 3 merged successfully");
    }

    Console.WriteLine("\nMerging complete!");
    Console.WriteLine($"  - Merged model parameters: {baseParams:N0} (same as base)");
    Console.WriteLine($"  - LoRA adaptations baked into weights");
    Console.WriteLine($"  - No additional inference overhead");
}
catch (Exception ex)
{
    Console.WriteLine($"  Merging demonstration: {ex.Message}");
}

// Best practices
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("LoRA Best Practices");
Console.WriteLine(new string('=', 60));

Console.WriteLine(@"
1. Choosing Rank:
   - Start with rank=8 (good default for most tasks)
   - Increase to 16-32 if model underfits
   - Decrease to 4 for extreme efficiency

2. Alpha Selection:
   - Common practice: alpha = rank (scaling factor = 1.0)
   - Higher alpha = stronger adaptation
   - Lower alpha = more conservative updates

3. Which Layers to Adapt:
   - Attention Q, K, V projections (most important)
   - Feed-forward layers (good for task-specific knowledge)
   - Skip normalization layers (no learnable weights)

4. Learning Rate:
   - LoRA often needs higher LR than full fine-tuning
   - Try 1e-4 to 1e-3 (10x higher than full fine-tuning)

5. Training Stability:
   - Use gradient clipping (max_norm=1.0)
   - Warm up learning rate for first 10% of steps
   - Monitor loss - should decrease steadily
");

Console.WriteLine("\n=== Sample Complete ===");
