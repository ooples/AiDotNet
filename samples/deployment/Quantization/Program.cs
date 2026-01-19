using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Quantization;
using AiDotNet.Quantization.Calibration;
using AiDotNet.LinearAlgebra;
using AiDotNet.Optimizers;
using AiDotNet.LossFunctions;
using System.Diagnostics;

Console.WriteLine("==========================================");
Console.WriteLine("  Model Quantization Demo                 ");
Console.WriteLine("==========================================\n");

// ============================================
// Step 1: Create and train a model
// ============================================
Console.WriteLine("Step 1: Creating and training a neural network...\n");

const int inputSize = 784;  // MNIST-like
const int hiddenSize = 256;
const int numClasses = 10;
const int numSamples = 1000;
const int epochs = 20;
var random = new Random(42);

// Create a simple MLP
var model = new NeuralNetwork<float>(
    new NeuralNetworkArchitecture<float>(
        inputFeatures: inputSize,
        numClasses: numClasses,
        complexity: NetworkComplexity.Medium));

// Add layers manually for more control
model.ClearLayers();
model.AddLayer(new DenseLayer<float>(inputSize, hiddenSize));
model.AddLayer(new ReLUActivation<float>());
model.AddLayer(new BatchNormalization<float>(hiddenSize));
model.AddLayer(new DropoutLayer<float>(0.2f));
model.AddLayer(new DenseLayer<float>(hiddenSize, hiddenSize / 2));
model.AddLayer(new ReLUActivation<float>());
model.AddLayer(new DenseLayer<float>(hiddenSize / 2, numClasses));
model.AddLayer(new SoftmaxActivation<float>());

Console.WriteLine("  Model architecture:");
Console.WriteLine($"    Input: {inputSize}");
Console.WriteLine($"    Hidden: {hiddenSize} -> {hiddenSize / 2}");
Console.WriteLine($"    Output: {numClasses}");
Console.WriteLine($"    Parameters: {model.GetParameterCount():N0}");

// Generate synthetic training data
Console.WriteLine("\n  Generating synthetic training data...");
var trainFeatures = new Tensor<float>([numSamples, inputSize]);
var trainLabels = new Tensor<float>([numSamples, numClasses]);

for (int i = 0; i < numSamples; i++)
{
    int label = i % numClasses;
    // Create digit-like patterns
    for (int j = 0; j < inputSize; j++)
    {
        float baseValue = ((j + label * 50) % 100) < 50 ? 0.8f : 0.2f;
        trainFeatures[[i, j]] = baseValue + (random.NextSingle() - 0.5f) * 0.2f;
    }
    trainLabels[[i, label]] = 1.0f;
}

// Train the model
Console.WriteLine($"  Training for {epochs} epochs...");

var optimizer = new AdamOptimizer<float>(learningRate: 0.001f);
var lossFunction = new CrossEntropyLoss<float>();

for (int epoch = 0; epoch < epochs; epoch++)
{
    model.Train(trainFeatures, trainLabels);

    if ((epoch + 1) % 5 == 0)
    {
        var predictions = model.Predict(trainFeatures);
        float accuracy = ComputeAccuracy(predictions, trainLabels);
        Console.WriteLine($"    Epoch {epoch + 1}/{epochs}: Accuracy = {accuracy:P1}");
    }
}

// ============================================
// Step 2: Evaluate baseline model
// ============================================
Console.WriteLine("\nStep 2: Evaluating baseline (FP32) model...\n");

// Generate test data
var testFeatures = new Tensor<float>([200, inputSize]);
var testLabels = new Tensor<float>([200, numClasses]);

for (int i = 0; i < 200; i++)
{
    int label = i % numClasses;
    for (int j = 0; j < inputSize; j++)
    {
        float baseValue = ((j + label * 50) % 100) < 50 ? 0.8f : 0.2f;
        testFeatures[[i, j]] = baseValue + (random.NextSingle() - 0.5f) * 0.25f;
    }
    testLabels[[i, label]] = 1.0f;
}

// Measure baseline performance
var sw = Stopwatch.StartNew();
var baselinePredictions = model.Predict(testFeatures);
sw.Stop();
float baselineAccuracy = ComputeAccuracy(baselinePredictions, testLabels);
double baselineLatency = sw.Elapsed.TotalMilliseconds;
long baselineSize = model.GetParameterCount() * sizeof(float);

Console.WriteLine($"  Baseline (FP32) Results:");
Console.WriteLine($"    Accuracy: {baselineAccuracy:P2}");
Console.WriteLine($"    Inference time: {baselineLatency:F2} ms");
Console.WriteLine($"    Model size: {baselineSize / 1024.0:F1} KB");

// ============================================
// Step 3: Post-Training Quantization (PTQ) to INT8
// ============================================
Console.WriteLine("\nStep 3: Applying Post-Training Quantization (INT8)...\n");

// Create calibration dataset (subset of training data)
Console.WriteLine("  Creating calibration dataset...");
var calibrationData = new Tensor<float>([100, inputSize]);
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < inputSize; j++)
    {
        calibrationData[[i, j]] = trainFeatures[[i, j]];
    }
}

// Configure INT8 quantization
var ptqConfig = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.INT8,
    CalibrationMethod = CalibrationMethod.MinMax, // or Histogram, Percentile
    CalibrationSamples = 100,
    PerChannelQuantization = true,
    SymmetricQuantization = true,

    // Layer-specific settings
    LayerConfig = new Dictionary<string, LayerQuantConfig>
    {
        // Keep first and last layers in higher precision
        ["input"] = new LayerQuantConfig { QuantizationType = QuantizationType.FP16 },
        ["output"] = new LayerQuantConfig { QuantizationType = QuantizationType.FP16 }
    }
};

// Create quantizer
var quantizer = new ModelQuantizer<float>(ptqConfig);

// Run calibration
Console.WriteLine("  Running calibration...");
var calibrationResult = quantizer.Calibrate(model, calibrationData);
Console.WriteLine($"    Calibrated {calibrationResult.LayerCount} layers");
Console.WriteLine($"    Scale range: [{calibrationResult.MinScale:F6}, {calibrationResult.MaxScale:F6}]");

// Quantize model
Console.WriteLine("  Quantizing model...");
var int8Model = quantizer.Quantize(model);

// Evaluate INT8 model
sw.Restart();
var int8Predictions = int8Model.Predict(testFeatures);
sw.Stop();
float int8Accuracy = ComputeAccuracy(int8Predictions, testLabels);
double int8Latency = sw.Elapsed.TotalMilliseconds;
long int8Size = int8Model.GetParameterCount() * sizeof(byte); // Approximate

Console.WriteLine($"\n  INT8 PTQ Results:");
Console.WriteLine($"    Accuracy: {int8Accuracy:P2} ({(int8Accuracy - baselineAccuracy) * 100:+0.00;-0.00}%)");
Console.WriteLine($"    Inference time: {int8Latency:F2} ms ({baselineLatency / int8Latency:F1}x speedup)");
Console.WriteLine($"    Model size: {int8Size / 1024.0:F1} KB ({baselineSize / (float)int8Size:F1}x smaller)");

// ============================================
// Step 4: FP16 Quantization
// ============================================
Console.WriteLine("\nStep 4: Applying FP16 Quantization...\n");

var fp16Config = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.FP16,
    // FP16 doesn't need calibration
};

var fp16Quantizer = new ModelQuantizer<float>(fp16Config);
var fp16Model = fp16Quantizer.Quantize(model);

// Evaluate FP16 model
sw.Restart();
var fp16Predictions = fp16Model.Predict(testFeatures);
sw.Stop();
float fp16Accuracy = ComputeAccuracy(fp16Predictions, testLabels);
double fp16Latency = sw.Elapsed.TotalMilliseconds;
long fp16Size = fp16Model.GetParameterCount() * 2; // FP16 = 2 bytes

Console.WriteLine($"  FP16 Results:");
Console.WriteLine($"    Accuracy: {fp16Accuracy:P2} ({(fp16Accuracy - baselineAccuracy) * 100:+0.00;-0.00}%)");
Console.WriteLine($"    Inference time: {fp16Latency:F2} ms ({baselineLatency / fp16Latency:F1}x speedup)");
Console.WriteLine($"    Model size: {fp16Size / 1024.0:F1} KB ({baselineSize / (float)fp16Size:F1}x smaller)");

// ============================================
// Step 5: Quantization-Aware Training (QAT)
// ============================================
Console.WriteLine("\nStep 5: Quantization-Aware Training (QAT)...\n");

// Create a fresh model for QAT
var qatModel = new NeuralNetwork<float>(
    new NeuralNetworkArchitecture<float>(
        inputFeatures: inputSize,
        numClasses: numClasses,
        complexity: NetworkComplexity.Medium));

qatModel.ClearLayers();
qatModel.AddLayer(new DenseLayer<float>(inputSize, hiddenSize));
qatModel.AddLayer(new ReLUActivation<float>());
qatModel.AddLayer(new BatchNormalization<float>(hiddenSize));
qatModel.AddLayer(new DenseLayer<float>(hiddenSize, hiddenSize / 2));
qatModel.AddLayer(new ReLUActivation<float>());
qatModel.AddLayer(new DenseLayer<float>(hiddenSize / 2, numClasses));
qatModel.AddLayer(new SoftmaxActivation<float>());

// Configure QAT
var qatConfig = new QATConfig<float>
{
    TargetQuantization = QuantizationType.INT8,
    SimulateQuantization = true,
    FreezeBackboneEpochs = 5, // Train normally first
    QATEpochs = 10,           // Then with fake quantization
    GradientScaling = true,
    StraightThroughEstimator = true
};

Console.WriteLine("  Phase 1: Pre-training without quantization simulation...");

// Pre-train
for (int epoch = 0; epoch < qatConfig.FreezeBackboneEpochs; epoch++)
{
    qatModel.Train(trainFeatures, trainLabels);
}

var preQatPredictions = qatModel.Predict(testFeatures);
float preQatAccuracy = ComputeAccuracy(preQatPredictions, testLabels);
Console.WriteLine($"    Pre-QAT accuracy: {preQatAccuracy:P2}");

// Enable fake quantization for QAT
Console.WriteLine("\n  Phase 2: Training with quantization simulation...");
var qatWrapper = new QATWrapper<float>(qatModel, qatConfig);
qatWrapper.EnableFakeQuantization();

for (int epoch = 0; epoch < qatConfig.QATEpochs; epoch++)
{
    qatWrapper.Train(trainFeatures, trainLabels);

    if ((epoch + 1) % 5 == 0)
    {
        var qatPreds = qatWrapper.Predict(testFeatures);
        float qatAcc = ComputeAccuracy(qatPreds, testLabels);
        Console.WriteLine($"    QAT Epoch {epoch + 1}/{qatConfig.QATEpochs}: Accuracy = {qatAcc:P2}");
    }
}

// Convert QAT model to quantized model
Console.WriteLine("\n  Converting QAT model to INT8...");
var qatQuantizedModel = qatWrapper.ConvertToQuantized();

// Evaluate QAT model
sw.Restart();
var qatQuantizedPredictions = qatQuantizedModel.Predict(testFeatures);
sw.Stop();
float qatAccuracy = ComputeAccuracy(qatQuantizedPredictions, testLabels);
double qatLatency = sw.Elapsed.TotalMilliseconds;
long qatSize = qatQuantizedModel.GetParameterCount() * sizeof(byte);

Console.WriteLine($"\n  QAT INT8 Results:");
Console.WriteLine($"    Accuracy: {qatAccuracy:P2} ({(qatAccuracy - baselineAccuracy) * 100:+0.00;-0.00}%)");
Console.WriteLine($"    Inference time: {qatLatency:F2} ms ({baselineLatency / qatLatency:F1}x speedup)");
Console.WriteLine($"    Model size: {qatSize / 1024.0:F1} KB ({baselineSize / (float)qatSize:F1}x smaller)");

// ============================================
// Step 6: Dynamic Quantization
// ============================================
Console.WriteLine("\nStep 6: Dynamic Quantization...\n");

var dynamicConfig = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.Dynamic,
    // Weights are quantized statically, activations at runtime
    DynamicRangeQuantization = true
};

var dynamicQuantizer = new ModelQuantizer<float>(dynamicConfig);
var dynamicModel = dynamicQuantizer.Quantize(model);

// Evaluate dynamic quantization
sw.Restart();
var dynamicPredictions = dynamicModel.Predict(testFeatures);
sw.Stop();
float dynamicAccuracy = ComputeAccuracy(dynamicPredictions, testLabels);
double dynamicLatency = sw.Elapsed.TotalMilliseconds;

Console.WriteLine($"  Dynamic Quantization Results:");
Console.WriteLine($"    Accuracy: {dynamicAccuracy:P2} ({(dynamicAccuracy - baselineAccuracy) * 100:+0.00;-0.00}%)");
Console.WriteLine($"    Inference time: {dynamicLatency:F2} ms ({baselineLatency / dynamicLatency:F1}x speedup)");

// ============================================
// Step 7: Mixed Precision Quantization
// ============================================
Console.WriteLine("\nStep 7: Mixed Precision Quantization...\n");

var mixedConfig = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.Mixed,
    SensitivityAnalysis = true, // Automatically detect sensitive layers

    // Manual layer configuration
    LayerConfig = new Dictionary<string, LayerQuantConfig>
    {
        ["Dense_0"] = new LayerQuantConfig { QuantizationType = QuantizationType.FP16 }, // First layer
        ["Dense_1"] = new LayerQuantConfig { QuantizationType = QuantizationType.INT8 },
        ["Dense_2"] = new LayerQuantConfig { QuantizationType = QuantizationType.INT8 },
        ["output"] = new LayerQuantConfig { QuantizationType = QuantizationType.FP16 }   // Last layer
    }
};

var mixedQuantizer = new ModelQuantizer<float>(mixedConfig);

// Run sensitivity analysis
Console.WriteLine("  Running layer sensitivity analysis...");
var sensitivityResults = mixedQuantizer.AnalyzeSensitivity(model, calibrationData, testFeatures, testLabels);

Console.WriteLine("  Layer sensitivities:");
foreach (var (layer, sensitivity) in sensitivityResults.OrderByDescending(x => x.Value))
{
    string recommendation = sensitivity > 0.5 ? "FP16" : "INT8";
    Console.WriteLine($"    {layer}: {sensitivity:F3} -> Recommend {recommendation}");
}

// Apply mixed precision
var mixedModel = mixedQuantizer.Quantize(model);

sw.Restart();
var mixedPredictions = mixedModel.Predict(testFeatures);
sw.Stop();
float mixedAccuracy = ComputeAccuracy(mixedPredictions, testLabels);
double mixedLatency = sw.Elapsed.TotalMilliseconds;

Console.WriteLine($"\n  Mixed Precision Results:");
Console.WriteLine($"    Accuracy: {mixedAccuracy:P2} ({(mixedAccuracy - baselineAccuracy) * 100:+0.00;-0.00}%)");
Console.WriteLine($"    Inference time: {mixedLatency:F2} ms ({baselineLatency / mixedLatency:F1}x speedup)");

// ============================================
// Summary
// ============================================
Console.WriteLine("\n==========================================");
Console.WriteLine("  Quantization Results Summary            ");
Console.WriteLine("==========================================\n");

Console.WriteLine("  | Method          | Accuracy | Speedup | Size Reduction |");
Console.WriteLine("  |-----------------|----------|---------|----------------|");
Console.WriteLine($"  | FP32 (baseline) | {baselineAccuracy,7:P1} |   1.0x  |      1.0x      |");
Console.WriteLine($"  | FP16            | {fp16Accuracy,7:P1} |  {baselineLatency / fp16Latency,4:F1}x  |      2.0x      |");
Console.WriteLine($"  | INT8 (PTQ)      | {int8Accuracy,7:P1} |  {baselineLatency / int8Latency,4:F1}x  |      4.0x      |");
Console.WriteLine($"  | INT8 (QAT)      | {qatAccuracy,7:P1} |  {baselineLatency / qatLatency,4:F1}x  |      4.0x      |");
Console.WriteLine($"  | Dynamic         | {dynamicAccuracy,7:P1} |  {baselineLatency / dynamicLatency,4:F1}x  |      ~2x       |");
Console.WriteLine($"  | Mixed Precision | {mixedAccuracy,7:P1} |  {baselineLatency / mixedLatency,4:F1}x  |      ~3x       |");

Console.WriteLine("\n  Recommendations:");
Console.WriteLine("    - FP16: Minimal accuracy loss, good for GPU deployment");
Console.WriteLine("    - INT8 PTQ: Quick optimization, good for edge devices");
Console.WriteLine("    - INT8 QAT: Best INT8 accuracy, worth extra training time");
Console.WriteLine("    - Mixed: Balance accuracy and efficiency for sensitive models");

// Helper function
static float ComputeAccuracy(Tensor<float> predictions, Tensor<float> labels)
{
    int correct = 0;
    int total = predictions.Shape[0];

    for (int i = 0; i < total; i++)
    {
        int predictedClass = 0;
        int actualClass = 0;
        float maxPred = float.MinValue;
        float maxLabel = float.MinValue;

        for (int j = 0; j < predictions.Shape[1]; j++)
        {
            if (predictions[[i, j]] > maxPred)
            {
                maxPred = predictions[[i, j]];
                predictedClass = j;
            }
            if (labels[[i, j]] > maxLabel)
            {
                maxLabel = labels[[i, j]];
                actualClass = j;
            }
        }

        if (predictedClass == actualClass) correct++;
    }

    return (float)correct / total;
}
