using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AdnTensor = AiDotNet.Tensors.LinearAlgebra.Tensor<float>;

Console.WriteLine("=== AiDotNet → ONNX Export Demo: Telco Customer Churn ===");
Console.WriteLine("Builds a small binary classifier and exports it to .onnx.");
Console.WriteLine("The resulting .onnx is the artifact the Databricks notebook loads.\n");

// ── Configuration ────────────────────────────────────────────────────────────
const int InputFeatures = 4;       // [tenure, monthlyCharges, totalCharges, contractType]
const int HiddenUnits   = 8;
const int OutputUnits   = 1;       // binary churn probability
const int NumSamples    = 2000;
var rng = new Random(42);

// ── 1. Generate synthetic Telco-Churn-shaped data ────────────────────────────
Console.WriteLine($"[1/4] Generating {NumSamples} synthetic Telco Churn samples...");

var features = new AdnTensor(new[] { NumSamples, InputFeatures });
var labels   = new AdnTensor(new[] { NumSamples, OutputUnits });

for (int i = 0; i < NumSamples; i++)
{
    float tenureMonths    = (float)(rng.NextDouble() * 71 + 1);
    float monthlyCharges  = (float)(rng.NextDouble() * 100 + 20);
    float totalCharges    = tenureMonths * monthlyCharges * (float)(0.85 + rng.NextDouble() * 0.30);
    float contractTypeId  = rng.Next(3);

    features[i, 0] = (tenureMonths - 36f) / 36f;
    features[i, 1] = (monthlyCharges - 70f) / 50f;
    features[i, 2] = (totalCharges - 2500f) / 2500f;
    features[i, 3] = (contractTypeId - 1f);

    float linear =
        -1.2f * features[i, 0]
        + 0.8f * features[i, 1]
        - 0.6f * features[i, 3]
        + (float)(rng.NextDouble() - 0.5) * 0.5f;
    float prob = 1f / (1f + MathF.Exp(-linear));
    labels[i, 0] = rng.NextDouble() < prob ? 1f : 0f;
}

int positives = 0;
for (int i = 0; i < NumSamples; i++) if (labels[i, 0] > 0.5f) positives++;
Console.WriteLine($"      Churn rate: {(double)positives / NumSamples:P1} ({positives}/{NumSamples})");

// ── 2. Build a layer chain directly (no NeuralNetwork wrapper) ───────────────
//
// NOTE FOR THE DEMO: the AiModelBuilder facade and NeuralNetwork's Train()
// path are the conventional way to train a model in AiDotNet, but those
// surfaces have ongoing churn across the codebase. For the Databricks demo
// the *exportable artifact* (.onnx) is what matters: the .onnx loads in any
// onnxruntime regardless of how the weights got there.
//
// This sample therefore constructs a layer chain directly, warms it up
// (which initialises the lazy weights), exports to .onnx, and emits a CSV
// of expected outputs. A future version of this sample can swap in real
// training once the AiModelBuilder + NeuralNetwork training path is locked
// down — the export step is unchanged.
Console.WriteLine($"\n[2/4] Building network: {InputFeatures} → {HiddenUnits} (Dense+ReLU) → {OutputUnits} (Dense+Sigmoid)");

var layers = new LayerBase<float>[]
{
    new DenseLayer<float>(HiddenUnits, (AiDotNet.Interfaces.IActivationFunction<float>)new ReLUActivation<float>()),
    new DenseLayer<float>(OutputUnits, (AiDotNet.Interfaces.IActivationFunction<float>)new SigmoidActivation<float>()),
};

// Warm-up forward pass — materialises the lazy weights so the export sees real numbers.
{
    var x = new AdnTensor(new[] { 1, InputFeatures });
    for (int f = 0; f < InputFeatures; f++) x[0, f] = features[0, f];
    foreach (var l in layers) x = l.Forward(x);
}
Console.WriteLine($"      Layers materialised with deterministic random weights (seed 42).");

// ── 3. Score the synthetic data so the CSV records what THIS model predicts ──
Console.WriteLine($"\n[3/4] Scoring {NumSamples} samples to capture expected outputs...");

var predictions = new float[NumSamples];
for (int i = 0; i < NumSamples; i++)
{
    var x = new AdnTensor(new[] { 1, InputFeatures });
    for (int f = 0; f < InputFeatures; f++) x[0, f] = features[i, f];
    foreach (var l in layers) x = l.Forward(x);
    predictions[i] = x[0, 0];
}

// Quick "accuracy" if we treat 0.5 as the decision threshold (random-weight
// baseline; not a real metric).
int correct = 0;
for (int i = 0; i < NumSamples; i++)
{
    if ((predictions[i] >= 0.5f) == (labels[i, 0] >= 0.5f)) correct++;
}
Console.WriteLine($"      Baseline accuracy with random init: {(double)correct / NumSamples:P2}");
Console.WriteLine("      (This is the random-weight baseline. Replace with a real training step for a useful model.)");

// ── 4. Export to ONNX via the new protobuf-based ConvertToOnnx path ─────────
const string OutputPath = "telco_churn.onnx";
Console.WriteLine($"\n[4/4] Exporting to {OutputPath} via the new ConvertToOnnx path...");

var builder = new OnnxGraphBuilder(new OnnxExportOptions
{
    ProducerVersion = "0.1.0",
    ModelDescription = "AiDotNet Telco Customer Churn binary classifier (demo for Databricks training session)",
});
builder.AddFloatInput("input", new[] { -1, InputFeatures });

var currentInputs = new OnnxLayerInputs("input");
foreach (var l in layers)
{
    var outputs = l.ConvertToOnnx(builder, currentInputs);
    currentInputs = new OnnxLayerInputs(outputs.Primary);
}
builder.AddFloatOutput(currentInputs.Primary, new[] { -1, OutputUnits });

using (var fs = File.Create(OutputPath))
{
    builder.WriteTo(fs);
}

var fi = new FileInfo(OutputPath);
Console.WriteLine($"      Wrote {fi.Length:N0} bytes to {fi.FullName}");

// ── Bonus: CSV of inputs + expected outputs for the Databricks notebook ──────
const string CsvPath = "telco_churn_test_data.csv";
Console.WriteLine($"\n[bonus] Writing {CsvPath} for the Databricks notebook to load...");

using (var sw = new StreamWriter(CsvPath))
{
    sw.WriteLine("tenure_norm,monthly_norm,total_norm,contract_norm,churn_actual,churn_predicted");
    int rowsWritten = Math.Min(200, NumSamples);
    for (int i = 0; i < rowsWritten; i++)
    {
        sw.WriteLine(
            $"{features[i, 0]:F4},{features[i, 1]:F4},{features[i, 2]:F4},{features[i, 3]:F4}," +
            $"{(int)labels[i, 0]},{predictions[i]:F4}");
    }
    Console.WriteLine($"      Wrote {rowsWritten} sample rows.");
}

Console.WriteLine("\n=== Done ===");
Console.WriteLine("Artifacts produced in the current directory:");
Console.WriteLine($"  - {OutputPath}             (the model for Databricks)");
Console.WriteLine($"  - {CsvPath}    (sample inputs + expected outputs for Databricks)");
Console.WriteLine();
Console.WriteLine("Upload both to a Databricks workspace. The notebook loads the .onnx with");
Console.WriteLine("onnxruntime and scores the .csv; predictions should match the churn_predicted column.");
