// AiDotNet — SimCLR Self-Supervised Learning
//
// Self-supervised pretraining through the AiModelBuilder facade:
// ConfigureSelfSupervisedLearning selects the SSL method (SimCLR) and the
// pretraining schedule; BuildAsync wires the SSL stage ahead of supervised
// training and returns an AiModelResult you predict through.

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;                    // SSLMethodType
using AiDotNet.Regression;               // RidgeRegression
using AiDotNet.SelfSupervisedLearning;   // SSLConfig
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("===========================================");
Console.WriteLine("  SimCLR - Self-Supervised Learning Demo  ");
Console.WriteLine("===========================================\n");

// ── Synthetic feature dataset ──────────────────────────────────────────────
const int rows = 64;
const int features = 8;
var rng = new Random(42);
var xData = new double[rows, features];
var yData = new double[rows];
for (int r = 0; r < rows; r++)
{
    double sum = 0;
    for (int c = 0; c < features; c++)
    {
        xData[r, c] = rng.NextDouble() * 2 - 1;
        sum += xData[r, c];
    }
    yData[r] = sum;
}
var x = new Matrix<double>(xData);
var y = new Vector<double>(yData);

Console.WriteLine($"Dataset: {rows} samples, {features} features");
Console.WriteLine("Pretraining method: SimCLR (contrastive)\n");

// ── Configure SSL pretraining through the facade ───────────────────────────
Console.WriteLine("Training through AiModelBuilder.ConfigureSelfSupervisedLearning ...");
try
{
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, y))
        .ConfigureModel(new RidgeRegression<double>())
        .ConfigureSelfSupervisedLearning(cfg =>
        {
            cfg.Method = SSLMethodType.SimCLR;
            cfg.PretrainingEpochs = 1;
        })
        .BuildAsync();

    Console.WriteLine("  Training complete.");
    var preds = result.Predict(x);
    Console.WriteLine($"  Predicted {preds.Length} values; first three: {preds[0]:F3}, {preds[1]:F3}, {preds[2]:F3}");
}
catch (Exception ex)
{
    Console.WriteLine($"  SSL pretraining reported: {ex.Message}");
}

Console.WriteLine(@"
SimCLR learns representations by pulling augmented views of the same sample
together and pushing different samples apart — no labels are needed for the
pretraining stage; the configured base model is then trained on top.
");

Console.WriteLine("=== Sample Complete ===");
