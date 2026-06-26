// AiDotNet — Image Classification
//
// A small image classifier trained and served entirely through the AiModelBuilder
// facade. Synthetic 16x16x3 images with distinct per-class patterns are flattened
// into feature vectors, a network is trained via ConfigureModel + ConfigureData
// Loader + BuildAsync, and predictions flow through result.Predict (then softmax +
// argmax for the top class). Kept small so it trains and runs quickly in CI.

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

const int channels = 3, height = 16, width = 16;
const int features = channels * height * width;
const int numClasses = 6;
string[] classNames = { "circle", "gradient", "horizontal", "vertical", "checker", "noise" };

Console.WriteLine("=== AiDotNet Image Classification ===");
Console.WriteLine($"Classifying {height}x{width}x{channels} pattern images into {numClasses} classes\n");

// ── 1. Build + train through the facade ────────────────────────────────────
var (trainX, trainY) = MakeDataset(600, seed: 42);
var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: features, numClasses: numClasses, complexity: NetworkComplexity.Simple));

Console.WriteLine("Training classifier through AiModelBuilder.ConfigureModel ...");
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();
Console.WriteLine("  Training complete.\n");

// ── 2. Classify held-out test images through result.Predict ────────────────
var (testX, testLabels) = MakeDataset(numClasses, seed: 7);   // one of each pattern
var predictions = result.Predict(testX);                       // [numClasses, numClasses]

Console.WriteLine("Predictions (top-3 per image):");
Console.WriteLine(new string('-', 56));
int correct = 0;
for (int i = 0; i < numClasses; i++)
{
    var scores = Softmax(predictions, i, numClasses);
    var ranked = Enumerable.Range(0, numClasses).OrderByDescending(c => scores[c]).ToArray();
    int actual = (int)Math.Round(ArgMaxRow(testLabels, i, numClasses));
    if (ranked[0] == actual) correct++;

    Console.WriteLine($"  actual = {classNames[actual],-10}");
    for (int k = 0; k < 3; k++)
        Console.WriteLine($"     #{k + 1}: {classNames[ranked[k]],-10} {scores[ranked[k]]:P1}");
}
Console.WriteLine(new string('-', 56));
Console.WriteLine($"Test accuracy: {correct}/{numClasses} = {100.0 * correct / numClasses:F0}%\n");

// ── Reference: common image-classification augmentations ───────────────────
Console.WriteLine("Typical training-time augmentations: RandomHorizontalFlip, RandomRotation,");
Console.WriteLine("RandomResizedCrop, ColorJitter, RandomErasing, Normalize, MixUp, CutMix.");
Console.WriteLine("\n=== Sample Complete ===");

// Build a flattened synthetic image dataset with a distinct pattern per class.
static (Tensor<float> x, Tensor<float> y) MakeDataset(int n, int seed)
{
    var rng = new Random(seed);
    var x = new Tensor<float>(new[] { n, features });
    var y = new Tensor<float>(new[] { n, numClasses });
    for (int s = 0; s < n; s++)
    {
        int label = s % numClasses;
        for (int c = 0; c < channels; c++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    float v = label switch
                    {
                        0 => MathF.Sqrt((h - 8f) * (h - 8f) + (w - 8f) * (w - 8f)) < 6 ? 0.9f : 0.1f, // circle
                        1 => (h + w) / 30f,                                                            // gradient
                        2 => h < 8 ? 0.85f : 0.15f,                                                    // horizontal split
                        3 => w < 8 ? 0.85f : 0.15f,                                                    // vertical split
                        4 => (h / 4 + w / 4) % 2 == 0 ? 0.9f : 0.1f,                                   // checkerboard
                        _ => (float)rng.NextDouble()                                                  // noise
                    };
                    x[new[] { s, c * height * width + h * width + w }] = Math.Clamp(v + (float)(rng.NextDouble() - 0.5) * 0.1f, 0, 1);
                }
        y[new[] { s, label }] = 1f;
    }
    return (x, y);
}

static float[] Softmax(Tensor<float> t, int row, int cols)
{
    var raw = new float[cols];
    for (int c = 0; c < cols; c++) raw[c] = t[new[] { row, c }];
    float max = raw.Max(), sum = 0;
    for (int c = 0; c < cols; c++) { raw[c] = MathF.Exp(raw[c] - max); sum += raw[c]; }
    for (int c = 0; c < cols; c++) raw[c] /= sum;
    return raw;
}

static float ArgMaxRow(Tensor<float> t, int row, int cols)
{
    int best = 0;
    for (int c = 1; c < cols; c++) if (t[new[] { row, c }] > t[new[] { row, best }]) best = c;
    return best;
}
