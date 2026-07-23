// AiDotNet — Video Generation
//
// A tiny generative demo built entirely on the AiModelBuilder facade. A network is
// trained to map a frame's normalized time -> that frame's pixels (a learned
// animation of a dot sweeping across the frame); frames are then generated through
// result.Predict. Kept small so it trains and runs quickly in CI. (Production video
// diffusion models are far heavier and load pretrained weights.)

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

const int frames = 24;
const int size = 8;
const int px = size * size;

Console.WriteLine("=== AiDotNet Video Generation ===");
Console.WriteLine($"Learning a {size}x{size} animation over {frames} frames via the facade\n");

// ── 1. Synthetic training "video": a bright dot sweeping left to right ──────
var inX = new Tensor<float>(new[] { frames, 1 });
var outY = new Tensor<float>(new[] { frames, px });
for (int f = 0; f < frames; f++)
{
    inX[new[] { f, 0 }] = f / (float)(frames - 1);     // normalized time
    int cx = Math.Min(size - 1, f * size / frames);    // dot column for this frame
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++)
            outY[new[] { f, y * size + x }] = x == cx ? 1f : 0.02f;
}

// ── 2. Train the frame generator through the facade ────────────────────────
var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 1, numClasses: px, complexity: NetworkComplexity.Simple));

Console.WriteLine("Training frame generator through AiModelBuilder.ConfigureModel ...");
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(inX, outY))
    .BuildAsync();
Console.WriteLine("  Training complete.\n");

// ── 3. Generate frames through result.Predict ──────────────────────────────
Console.WriteLine("Generated frames (brightest column marks the dot):\n");
for (int f = 0; f < frames; f += 4)
{
    var input = new Tensor<float>(new[] { 1, 1 });
    input[new[] { 0, 0 }] = f / (float)(frames - 1);
    var frame = result.Predict(input);   // [1, px]

    // Brightest column = the generated dot position.
    int bestCol = 0;
    float bestColScore = float.MinValue;
    for (int x = 0; x < size; x++)
    {
        float colSum = 0;
        for (int y = 0; y < size; y++) colSum += frame[new[] { 0, y * size + x }];
        if (colSum > bestColScore) { bestColScore = colSum; bestCol = x; }
    }

    var bar = new string('.', size).ToCharArray();
    bar[bestCol] = '#';
    Console.WriteLine($"  frame {f,2} (t={input[new[] { 0, 0 }]:F2}): [{new string(bar)}]");
}

Console.WriteLine("\nEach frame was generated through result.Predict from its time input; a larger");
Console.WriteLine("model and more frames sharpen the learned motion.");
Console.WriteLine("\n=== Sample Complete ===");
