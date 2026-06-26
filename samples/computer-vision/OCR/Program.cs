// AiDotNet — Optical Character Recognition (OCR)
//
// Character recognition through the AiModelBuilder facade. A classifier is trained
// on small synthetic digit glyphs (0-9) via ConfigureModel + ConfigureDataLoader +
// BuildAsync, then a sequence of glyphs is "read" into text through result.Predict.
// Kept small so it trains and runs quickly in CI; production OCR pairs a text
// detector with a recognizer over full document images.

using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

const int numDigits = 10;
const int gh = 8, gw = 8;
const int px = gh * gw;

Console.WriteLine("=== AiDotNet Optical Character Recognition (OCR) ===");
Console.WriteLine("Recognizing synthetic 8x8 digit glyphs through the facade\n");

// ── 1. Build a labelled glyph dataset (features = pixels, label = digit) ───
var rng = new Random(42);
const int perDigit = 40;
var features = new List<float[]>();
var labels = new List<float>();
for (int d = 0; d < numDigits; d++)
    for (int k = 0; k < perDigit; k++)
    {
        features.Add(Glyph(d, rng));
        labels.Add(d);
    }

// ── 2. Train the recognizer through the facade ─────────────────────────────
Console.WriteLine("Training the recognizer through AiModelBuilder.ConfigureModel ...");
var result = await new AiModelBuilder<float, Matrix<float>, Vector<float>>()
    .ConfigureModel(new RandomForestClassifier<float>(
        new RandomForestClassifierOptions<float> { NEstimators = 50, MinSamplesSplit = 2 }))
    .ConfigureDataLoader(DataLoaders.FromArrays(features.ToArray(), labels.ToArray()))
    .BuildAsync();
Console.WriteLine("  Training complete.\n");

// ── 3. "Read" a sequence of glyphs into text through result.Predict ────────
int[] message = { 4, 2, 0, 1, 3, 7 };
var seq = message.Select(d => Glyph(d, rng)).ToArray();
var predictions = result.Predict(ToMatrix(seq));   // Vector<float> of class indices

var recognized = new System.Text.StringBuilder();
int correct = 0;
for (int i = 0; i < message.Length; i++)
{
    int digit = Math.Clamp((int)Math.Round(predictions[i]), 0, numDigits - 1);
    recognized.Append(digit);
    if (digit == message[i]) correct++;
}

Console.WriteLine($"  Expected:   {string.Concat(message)}");
Console.WriteLine($"  Recognized: {recognized}");
Console.WriteLine($"  Characters correct: {correct}/{message.Length}\n");

Console.WriteLine("=== Sample Complete ===");

// Pack a jagged feature array into the dense Matrix the model's Predict expects.
static Matrix<float> ToMatrix(float[][] rows)
{
    var m = new Matrix<float>(rows.Length, rows[0].Length);
    for (int i = 0; i < rows.Length; i++)
        for (int j = 0; j < rows[0].Length; j++)
            m[i, j] = rows[i][j];
    return m;
}

// A distinct glyph per digit: each digit lights its own 2x2 block in a 4x3 grid,
// with light pixel noise so the recognizer must generalize rather than memorize.
static float[] Glyph(int d, Random rng)
{
    int blockRow = (d / 4) * 2;   // 0, 2, 4
    int blockCol = (d % 4) * 2;   // 0, 2, 4, 6
    var g = new float[px];
    for (int y = 0; y < gh; y++)
        for (int x = 0; x < gw; x++)
        {
            bool on = y >= blockRow && y < blockRow + 2 && x >= blockCol && x < blockCol + 2;
            g[y * gw + x] = Math.Clamp((on ? 0.95f : 0.05f) + (float)(rng.NextDouble() - 0.5) * 0.1f, 0, 1);
        }
    return g;
}
