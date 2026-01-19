using AiDotNet;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.SelfSupervisedLearning.Contrastive;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Architectures;
using AiDotNet.Optimizers;
using AiDotNet.LossFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.DataAugmentation;

Console.WriteLine("===========================================");
Console.WriteLine("  SimCLR - Self-Supervised Learning Demo  ");
Console.WriteLine("===========================================\n");

// ============================================
// Step 1: Generate synthetic unlabeled images
// ============================================
Console.WriteLine("Step 1: Generating synthetic unlabeled image data...");

const int numUnlabeledImages = 1000;
const int imageHeight = 32;
const int imageWidth = 32;
const int channels = 3;
const int batchSize = 64;
const int pretrainEpochs = 100;
const int finetuneEpochs = 50;
const int numClasses = 10;
const float temperature = 0.5f;
const int projectionDim = 128;

var random = new Random(42);

// Generate unlabeled images (simulating real unlabeled data)
var unlabeledImages = new List<Tensor<float>>();
for (int i = 0; i < numUnlabeledImages; i++)
{
    var image = new Tensor<float>([channels, imageHeight, imageWidth]);
    // Generate synthetic patterns
    int pattern = random.Next(10);
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imageHeight; h++)
        {
            for (int w = 0; w < imageWidth; w++)
            {
                float value = pattern switch
                {
                    0 => (float)(h + w) / (imageHeight + imageWidth), // Diagonal gradient
                    1 => (float)Math.Sin(h * 0.5) * 0.5f + 0.5f,      // Horizontal waves
                    2 => (float)Math.Sin(w * 0.5) * 0.5f + 0.5f,      // Vertical waves
                    3 => h < imageHeight / 2 ? 0.8f : 0.2f,           // Top/bottom split
                    4 => w < imageWidth / 2 ? 0.8f : 0.2f,            // Left/right split
                    5 => (h + w) % 4 < 2 ? 0.9f : 0.1f,               // Checkerboard
                    6 => (float)Math.Sqrt(Math.Pow(h - imageHeight/2, 2) + Math.Pow(w - imageWidth/2, 2)) / 20f,
                    7 => random.NextSingle() * 0.3f + 0.35f,          // Noise with bias
                    8 => h % 8 < 4 ? 0.7f : 0.3f,                     // Horizontal stripes
                    _ => w % 8 < 4 ? 0.7f : 0.3f,                     // Vertical stripes
                };
                image[[c, h, w]] = value + (random.NextSingle() - 0.5f) * 0.1f;
            }
        }
    }
    unlabeledImages.Add(image);
}

Console.WriteLine($"  Generated {numUnlabeledImages} unlabeled images ({imageHeight}x{imageWidth}x{channels})");

// ============================================
// Step 2: Configure data augmentation pipeline
// ============================================
Console.WriteLine("\nStep 2: Configuring augmentation pipeline...");

var augmentationConfig = new DataAugmentationConfig<float>
{
    // Random resized crop (most important augmentation for SimCLR)
    RandomCrop = new RandomCropConfig
    {
        Enabled = true,
        MinScale = 0.08f,
        MaxScale = 1.0f,
        AspectRatioMin = 0.75f,
        AspectRatioMax = 1.33f
    },

    // Color jittering
    ColorJitter = new ColorJitterConfig
    {
        Enabled = true,
        BrightnessRange = 0.8f,
        ContrastRange = 0.8f,
        SaturationRange = 0.8f,
        HueRange = 0.2f
    },

    // Gaussian blur
    GaussianBlur = new GaussianBlurConfig
    {
        Enabled = true,
        Probability = 0.5f,
        KernelSize = 3,
        SigmaMin = 0.1f,
        SigmaMax = 2.0f
    },

    // Random horizontal flip
    RandomHorizontalFlip = new RandomFlipConfig
    {
        Enabled = true,
        Probability = 0.5f
    },

    // Grayscale conversion (with probability)
    RandomGrayscale = new RandomGrayscaleConfig
    {
        Enabled = true,
        Probability = 0.2f
    }
};

Console.WriteLine("  Configured augmentations:");
Console.WriteLine("    - Random resized crop (scale: 0.08-1.0)");
Console.WriteLine("    - Color jittering (brightness, contrast, saturation, hue)");
Console.WriteLine("    - Gaussian blur (p=0.5)");
Console.WriteLine("    - Random horizontal flip (p=0.5)");
Console.WriteLine("    - Random grayscale (p=0.2)");

// ============================================
// Step 3: Build SimCLR model
// ============================================
Console.WriteLine("\nStep 3: Building SimCLR model...");

// Base encoder (ResNet-18 style)
var encoderConfig = new ResNetConfig<float>
{
    InputChannels = channels,
    InputHeight = imageHeight,
    InputWidth = imageWidth,
    NumBlocks = [2, 2, 2, 2], // ResNet-18 configuration
    NumFilters = [64, 128, 256, 512],
    UseBottleneck = false
};

// SimCLR configuration
var simclrConfig = new SimCLRConfig<float>
{
    // Encoder settings
    EncoderConfig = encoderConfig,
    EncoderOutputDim = 512,

    // Projection head (MLP)
    ProjectionHiddenDim = 512,
    ProjectionOutputDim = projectionDim,

    // Contrastive learning settings
    Temperature = temperature,
    UseLARSOptimizer = true,

    // Training settings
    BatchSize = batchSize,
    PretrainEpochs = pretrainEpochs
};

// Build SimCLR model
var simclr = new SimCLR<float>(simclrConfig);

Console.WriteLine($"  Encoder: ResNet-18 (output dim: {simclrConfig.EncoderOutputDim})");
Console.WriteLine($"  Projection head: MLP (512 -> {projectionDim})");
Console.WriteLine($"  Temperature: {temperature}");
Console.WriteLine($"  Total parameters: {simclr.GetParameterCount():N0}");

// ============================================
// Step 4: Pre-train with contrastive learning
// ============================================
Console.WriteLine("\nStep 4: Pre-training with contrastive learning...");
Console.WriteLine("  (This learns visual representations without labels)\n");

// Configure optimizer (LARS is recommended for SimCLR)
var optimizer = new LARSOptimizer<float>(
    learningRate: 0.3f * batchSize / 256, // Linear scaling rule
    momentum: 0.9f,
    weightDecay: 1e-6f,
    trustCoefficient: 0.001f
);

// Configure NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
var contrastiveLoss = new NTXentLoss<float>(temperature);

// Training loop
var pretrainLosses = new List<float>();

for (int epoch = 0; epoch < pretrainEpochs; epoch++)
{
    float epochLoss = 0;
    int numBatches = 0;

    // Shuffle data
    var shuffledIndices = Enumerable.Range(0, numUnlabeledImages)
        .OrderBy(_ => random.Next())
        .ToList();

    for (int batchStart = 0; batchStart < numUnlabeledImages; batchStart += batchSize)
    {
        int actualBatchSize = Math.Min(batchSize, numUnlabeledImages - batchStart);
        if (actualBatchSize < 2) continue; // Need at least 2 samples for contrastive learning

        // Create batch tensors
        var batchImages = new Tensor<float>([actualBatchSize, channels, imageHeight, imageWidth]);

        for (int i = 0; i < actualBatchSize; i++)
        {
            int idx = shuffledIndices[batchStart + i];
            var image = unlabeledImages[idx];
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < imageHeight; h++)
                {
                    for (int w = 0; w < imageWidth; w++)
                    {
                        batchImages[[i, c, h, w]] = image[[c, h, w]];
                    }
                }
            }
        }

        // Create two augmented views of each image
        var augmentedView1 = simclr.Augment(batchImages, augmentationConfig);
        var augmentedView2 = simclr.Augment(batchImages, augmentationConfig);

        // Forward pass through encoder and projection head
        var z1 = simclr.Forward(augmentedView1);
        var z2 = simclr.Forward(augmentedView2);

        // Compute contrastive loss
        float loss = contrastiveLoss.Compute(z1, z2);
        epochLoss += loss;
        numBatches++;

        // Backward pass and update
        simclr.Backward(contrastiveLoss);
        optimizer.Step(simclr);
    }

    float avgLoss = epochLoss / numBatches;
    pretrainLosses.Add(avgLoss);

    if ((epoch + 1) % 10 == 0 || epoch == 0)
    {
        Console.WriteLine($"  Epoch {epoch + 1,3}/{pretrainEpochs}: Loss = {avgLoss:F4}");
    }
}

Console.WriteLine($"\n  Pre-training complete!");
Console.WriteLine($"  Final contrastive loss: {pretrainLosses.Last():F4}");

// ============================================
// Step 5: Generate labeled data for fine-tuning
// ============================================
Console.WriteLine("\nStep 5: Generating labeled data for fine-tuning...");

const int numLabeledImages = 200; // Small labeled dataset
var labeledImages = new List<Tensor<float>>();
var labels = new List<int>();

// Generate labeled images (same patterns but with labels)
for (int i = 0; i < numLabeledImages; i++)
{
    var image = new Tensor<float>([channels, imageHeight, imageWidth]);
    int label = i % numClasses;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imageHeight; h++)
        {
            for (int w = 0; w < imageWidth; w++)
            {
                float value = label switch
                {
                    0 => (float)(h + w) / (imageHeight + imageWidth),
                    1 => (float)Math.Sin(h * 0.5) * 0.5f + 0.5f,
                    2 => (float)Math.Sin(w * 0.5) * 0.5f + 0.5f,
                    3 => h < imageHeight / 2 ? 0.8f : 0.2f,
                    4 => w < imageWidth / 2 ? 0.8f : 0.2f,
                    5 => (h + w) % 4 < 2 ? 0.9f : 0.1f,
                    6 => (float)Math.Sqrt(Math.Pow(h - imageHeight/2, 2) + Math.Pow(w - imageWidth/2, 2)) / 20f,
                    7 => random.NextSingle() * 0.3f + 0.35f,
                    8 => h % 8 < 4 ? 0.7f : 0.3f,
                    _ => w % 8 < 4 ? 0.7f : 0.3f,
                };
                image[[c, h, w]] = value + (random.NextSingle() - 0.5f) * 0.15f;
            }
        }
    }
    labeledImages.Add(image);
    labels.Add(label);
}

Console.WriteLine($"  Generated {numLabeledImages} labeled images across {numClasses} classes");

// ============================================
// Step 6: Fine-tune on downstream task
// ============================================
Console.WriteLine("\nStep 6: Fine-tuning classifier on labeled data...");
Console.WriteLine("  (Using frozen encoder features from SimCLR)\n");

// Freeze encoder and add classification head
simclr.FreezeEncoder();
simclr.AddClassificationHead(numClasses);

// Use standard optimizer for fine-tuning
var finetuneOptimizer = new AdamOptimizer<float>(learningRate: 1e-3f);
var classificationLoss = new CrossEntropyLoss<float>();

// Split into train/test
int trainSize = (int)(numLabeledImages * 0.8);
var trainIndices = Enumerable.Range(0, numLabeledImages).OrderBy(_ => random.Next()).Take(trainSize).ToList();
var testIndices = Enumerable.Range(0, numLabeledImages).Except(trainIndices).ToList();

for (int epoch = 0; epoch < finetuneEpochs; epoch++)
{
    float epochLoss = 0;
    int correct = 0;

    // Shuffle training data
    trainIndices = trainIndices.OrderBy(_ => random.Next()).ToList();

    foreach (int idx in trainIndices)
    {
        var image = labeledImages[idx];
        int label = labels[idx];

        // Extract features using frozen encoder
        var features = simclr.ExtractFeatures(image);

        // Forward through classification head
        var logits = simclr.Classify(features);

        // Compute loss
        var labelTensor = new Tensor<float>([numClasses]);
        labelTensor[[label]] = 1.0f;
        float loss = classificationLoss.Compute(logits, labelTensor);
        epochLoss += loss;

        // Check prediction
        int predicted = logits.ArgMax();
        if (predicted == label) correct++;

        // Backward pass (only updates classification head)
        simclr.BackwardClassifier(classificationLoss);
        finetuneOptimizer.Step(simclr.ClassificationHead);
    }

    float avgLoss = epochLoss / trainSize;
    float accuracy = (float)correct / trainSize * 100;

    if ((epoch + 1) % 10 == 0 || epoch == 0)
    {
        Console.WriteLine($"  Epoch {epoch + 1,3}/{finetuneEpochs}: Loss = {avgLoss:F4}, Train Accuracy = {accuracy:F1}%");
    }
}

// ============================================
// Step 7: Evaluate on test set
// ============================================
Console.WriteLine("\nStep 7: Evaluating on test set...");

int testCorrect = 0;
var confusionMatrix = new int[numClasses, numClasses];

foreach (int idx in testIndices)
{
    var image = labeledImages[idx];
    int trueLabel = labels[idx];

    // Extract features and classify
    var features = simclr.ExtractFeatures(image);
    var logits = simclr.Classify(features);
    int predicted = logits.ArgMax();

    confusionMatrix[trueLabel, predicted]++;
    if (predicted == trueLabel) testCorrect++;
}

float testAccuracy = (float)testCorrect / testIndices.Count * 100;

Console.WriteLine($"\n  Test Results:");
Console.WriteLine($"  - Accuracy: {testAccuracy:F1}% ({testCorrect}/{testIndices.Count})");

// Per-class accuracy
Console.WriteLine("\n  Per-class accuracy:");
for (int c = 0; c < numClasses; c++)
{
    int classTotal = 0;
    int classCorrect = confusionMatrix[c, c];
    for (int p = 0; p < numClasses; p++)
    {
        classTotal += confusionMatrix[c, p];
    }
    if (classTotal > 0)
    {
        float classAcc = (float)classCorrect / classTotal * 100;
        Console.WriteLine($"    Class {c}: {classAcc:F1}%");
    }
}

// ============================================
// Step 8: Demonstrate feature quality
// ============================================
Console.WriteLine("\nStep 8: Analyzing learned representations...");

// Extract features for all labeled images
var allFeatures = new List<Tensor<float>>();
foreach (var image in labeledImages)
{
    var features = simclr.ExtractFeatures(image);
    allFeatures.Add(features);
}

// Compute average intra-class and inter-class distances
float avgIntraClassDist = 0;
float avgInterClassDist = 0;
int intraCount = 0;
int interCount = 0;

for (int i = 0; i < numLabeledImages; i++)
{
    for (int j = i + 1; j < numLabeledImages; j++)
    {
        float dist = ComputeCosineSimilarity(allFeatures[i], allFeatures[j]);

        if (labels[i] == labels[j])
        {
            avgIntraClassDist += dist;
            intraCount++;
        }
        else
        {
            avgInterClassDist += dist;
            interCount++;
        }
    }
}

avgIntraClassDist /= intraCount;
avgInterClassDist /= interCount;

Console.WriteLine($"  Feature space analysis:");
Console.WriteLine($"    - Avg intra-class similarity: {avgIntraClassDist:F4} (higher is better)");
Console.WriteLine($"    - Avg inter-class similarity: {avgInterClassDist:F4} (lower is better)");
Console.WriteLine($"    - Separation ratio: {avgIntraClassDist / avgInterClassDist:F2}x");

// ============================================
// Summary
// ============================================
Console.WriteLine("\n===========================================");
Console.WriteLine("  SimCLR Training Complete!");
Console.WriteLine("===========================================");
Console.WriteLine($"\n  Pre-training:");
Console.WriteLine($"    - Unlabeled images: {numUnlabeledImages}");
Console.WriteLine($"    - Epochs: {pretrainEpochs}");
Console.WriteLine($"    - Final contrastive loss: {pretrainLosses.Last():F4}");
Console.WriteLine($"\n  Fine-tuning:");
Console.WriteLine($"    - Labeled images: {numLabeledImages}");
Console.WriteLine($"    - Epochs: {finetuneEpochs}");
Console.WriteLine($"    - Test accuracy: {testAccuracy:F1}%");
Console.WriteLine($"\n  Key insight: SimCLR learned useful representations");
Console.WriteLine($"  from unlabeled data that transfer well to classification!");

// Helper function for cosine similarity
static float ComputeCosineSimilarity(Tensor<float> a, Tensor<float> b)
{
    float dot = 0;
    float normA = 0;
    float normB = 0;

    for (int i = 0; i < a.Shape[0]; i++)
    {
        dot += a[[i]] * b[[i]];
        normA += a[[i]] * a[[i]];
        normB += b[[i]] * b[[i]];
    }

    return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB) + 1e-8f);
}
