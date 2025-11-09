# Junior Developer Implementation Guide: Issue #281
## Evaluation Metrics for Image/Audio/Video

**Issue:** [#281 - Evaluation Metrics for Image/Audio/Video (FID/KID/CLIPScore, WER/CER, PESQ/STOI, FVD)](https://github.com/ooples/AiDotNet/issues/281)

**Estimated Complexity:** Intermediate (Good First Issue)

**Time Estimate:** 15-20 hours

---

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Background Concepts](#background-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Steps](#implementation-steps)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Resources](#resources)

---

## Understanding the Problem

### What Are Evaluation Metrics?

Evaluation metrics are mathematical functions that measure how well a machine learning model performs. They're like grading systems for AI:

- **For Students**: A test score tells you how well you understood the material
- **For AI Models**: Metrics tell you how accurate, realistic, or useful the model's outputs are

### Why Do We Need Specialized Metrics?

Different types of data require different evaluation approaches:

- **Images**: Measure visual quality, realism, and similarity
- **Audio/Speech**: Measure transcription accuracy and audio quality
- **Video**: Measure temporal consistency and motion realism

Just like you wouldn't grade a math test using the same rubric as an essay, we need different metrics for different data types.

---

## Background Concepts

### 1. PSNR (Peak Signal-to-Noise Ratio)

**What it is:** A metric that measures image quality by comparing pixel-level differences.

**How it works:**
1. Calculate the Mean Squared Error (MSE) between two images
2. Use the formula: `PSNR = 20 * log10(MAX_VALUE) - 10 * log10(MSE)`
3. Higher PSNR = Better quality (less noise/distortion)

**Analogy:** Like comparing two photocopies - PSNR measures how many pixels are different and by how much.

**Real-world example:**
```
Perfect copy: PSNR = Infinity (no differences)
High quality JPEG: PSNR = 30-40 dB
Low quality JPEG: PSNR = 20-25 dB
Very poor quality: PSNR < 20 dB
```

**Mathematical Formula:**
```
MSE = (1 / N) * Σ(prediction[i] - target[i])²
PSNR = 20 * log10(maxValue) - 10 * log10(MSE)
```

### 2. SSIM (Structural Similarity Index Measure)

**What it is:** A perceptual metric that measures structural similarity, not just pixel differences.

**Why it's better than PSNR:** SSIM considers luminance, contrast, and structure - how humans actually perceive images.

**How it works:**
1. Calculate mean (luminance) of both images
2. Calculate variance (contrast) of both images
3. Calculate covariance (structural correlation) between images
4. Combine using the SSIM formula

**Analogy:** Like comparing two paintings - SSIM looks at overall composition, lighting, and structure, not just individual brush strokes.

**Real-world example:**
```
Identical images: SSIM = 1.0
Very similar: SSIM > 0.9
Somewhat similar: SSIM = 0.7-0.9
Different: SSIM < 0.5
```

**Mathematical Formula:**
```
SSIM = (2*μx*μy + c1) * (2*σxy + c2) / ((μx² + μy² + c1) * (σx² + σy² + c2))

Where:
- μx, μy = means
- σx², σy² = variances
- σxy = covariance
- c1, c2 = stabilization constants
```

### 3. WER (Word Error Rate)

**What it is:** A metric for evaluating speech-to-text accuracy by counting word-level errors.

**How it works:**
1. Compare predicted text to reference text word-by-word
2. Count substitutions (wrong word), deletions (missing word), and insertions (extra word)
3. Calculate: `WER = (S + D + I) / N`, where N = total words in reference

**Analogy:** Like grading a spelling test - count how many words were misspelled, skipped, or added.

**Real-world example:**
```
Reference:  "the cat sat on the mat"
Prediction: "the cat sat on a mat"

Errors: 1 substitution ("the" → "a")
WER = 1/6 = 0.167 (16.7% error rate)
```

**Implementation Detail - Levenshtein Distance:**

WER uses the Levenshtein distance algorithm (edit distance) to compute the minimum number of operations needed to transform one sequence into another.

**Dynamic Programming Approach:**
```
Create a matrix D where D[i][j] represents the edit distance between
the first i words of reference and first j words of prediction.

D[i][j] = min(
    D[i-1][j] + 1,      // deletion
    D[i][j-1] + 1,      // insertion
    D[i-1][j-1] + cost  // substitution (cost = 0 if words match, 1 otherwise)
)
```

**Example Matrix:**
```
Reference: ["cat", "sat", "mat"]
Prediction: ["cat", "sit", "on", "mat"]

    ""  cat  sit  on  mat
""   0   1    2   3   4
cat  1   0    1   2   3
sat  2   1    1   2   3
mat  3   2    2   2   2

Levenshtein distance = 2
WER = 2/3 = 0.667 (66.7% error rate)
```

### 4. Advanced Metrics (Future Extensions)

These are mentioned in the issue title but not required for the initial implementation:

- **FID (Frechet Inception Distance)**: Measures distribution similarity between real and generated images using deep features
- **KID (Kernel Inception Distance)**: Similar to FID but uses kernel methods
- **CLIPScore**: Uses CLIP embeddings to measure image-text alignment
- **CER (Character Error Rate)**: Like WER but at character level
- **PESQ (Perceptual Evaluation of Speech Quality)**: Measures audio quality
- **STOI (Short-Time Objective Intelligibility)**: Measures speech intelligibility
- **FVD (Frechet Video Distance)**: Like FID but for video sequences

**Note:** Focus on PSNR, SSIM, and WER first. These advanced metrics can be added later as separate issues.

---

## Architecture Overview

### AiDotNet's Standard Pattern

All features in AiDotNet follow this three-layer pattern:

```
Interface (IMetric<T>)
    ↓
Base Class (MetricBase<T>)
    ↓
Concrete Implementations (PSNRMetric<T>, SSIMMetric<T>, WERMetric)
```

**Why this pattern?**
1. **Interface**: Defines the contract (what methods must exist)
2. **Base Class**: Provides common functionality and validation
3. **Concrete Classes**: Implement specific metric logic

### File Organization

```
AiDotNet/
├── src/
│   ├── Interfaces/
│   │   ├── IMetric.cs          // Generic tensor-based metric interface
│   │   └── ITextMetric.cs      // Text-based metric interface
│   ├── Evaluation/
│   │   ├── Metrics/
│   │   │   ├── PSNRMetric.cs   // Image quality metric
│   │   │   ├── SSIMMetric.cs   // Image similarity metric
│   │   │   └── WERMetric.cs    // Speech recognition metric
└── tests/
    └── UnitTests/
        └── Evaluation/
            ├── PSNRMetricTests.cs
            ├── SSIMMetricTests.cs
            └── WERMetricTests.cs
```

### Key Architectural Requirements

#### 1. Use Generic Types with INumericOperations<T>

**NEVER** hardcode specific numeric types like `double` or `float`. Always use generic `T`:

```csharp
// ❌ WRONG - Hardcoded double
public class PSNRMetric
{
    public double Compute(Tensor<double> predictions, Tensor<double> targets) { }
}

// ✅ CORRECT - Generic type with INumericOperations
public class PSNRMetric<T> : IMetric<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public T Compute(Tensor<T> predictions, Tensor<T> targets) { }
}
```

**Why?** This allows the same code to work with `float`, `double`, `decimal`, etc.

#### 2. Never Use `default(T)` or `default!`

```csharp
// ❌ WRONG
T zero = default(T);
T result = default!;

// ✅ CORRECT
T zero = NumOps.Zero;
T one = NumOps.One;
T epsilon = NumOps.FromDouble(1e-8);
```

**Why?** The `default!` operator is not compatible with older .NET Framework versions (net462, net471) and defeats nullable reference type checking.

#### 3. Property Initialization

```csharp
// ❌ WRONG
public string Name { get; set; } = default!;
public List<T> Items { get; set; } = default!;

// ✅ CORRECT
public string Name { get; set; } = string.Empty;
public List<T> Items { get; set; } = new List<T>();
public Vector<T> Values { get; set; } = new Vector<T>(0);
```

#### 4. Use NumOps for All Arithmetic

```csharp
// ❌ WRONG
T sum = a + b;
T product = a * b;
bool isGreater = a > b;

// ✅ CORRECT
T sum = NumOps.Add(a, b);
T product = NumOps.Multiply(a, b);
bool isGreater = NumOps.GreaterThan(a, b);
```

---

## Implementation Steps

### Phase 1: Core Metric Abstractions (3 points)

#### Step 1.1: Create `IMetric<T>` Interface

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IMetric.cs`

```csharp
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for metrics that compare two tensors.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface is used for evaluation metrics that compare predictions against targets,
/// such as image quality metrics (PSNR, SSIM) or other tensor-based comparisons.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a scoring function that compares two sets of data.
///
/// Examples:
/// - Image quality: Compare a compressed image to the original
/// - Model predictions: Compare predicted values to actual values
/// - Similarity: Measure how close two data points are
///
/// The metric returns a single number that represents the quality or similarity.
/// Higher or lower values may be better depending on the specific metric.
/// </para>
/// </remarks>
public interface IMetric<T>
{
    /// <summary>
    /// Computes the metric value by comparing predictions against targets.
    /// </summary>
    /// <param name="predictions">The predicted or generated tensor.</param>
    /// <param name="targets">The ground truth or reference tensor.</param>
    /// <returns>The computed metric value. Interpretation depends on the specific metric.</returns>
    /// <exception cref="ArgumentNullException">Thrown when predictions or targets is null.</exception>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes two tensors and returns a score.
    ///
    /// Think of it like grading:
    /// - predictions = student's answers
    /// - targets = correct answers
    /// - return value = the grade/score
    ///
    /// The tensors must have the same shape (same dimensions and size).
    /// </para>
    /// </remarks>
    T Compute(Tensor<T> predictions, Tensor<T> targets);
}
```

**Key Points:**
- Lives in `src/Interfaces/` (root level, NOT in subdirectories)
- Generic type parameter `T` for numeric type flexibility
- Single method: `Compute` that takes two tensors
- Returns type `T` (the computed metric value)
- Comprehensive XML documentation with beginner-friendly remarks

#### Step 1.2: Create `ITextMetric` Interface

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\ITextMetric.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for metrics that compare two text strings.
/// </summary>
/// <remarks>
/// <para>
/// This interface is used for text-based evaluation metrics such as Word Error Rate (WER),
/// Character Error Rate (CER), BLEU scores, or other string comparison metrics.
/// </para>
/// <para><b>For Beginners:</b> This is for scoring text comparisons.
///
/// Common use cases:
/// - Speech-to-text: Compare transcribed text to correct transcription
/// - Translation: Compare machine translation to human translation
/// - Text generation: Compare generated text to reference text
///
/// Unlike tensor metrics, this works directly with strings (text).
/// The result is typically a score between 0 and 1, or a percentage.
/// </para>
/// </remarks>
public interface ITextMetric
{
    /// <summary>
    /// Computes the metric value by comparing a prediction string against a target string.
    /// </summary>
    /// <param name="prediction">The predicted or generated text.</param>
    /// <param name="target">The ground truth or reference text.</param>
    /// <returns>
    /// The computed metric value. Lower values typically indicate better performance for error rates.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when prediction or target is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method compares two strings and returns a score.
    ///
    /// Example for Word Error Rate:
    /// - prediction = "the cat sat on a mat"
    /// - target = "the cat sat on the mat"
    /// - return value = 0.167 (16.7% error - 1 word wrong out of 6)
    ///
    /// For error rate metrics, lower is better (0 = perfect match).
    /// For similarity metrics, higher is better (1 = perfect match).
    /// </para>
    /// </remarks>
    double Compute(string prediction, string target);
}
```

**Key Points:**
- Separate interface for text-based metrics (strings, not tensors)
- Returns `double` (not generic `T`) since text metrics are typically percentages or ratios
- Used for speech recognition, translation, and text generation evaluation

**Why two interfaces?**
- `IMetric<T>`: For numerical data (images, audio signals, sensor data)
- `ITextMetric`: For text data (transcriptions, translations, generated text)

### Phase 2: Metric Implementations (16 points total)

#### Step 2.1: Implement `PSNRMetric<T>` (3 points)

**What is PSNR?**
Peak Signal-to-Noise Ratio measures image quality by calculating the ratio between the maximum possible pixel value and the mean squared error.

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Evaluation\Metrics\PSNRMetric.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Evaluation.Metrics;

/// <summary>
/// Implements Peak Signal-to-Noise Ratio (PSNR) metric for image quality assessment.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// PSNR is a widely-used metric for measuring image quality, particularly for comparing
/// compressed or reconstructed images to their originals. It's based on the Mean Squared
/// Error (MSE) between pixel values.
/// </para>
/// <para><b>For Beginners:</b> PSNR measures how different two images are at the pixel level.
///
/// Think of it like measuring photocopy quality:
/// - Perfect copy (identical pixels): PSNR = Infinity
/// - High quality (barely visible differences): PSNR = 30-40 dB
/// - Medium quality (noticeable differences): PSNR = 20-30 dB
/// - Poor quality (obvious distortion): PSNR < 20 dB
///
/// Higher PSNR = Better quality (less distortion)
///
/// <b>When to use PSNR:</b>
/// - Comparing image compression quality (JPEG, WebP, etc.)
/// - Evaluating image reconstruction algorithms
/// - Measuring noise reduction effectiveness
///
/// <b>Limitations:</b>
/// - Doesn't match human perception well (a blurry image might have good PSNR)
/// - Consider using SSIM for perceptual quality
/// </para>
/// <para>
/// <b>Formula:</b> PSNR = 20 * log10(maxValue) - 10 * log10(MSE)
///
/// Where MSE = (1/N) * Σ(prediction[i] - target[i])²
/// </para>
/// <para>
/// <b>Default maxValue:</b> 255.0 (standard for 8-bit images)
///
/// This default comes from the standard RGB color representation where each channel
/// has values from 0-255. If your images use a different range (e.g., 0-1 for normalized
/// images), you should set maxValue accordingly.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // For standard 8-bit images (0-255 range)
/// var psnr = new PSNRMetric&lt;double&gt;();
/// double score = psnr.Compute(compressedImage, originalImage);
/// Console.WriteLine($"PSNR: {score} dB");
///
/// // For normalized images (0-1 range)
/// var psnrNormalized = new PSNRMetric&lt;double&gt;(maxValue: 1.0);
/// double scoreNorm = psnrNormalized.Compute(normalizedCompressed, normalizedOriginal);
/// </code>
/// </example>
public class PSNRMetric<T> : IMetric<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The maximum possible pixel value in the images.
    /// </summary>
    private readonly T _maxValue;

    /// <summary>
    /// Initializes a new instance of the PSNRMetric class.
    /// </summary>
    /// <param name="maxValue">
    /// The maximum possible pixel value. Default is 255.0 for 8-bit images.
    /// Use 1.0 for normalized images, or other values for HDR images.
    /// </param>
    /// <exception cref="ArgumentException">Thrown when maxValue is not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> The maxValue parameter defines the scale of your pixel values.
    ///
    /// Common values:
    /// - 255.0: Standard 8-bit RGB images (most common)
    /// - 1.0: Normalized/floating-point images (0.0 to 1.0 range)
    /// - 65535.0: 16-bit images (HDR photography)
    ///
    /// Example: If your image pixels range from 0 to 255, use the default (255.0).
    /// If your image pixels range from 0.0 to 1.0, pass maxValue: 1.0.
    /// </para>
    /// </remarks>
    public PSNRMetric(double maxValue = 255.0)
    {
        if (maxValue <= 0)
        {
            throw new ArgumentException("maxValue must be positive", nameof(maxValue));
        }

        _maxValue = NumOps.FromDouble(maxValue);
    }

    /// <summary>
    /// Computes the PSNR between two images.
    /// </summary>
    /// <param name="predictions">The predicted or compressed image tensor.</param>
    /// <param name="targets">The ground truth or original image tensor.</param>
    /// <returns>
    /// The PSNR value in decibels (dB). Returns positive infinity if images are identical.
    /// Higher values indicate better quality (less distortion).
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when predictions or targets is null.</exception>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how similar two images are.
    ///
    /// Steps:
    /// 1. Check that both images have the same size
    /// 2. Calculate Mean Squared Error (MSE) - average of squared pixel differences
    /// 3. Apply the PSNR formula: 20*log10(maxValue) - 10*log10(MSE)
    /// 4. Return the result in decibels (dB)
    ///
    /// Special case: If images are identical (MSE = 0), returns Infinity.
    ///
    /// Interpretation:
    /// - PSNR > 40 dB: Excellent quality, differences barely visible
    /// - 30-40 dB: Good quality, minor artifacts
    /// - 20-30 dB: Acceptable quality, visible compression
    /// - < 20 dB: Poor quality, significant distortion
    /// </para>
    /// </remarks>
    public T Compute(Tensor<T> predictions, Tensor<T> targets)
    {
        // Validate inputs
        if (predictions == null)
        {
            throw new ArgumentNullException(nameof(predictions));
        }

        if (targets == null)
        {
            throw new ArgumentNullException(nameof(targets));
        }

        if (!predictions.Shape.SequenceEqual(targets.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Predictions: [{string.Join(", ", predictions.Shape)}], " +
                $"Targets: [{string.Join(", ", targets.Shape)}]");
        }

        // Calculate Mean Squared Error (MSE)
        T mse = CalculateMSE(predictions, targets);

        // If MSE is zero, images are identical -> PSNR = infinity
        if (NumOps.Equals(mse, NumOps.Zero))
        {
            return NumOps.FromDouble(double.PositiveInfinity);
        }

        // PSNR = 20 * log10(maxValue) - 10 * log10(MSE)
        double maxValueDouble = NumOps.ToDouble(_maxValue);
        double mseDouble = NumOps.ToDouble(mse);

        double psnr = 20.0 * Math.Log10(maxValueDouble) - 10.0 * Math.Log10(mseDouble);

        return NumOps.FromDouble(psnr);
    }

    /// <summary>
    /// Calculates the Mean Squared Error between two tensors.
    /// </summary>
    /// <param name="predictions">The predicted tensor.</param>
    /// <param name="targets">The target tensor.</param>
    /// <returns>The MSE value.</returns>
    private T CalculateMSE(Tensor<T> predictions, Tensor<T> targets)
    {
        // Flatten tensors to 1D for easier processing
        var predFlat = predictions.Flatten();
        var targFlat = targets.Flatten();

        int length = predFlat.Length;
        T sumSquaredErrors = NumOps.Zero;

        // Calculate sum of squared differences
        for (int i = 0; i < length; i++)
        {
            T diff = NumOps.Subtract(predFlat[i], targFlat[i]);
            T squaredDiff = NumOps.Multiply(diff, diff);
            sumSquaredErrors = NumOps.Add(sumSquaredErrors, squaredDiff);
        }

        // Calculate mean
        T lengthT = NumOps.FromInt32(length);
        T mse = NumOps.Divide(sumSquaredErrors, lengthT);

        return mse;
    }
}
```

**Key Implementation Details:**

1. **Constructor Parameter with Default:**
   - `maxValue = 255.0` (industry standard for 8-bit images)
   - Validates that maxValue is positive
   - Converts to generic type `T` using `NumOps.FromDouble()`

2. **MSE Calculation:**
   - Flatten tensors to simplify iteration
   - Use `NumOps.Subtract()` for differences
   - Use `NumOps.Multiply()` for squaring
   - Use `NumOps.Divide()` for averaging

3. **Special Case Handling:**
   - If MSE = 0 (identical images), return infinity
   - Use `NumOps.Equals()` to check for zero

4. **PSNR Formula:**
   - Convert to double for logarithm calculation
   - Apply: `20 * log10(maxValue) - 10 * log10(MSE)`
   - Convert result back to type `T`

#### Step 2.2: Implement `SSIMMetric<T>` (5 points)

**What is SSIM?**
Structural Similarity Index Measure is a perceptual metric that considers luminance, contrast, and structure - closer to how humans judge image quality than PSNR.

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Evaluation\Metrics\SSIMMetric.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Evaluation.Metrics;

/// <summary>
/// Implements Structural Similarity Index Measure (SSIM) for perceptual image quality assessment.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// SSIM is a perceptual metric designed to match human visual perception better than
/// simple pixel-wise metrics like MSE or PSNR. It measures similarity based on three
/// components: luminance, contrast, and structure.
/// </para>
/// <para><b>For Beginners:</b> SSIM measures image similarity the way humans see it.
///
/// Unlike PSNR which just compares pixels:
/// - SSIM considers overall brightness (luminance)
/// - SSIM considers contrast differences
/// - SSIM considers structural patterns
///
/// This makes SSIM better at matching human perception:
/// - A slightly blurry image might have good PSNR but poor SSIM
/// - A brightness-adjusted image might have poor PSNR but good SSIM
///
/// <b>SSIM Scale:</b>
/// - 1.0 = Identical images (perfect similarity)
/// - > 0.9 = Excellent similarity (barely noticeable differences)
/// - 0.7-0.9 = Good similarity (minor differences)
/// - 0.5-0.7 = Fair similarity (noticeable differences)
/// - < 0.5 = Poor similarity (significantly different)
///
/// <b>When to use SSIM:</b>
/// - Evaluating perceptual image quality
/// - Comparing image processing algorithms (denoising, super-resolution)
/// - Measuring visual similarity for human viewers
///
/// <b>Advantages over PSNR:</b>
/// - Better correlation with human perception
/// - More robust to luminance/contrast changes
/// - Considers structural information
/// </para>
/// <para>
/// <b>Formula:</b>
/// SSIM = [(2*μx*μy + c1) * (2*σxy + c2)] / [(μx² + μy² + c1) * (σx² + σy² + c2)]
///
/// Where:
/// - μx, μy = mean values (luminance)
/// - σx², σy² = variances (contrast)
/// - σxy = covariance (structure)
/// - c1, c2 = stabilization constants
/// </para>
/// <para>
/// <b>Implementation Notes:</b>
/// This is a simplified global SSIM implementation that computes statistics over the
/// entire image. For production use, consider implementing local SSIM with sliding
/// windows (typically 11x11) as described in the original paper.
///
/// Reference: Wang et al., "Image Quality Assessment: From Error Visibility to
/// Structural Similarity," IEEE Transactions on Image Processing, 2004.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Compare two images
/// var ssim = new SSIMMetric&lt;double&gt;();
/// double similarity = ssim.Compute(processedImage, originalImage);
///
/// if (similarity > 0.95)
///     Console.WriteLine("Excellent quality - differences barely visible");
/// else if (similarity > 0.80)
///     Console.WriteLine("Good quality - minor differences");
/// else
///     Console.WriteLine("Fair quality - noticeable differences");
///
/// // For normalized images (0-1 range), adjust L parameter
/// var ssimNorm = new SSIMMetric&lt;double&gt;(L: 1.0);
/// double similarityNorm = ssimNorm.Compute(normalizedImage1, normalizedImage2);
/// </code>
/// </example>
public class SSIMMetric<T> : IMetric<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Stabilization constant c1 for luminance comparison.
    /// </summary>
    private readonly T _c1;

    /// <summary>
    /// Stabilization constant c2 for contrast/structure comparison.
    /// </summary>
    private readonly T _c2;

    /// <summary>
    /// Initializes a new instance of the SSIMMetric class.
    /// </summary>
    /// <param name="L">
    /// The dynamic range of pixel values. Default is 255.0 for 8-bit images.
    /// Use 1.0 for normalized images.
    /// </param>
    /// <param name="k1">
    /// Constant for c1 calculation. Default is 0.01 (from SSIM paper).
    /// </param>
    /// <param name="k2">
    /// Constant for c2 calculation. Default is 0.03 (from SSIM paper).
    /// </param>
    /// <exception cref="ArgumentException">Thrown when L, k1, or k2 are not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> The constants c1 and c2 prevent division by zero.
    ///
    /// - L: The range of your pixel values (255 for standard images, 1.0 for normalized)
    /// - k1, k2: Small constants that stabilize the calculation
    ///
    /// The defaults (k1=0.01, k2=0.03) come from the original SSIM paper and work
    /// well for most images. You typically don't need to change them.
    ///
    /// Calculation:
    /// - c1 = (k1 * L)² = (0.01 * 255)² = 6.5025 (for 8-bit images)
    /// - c2 = (k2 * L)² = (0.03 * 255)² = 58.5225 (for 8-bit images)
    ///
    /// These constants are added to numerator and denominator to avoid dividing by zero
    /// when comparing very dark or uniform regions.
    /// </para>
    /// </remarks>
    public SSIMMetric(double L = 255.0, double k1 = 0.01, double k2 = 0.03)
    {
        if (L <= 0)
        {
            throw new ArgumentException("L must be positive", nameof(L));
        }

        if (k1 <= 0)
        {
            throw new ArgumentException("k1 must be positive", nameof(k1));
        }

        if (k2 <= 0)
        {
            throw new ArgumentException("k2 must be positive", nameof(k2));
        }

        // Calculate stabilization constants
        // c1 = (k1 * L)²
        // c2 = (k2 * L)²
        double c1 = (k1 * L) * (k1 * L);
        double c2 = (k2 * L) * (k2 * L);

        _c1 = NumOps.FromDouble(c1);
        _c2 = NumOps.FromDouble(c2);
    }

    /// <summary>
    /// Computes the SSIM between two images.
    /// </summary>
    /// <param name="predictions">The first image tensor.</param>
    /// <param name="targets">The second image tensor (reference).</param>
    /// <returns>
    /// The SSIM value between 0 and 1. Higher values indicate greater similarity.
    /// A value of 1.0 indicates identical images.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when predictions or targets is null.</exception>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates perceptual similarity.
    ///
    /// Steps:
    /// 1. Validate that both images have the same size
    /// 2. Calculate mean (average brightness) of each image
    /// 3. Calculate variance (contrast spread) of each image
    /// 4. Calculate covariance (how the images vary together)
    /// 5. Combine using the SSIM formula
    /// 6. Return value between 0 (completely different) and 1 (identical)
    ///
    /// The formula considers three aspects:
    /// - Luminance: Are the images similarly bright?
    /// - Contrast: Do they have similar contrast levels?
    /// - Structure: Do they have similar patterns and textures?
    ///
    /// This global version computes one SSIM value for the entire image.
    /// For better results, the original paper recommends computing SSIM
    /// on small windows and averaging (local SSIM).
    /// </para>
    /// </remarks>
    public T Compute(Tensor<T> predictions, Tensor<T> targets)
    {
        // Validate inputs
        if (predictions == null)
        {
            throw new ArgumentNullException(nameof(predictions));
        }

        if (targets == null)
        {
            throw new ArgumentNullException(nameof(targets));
        }

        if (!predictions.Shape.SequenceEqual(targets.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Predictions: [{string.Join(", ", predictions.Shape)}], " +
                $"Targets: [{string.Join(", ", targets.Shape)}]");
        }

        // Flatten tensors for easier processing
        var x = predictions.Flatten();
        var y = targets.Flatten();

        // Calculate statistics
        T meanX = CalculateMean(x);
        T meanY = CalculateMean(y);
        T varX = CalculateVariance(x, meanX);
        T varY = CalculateVariance(y, meanY);
        T covXY = CalculateCovariance(x, y, meanX, meanY);

        // SSIM formula:
        // SSIM = [(2*μx*μy + c1) * (2*σxy + c2)] / [(μx² + μy² + c1) * (σx² + σy² + c2)]

        // Numerator: (2*μx*μy + c1) * (2*σxy + c2)
        T twoMeanProduct = NumOps.Multiply(
            NumOps.Multiply(NumOps.FromDouble(2.0), meanX),
            meanY);
        T luminance = NumOps.Add(twoMeanProduct, _c1);

        T twoCov = NumOps.Multiply(NumOps.FromDouble(2.0), covXY);
        T structure = NumOps.Add(twoCov, _c2);

        T numerator = NumOps.Multiply(luminance, structure);

        // Denominator: (μx² + μy² + c1) * (σx² + σy² + c2)
        T meanXSquared = NumOps.Multiply(meanX, meanX);
        T meanYSquared = NumOps.Multiply(meanY, meanY);
        T meanSum = NumOps.Add(meanXSquared, meanYSquared);
        T denomLuminance = NumOps.Add(meanSum, _c1);

        T varSum = NumOps.Add(varX, varY);
        T denomContrast = NumOps.Add(varSum, _c2);

        T denominator = NumOps.Multiply(denomLuminance, denomContrast);

        // Final SSIM
        T ssim = NumOps.Divide(numerator, denominator);

        return ssim;
    }

    /// <summary>
    /// Calculates the mean of a vector.
    /// </summary>
    private T CalculateMean(Vector<T> vector)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sum = NumOps.Add(sum, vector[i]);
        }

        T length = NumOps.FromInt32(vector.Length);
        return NumOps.Divide(sum, length);
    }

    /// <summary>
    /// Calculates the variance of a vector given its mean.
    /// </summary>
    private T CalculateVariance(Vector<T> vector, T mean)
    {
        T sumSquaredDiffs = NumOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            T diff = NumOps.Subtract(vector[i], mean);
            T squaredDiff = NumOps.Multiply(diff, diff);
            sumSquaredDiffs = NumOps.Add(sumSquaredDiffs, squaredDiff);
        }

        T length = NumOps.FromInt32(vector.Length);
        return NumOps.Divide(sumSquaredDiffs, length);
    }

    /// <summary>
    /// Calculates the covariance between two vectors given their means.
    /// </summary>
    private T CalculateCovariance(Vector<T> x, Vector<T> y, T meanX, T meanY)
    {
        T sumProducts = NumOps.Zero;

        for (int i = 0; i < x.Length; i++)
        {
            T diffX = NumOps.Subtract(x[i], meanX);
            T diffY = NumOps.Subtract(y[i], meanY);
            T product = NumOps.Multiply(diffX, diffY);
            sumProducts = NumOps.Add(sumProducts, product);
        }

        T length = NumOps.FromInt32(x.Length);
        return NumOps.Divide(sumProducts, length);
    }
}
```

**Key Implementation Details:**

1. **Constructor with Research-Based Defaults:**
   - `L = 255.0`: Dynamic range for 8-bit images
   - `k1 = 0.01`, `k2 = 0.03`: From original SSIM paper (Wang et al., 2004)
   - Calculates c1 and c2 stabilization constants

2. **Three Statistical Components:**
   - Mean (luminance): Average pixel value
   - Variance (contrast): Spread of pixel values
   - Covariance (structure): How images vary together

3. **SSIM Formula Implementation:**
   - Numerator: Combines luminance and structure terms
   - Denominator: Normalizes by variance terms
   - Result: Value between 0 and 1

4. **Helper Methods:**
   - `CalculateMean()`: Computes average
   - `CalculateVariance()`: Computes spread around mean
   - `CalculateCovariance()`: Computes joint variation

#### Step 2.3: Implement `WERMetric` (8 points)

**What is WER?**
Word Error Rate measures speech-to-text accuracy by counting word-level substitutions, deletions, and insertions using the Levenshtein distance algorithm.

**Location:** `C:\Users\cheat\source\repos\AiDotNet\src\Evaluation\Metrics\WERMetric.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Metrics;

/// <summary>
/// Implements Word Error Rate (WER) for speech recognition and text comparison evaluation.
/// </summary>
/// <remarks>
/// <para>
/// WER is the standard metric for evaluating automatic speech recognition (ASR) systems.
/// It calculates the percentage of words that were incorrectly transcribed by counting
/// substitutions (wrong word), deletions (missing word), and insertions (extra word).
/// </para>
/// <para><b>For Beginners:</b> WER measures how many mistakes a speech-to-text system makes.
///
/// Think of it like grading a transcription:
/// - Reference (correct): "the cat sat on the mat"
/// - Hypothesis (ASR output): "the cat sit on a mat"
///
/// Errors:
/// - "sat" → "sit" = 1 substitution (wrong word)
/// - "the" → "a" = 1 substitution (wrong word)
/// - Total: 2 errors out of 6 words = WER of 33.3%
///
/// <b>WER Scale:</b>
/// - 0% = Perfect transcription (no errors)
/// - < 5% = Excellent (professional-grade ASR)
/// - 5-10% = Good (usable for most applications)
/// - 10-20% = Fair (needs improvement)
/// - > 20% = Poor (significant errors)
///
/// <b>When to use WER:</b>
/// - Evaluating speech recognition systems
/// - Comparing ASR model performance
/// - Measuring transcription quality
/// - Testing voice assistant accuracy
///
/// <b>Important Notes:</b>
/// - Lower WER is better (0 = perfect)
/// - WER can exceed 100% if many insertions occur
/// - Case-sensitive by default (can normalize if needed)
/// - Punctuation affects the score
/// </para>
/// <para>
/// <b>Formula:</b>
/// WER = (Substitutions + Deletions + Insertions) / Total Words in Reference
///
/// Where:
/// - Substitutions (S): Words that were replaced with incorrect words
/// - Deletions (D): Words that were omitted
/// - Insertions (I): Extra words that were added
/// - Total Words: Number of words in the reference (correct) transcription
/// </para>
/// <para>
/// <b>Implementation:</b>
/// Uses the Levenshtein distance algorithm (dynamic programming) to find the minimum
/// edit distance between word sequences. The algorithm considers all three error types
/// and finds the optimal alignment.
///
/// Time Complexity: O(m * n) where m and n are the number of words
/// Space Complexity: O(m * n) for the dynamic programming matrix
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var wer = new WERMetric();
///
/// // Example 1: One substitution
/// string reference = "the cat sat on the mat";
/// string hypothesis = "the cat sat on a mat";
/// double error = wer.Compute(hypothesis, reference);
/// // Result: 0.167 (16.7% - 1 error out of 6 words)
///
/// // Example 2: One deletion
/// reference = "the cat sat on the mat";
/// hypothesis = "the cat on the mat";
/// error = wer.Compute(hypothesis, reference);
/// // Result: 0.167 (16.7% - 1 deletion out of 6 words)
///
/// // Example 3: One insertion
/// reference = "the cat sat on the mat";
/// hypothesis = "the black cat sat on the mat";
/// error = wer.Compute(hypothesis, reference);
/// // Result: 0.167 (16.7% - 1 insertion out of 6 words)
///
/// // Example 4: Perfect match
/// reference = "hello world";
/// hypothesis = "hello world";
/// error = wer.Compute(hypothesis, reference);
/// // Result: 0.0 (0% - no errors)
/// </code>
/// </example>
public class WERMetric : ITextMetric
{
    /// <summary>
    /// Computes the Word Error Rate between a predicted transcription and reference text.
    /// </summary>
    /// <param name="prediction">The predicted or ASR-generated transcription.</param>
    /// <param name="target">The ground truth or reference transcription.</param>
    /// <returns>
    /// The WER as a decimal value (0.0 to 1.0+). Lower is better.
    /// Returns 0.0 for identical strings. Can exceed 1.0 if many insertions occur.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when prediction or target is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method compares two sentences word by word.
    ///
    /// Steps:
    /// 1. Split both strings into arrays of words (using spaces)
    /// 2. Use Levenshtein distance to find minimum edits needed
    /// 3. Divide edit distance by number of words in reference
    /// 4. Return the error rate (0 = perfect, 1 = 100% error)
    ///
    /// Special Cases:
    /// - Empty reference: Returns 0 if both empty, otherwise number of words in prediction
    /// - Identical strings: Returns 0 (no errors)
    /// - More errors than words: WER can exceed 100% (e.g., many insertions)
    ///
    /// The algorithm finds the minimum number of operations (substitute, delete, insert)
    /// needed to transform the prediction into the reference. This is optimal - there's
    /// no way to fix the transcription with fewer changes.
    /// </para>
    /// </remarks>
    public double Compute(string prediction, string target)
    {
        if (prediction == null)
        {
            throw new ArgumentNullException(nameof(prediction));
        }

        if (target == null)
        {
            throw new ArgumentNullException(nameof(target));
        }

        // Split into word arrays
        string[] predictionWords = SplitIntoWords(prediction);
        string[] targetWords = SplitIntoWords(target);

        // Handle edge case: empty target
        if (targetWords.Length == 0)
        {
            return predictionWords.Length == 0 ? 0.0 : predictionWords.Length;
        }

        // Calculate Levenshtein distance (edit distance)
        int editDistance = CalculateLevenshteinDistance(predictionWords, targetWords);

        // WER = edit distance / number of words in reference
        double wer = (double)editDistance / targetWords.Length;

        return wer;
    }

    /// <summary>
    /// Splits a string into an array of words.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>Array of words, with empty entries removed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This breaks a sentence into individual words.
    ///
    /// Example: "the cat sat" → ["the", "cat", "sat"]
    ///
    /// The method:
    /// 1. Splits on whitespace (spaces, tabs, newlines)
    /// 2. Removes empty entries (multiple spaces become one split)
    /// 3. Returns a clean array of words
    ///
    /// Note: This simple implementation treats punctuation as part of words.
    /// For production use, consider normalizing:
    /// - Removing punctuation
    /// - Converting to lowercase
    /// - Handling contractions
    /// </para>
    /// </remarks>
    private string[] SplitIntoWords(string text)
    {
        // Split on whitespace and remove empty entries
        return text.Split(
            new[] { ' ', '\t', '\n', '\r' },
            StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>
    /// Calculates the Levenshtein distance (edit distance) between two word sequences.
    /// </summary>
    /// <param name="source">The source word sequence (prediction).</param>
    /// <param name="target">The target word sequence (reference).</param>
    /// <returns>The minimum number of edits (substitutions + deletions + insertions) needed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds the minimum changes needed to transform one sentence into another.
    ///
    /// The algorithm uses dynamic programming - a technique for solving complex problems
    /// by breaking them into simpler subproblems.
    ///
    /// Think of it like a grid/table:
    /// - Rows represent words in the reference (target)
    /// - Columns represent words in the prediction (source)
    /// - Each cell [i,j] contains the edit distance between first i reference words and first j prediction words
    ///
    /// Example:
    /// Reference: ["cat", "sat", "mat"]
    /// Prediction: ["cat", "sit", "on", "mat"]
    ///
    /// Build this matrix:
    ///       ""  cat  sit  on  mat
    ///   ""   0   1    2   3   4
    ///  cat   1   0    1   2   3
    ///  sat   2   1    1   2   3
    ///  mat   3   2    2   2   2
    ///
    /// Bottom-right cell (2) is the answer: 2 edits needed
    ///
    /// At each cell, choose the minimum of:
    /// - Cell above + 1 (deletion - remove word from reference)
    /// - Cell to left + 1 (insertion - add word from prediction)
    /// - Cell diagonally above-left + cost (substitution - 0 if words match, 1 if different)
    /// </para>
    /// <para>
    /// <b>Algorithm Steps:</b>
    /// 1. Create a (target.Length + 1) × (source.Length + 1) matrix
    /// 2. Initialize first row: [0, 1, 2, 3, ...] (cost of insertions)
    /// 3. Initialize first column: [0, 1, 2, 3, ...] (cost of deletions)
    /// 4. Fill each cell using the recurrence relation:
    ///    dp[i,j] = min(
    ///      dp[i-1,j] + 1,           // deletion
    ///      dp[i,j-1] + 1,           // insertion
    ///      dp[i-1,j-1] + cost       // substitution (cost = 0 if match, 1 if different)
    ///    )
    /// 5. Return dp[target.Length, source.Length]
    /// </para>
    /// </remarks>
    private int CalculateLevenshteinDistance(string[] source, string[] target)
    {
        int sourceLen = source.Length;
        int targetLen = target.Length;

        // Create matrix for dynamic programming
        // dp[i, j] = edit distance between first i target words and first j source words
        int[,] dp = new int[targetLen + 1, sourceLen + 1];

        // Initialize first row (cost of insertions to get from empty to source)
        for (int j = 0; j <= sourceLen; j++)
        {
            dp[0, j] = j;
        }

        // Initialize first column (cost of deletions to get from target to empty)
        for (int i = 0; i <= targetLen; i++)
        {
            dp[i, 0] = i;
        }

        // Fill the matrix using dynamic programming
        for (int i = 1; i <= targetLen; i++)
        {
            for (int j = 1; j <= sourceLen; j++)
            {
                // Check if current words match
                bool wordsMatch = string.Equals(
                    target[i - 1],
                    source[j - 1],
                    StringComparison.Ordinal);

                int substitutionCost = wordsMatch ? 0 : 1;

                // Calculate minimum of three operations
                int deletion = dp[i - 1, j] + 1;        // Delete from target
                int insertion = dp[i, j - 1] + 1;       // Insert from source
                int substitution = dp[i - 1, j - 1] + substitutionCost;  // Match or substitute

                dp[i, j] = Math.Min(Math.Min(deletion, insertion), substitution);
            }
        }

        // Bottom-right cell contains the final edit distance
        return dp[targetLen, sourceLen];
    }
}
```

**Key Implementation Details:**

1. **Word Splitting:**
   - Splits on whitespace (spaces, tabs, newlines)
   - Removes empty entries (handles multiple spaces)
   - Simple implementation (consider normalization for production)

2. **Levenshtein Distance Algorithm:**
   - **Dynamic Programming approach** (O(m×n) time and space)
   - Creates a matrix where `dp[i,j]` = edit distance between first `i` target words and first `j` source words
   - Three possible operations at each step:
     - Deletion: `dp[i-1,j] + 1`
     - Insertion: `dp[i,j-1] + 1`
     - Substitution: `dp[i-1,j-1] + cost` (cost=0 if match, cost=1 if different)

3. **Matrix Initialization:**
   - First row: Cost of inserting all source words (0, 1, 2, 3, ...)
   - First column: Cost of deleting all target words (0, 1, 2, 3, ...)

4. **Final Result:**
   - Bottom-right cell contains the minimum edit distance
   - WER = edit distance / number of reference words

### Phase 3: Validation and Testing (5 points)

#### Step 3.1: Create PSNRMetric Tests

**Location:** `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Evaluation\PSNRMetricTests.cs`

```csharp
using AiDotNet.Evaluation.Metrics;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

/// <summary>
/// Unit tests for PSNRMetric.
/// </summary>
public class PSNRMetricTests
{
    [Fact]
    public void Compute_IdenticalTensors_ReturnsInfinity()
    {
        // Arrange
        var metric = new PSNRMetric<double>();
        var tensor = new Tensor<double>(new[] { 2, 2 }, new double[] { 100, 150, 200, 250 });

        // Act
        double result = metric.Compute(tensor, tensor);

        // Assert
        Assert.Equal(double.PositiveInfinity, result);
    }

    [Fact]
    public void Compute_BlackAndWhiteTensors_ReturnsZero()
    {
        // Arrange: All black (0) vs all white (255)
        var metric = new PSNRMetric<double>(maxValue: 255.0);
        var blackTensor = new Tensor<double>(new[] { 2, 2 }, new double[] { 0, 0, 0, 0 });
        var whiteTensor = new Tensor<double>(new[] { 2, 2 }, new double[] { 255, 255, 255, 255 });

        // Act
        double result = metric.Compute(whiteTensor, blackTensor);

        // Assert
        // PSNR = 20*log10(255) - 10*log10(MSE)
        // MSE = 255²  = 65025
        // PSNR = 20*log10(255) - 10*log10(65025)
        // PSNR = 20*2.407 - 10*4.813 = 48.14 - 48.13 ≈ 0
        Assert.True(result < 0.1, $"Expected PSNR near 0 for black vs white, got {result}");
    }

    [Fact]
    public void Compute_SlightlyDifferentTensors_ReturnsHighPSNR()
    {
        // Arrange: Very similar tensors (1 pixel different by 1 value)
        var metric = new PSNRMetric<double>(maxValue: 255.0);
        var original = new Tensor<double>(new[] { 2, 2 }, new double[] { 100, 100, 100, 100 });
        var modified = new Tensor<double>(new[] { 2, 2 }, new double[] { 100, 100, 100, 101 });

        // Act
        double result = metric.Compute(modified, original);

        // Assert
        // MSE = (0² + 0² + 0² + 1²) / 4 = 0.25
        // PSNR = 20*log10(255) - 10*log10(0.25)
        // PSNR = 48.14 - 10*(-0.602) = 48.14 + 6.02 = 54.16
        Assert.True(result > 50, $"Expected PSNR > 50 dB for minimal difference, got {result}");
    }

    [Fact]
    public void Compute_DifferentShapes_ThrowsArgumentException()
    {
        // Arrange
        var metric = new PSNRMetric<double>();
        var tensor1 = new Tensor<double>(new[] { 2, 2 });
        var tensor2 = new Tensor<double>(new[] { 3, 3 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => metric.Compute(tensor1, tensor2));
    }

    [Fact]
    public void Compute_NullPredictions_ThrowsArgumentNullException()
    {
        // Arrange
        var metric = new PSNRMetric<double>();
        var target = new Tensor<double>(new[] { 2, 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => metric.Compute(null!, target));
    }

    [Fact]
    public void Compute_NullTargets_ThrowsArgumentNullException()
    {
        // Arrange
        var metric = new PSNRMetric<double>();
        var prediction = new Tensor<double>(new[] { 2, 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => metric.Compute(prediction, null!));
    }

    [Fact]
    public void Constructor_NegativeMaxValue_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new PSNRMetric<double>(maxValue: -1.0));
    }

    [Fact]
    public void Constructor_ZeroMaxValue_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new PSNRMetric<double>(maxValue: 0.0));
    }

    [Fact]
    public void Compute_NormalizedImages_WorksWithMaxValue1()
    {
        // Arrange: Normalized images (0.0 to 1.0 range)
        var metric = new PSNRMetric<double>(maxValue: 1.0);
        var original = new Tensor<double>(new[] { 2, 2 }, new double[] { 0.5, 0.5, 0.5, 0.5 });
        var modified = new Tensor<double>(new[] { 2, 2 }, new double[] { 0.5, 0.5, 0.5, 0.6 });

        // Act
        double result = metric.Compute(modified, original);

        // Assert
        // MSE = (0² + 0² + 0² + 0.1²) / 4 = 0.0025
        // PSNR = 20*log10(1.0) - 10*log10(0.0025)
        // PSNR = 0 - 10*(-2.602) = 26.02
        Assert.True(result > 20 && result < 30, $"Expected PSNR between 20-30 dB, got {result}");
    }

    [Fact]
    public void Compute_FloatType_WorksCorrectly()
    {
        // Arrange
        var metric = new PSNRMetric<float>(maxValue: 255.0f);
        var tensor1 = new Tensor<float>(new[] { 2, 2 }, new float[] { 100f, 150f, 200f, 250f });
        var tensor2 = new Tensor<float>(new[] { 2, 2 }, new float[] { 100f, 150f, 200f, 250f });

        // Act
        float result = metric.Compute(tensor1, tensor2);

        // Assert
        Assert.Equal(float.PositiveInfinity, result);
    }
}
```

#### Step 3.2: Create SSIMMetric Tests

**Location:** `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Evaluation\SSIMMetricTests.cs`

```csharp
using AiDotNet.Evaluation.Metrics;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

/// <summary>
/// Unit tests for SSIMMetric.
/// </summary>
public class SSIMMetricTests
{
    [Fact]
    public void Compute_IdenticalTensors_ReturnsOne()
    {
        // Arrange
        var metric = new SSIMMetric<double>();
        var tensor = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            100, 150, 200,
            120, 180, 220,
            110, 160, 210
        });

        // Act
        double result = metric.Compute(tensor, tensor);

        // Assert
        Assert.Equal(1.0, result, precision: 10);
    }

    [Fact]
    public void Compute_SimilarTensors_ReturnsHighSSIM()
    {
        // Arrange
        var metric = new SSIMMetric<double>();
        var original = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            100, 150, 200,
            120, 180, 220,
            110, 160, 210
        });
        var modified = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            101, 151, 201,  // Slight differences
            121, 181, 221,
            111, 161, 211
        });

        // Act
        double result = metric.Compute(modified, original);

        // Assert
        Assert.True(result > 0.99, $"Expected SSIM > 0.99 for very similar images, got {result}");
    }

    [Fact]
    public void Compute_VeryDifferentTensors_ReturnsLowSSIM()
    {
        // Arrange
        var metric = new SSIMMetric<double>();
        var tensor1 = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            10, 15, 20,
            12, 18, 22,
            11, 16, 21
        });
        var tensor2 = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            200, 210, 220,
            195, 215, 225,
            205, 200, 210
        });

        // Act
        double result = metric.Compute(tensor1, tensor2);

        // Assert
        Assert.True(result < 0.5, $"Expected SSIM < 0.5 for very different images, got {result}");
    }

    [Fact]
    public void Compute_DifferentShapes_ThrowsArgumentException()
    {
        // Arrange
        var metric = new SSIMMetric<double>();
        var tensor1 = new Tensor<double>(new[] { 2, 2 });
        var tensor2 = new Tensor<double>(new[] { 3, 3 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => metric.Compute(tensor1, tensor2));
    }

    [Fact]
    public void Compute_NullPredictions_ThrowsArgumentNullException()
    {
        // Arrange
        var metric = new SSIMMetric<double>();
        var target = new Tensor<double>(new[] { 2, 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => metric.Compute(null!, target));
    }

    [Fact]
    public void Compute_NullTargets_ThrowsArgumentNullException()
    {
        // Arrange
        var metric = new SSIMMetric<double>();
        var prediction = new Tensor<double>(new[] { 2, 2 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => metric.Compute(prediction, null!));
    }

    [Fact]
    public void Constructor_NegativeL_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SSIMMetric<double>(L: -1.0));
    }

    [Fact]
    public void Constructor_NegativeK1_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SSIMMetric<double>(k1: -0.01));
    }

    [Fact]
    public void Constructor_NegativeK2_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SSIMMetric<double>(k2: -0.03));
    }

    [Fact]
    public void Compute_NormalizedImages_WorksWithL1()
    {
        // Arrange: Normalized images (0.0 to 1.0 range)
        var metric = new SSIMMetric<double>(L: 1.0);
        var original = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            0.4, 0.5, 0.6,
            0.45, 0.55, 0.65,
            0.42, 0.52, 0.62
        });

        // Act
        double result = metric.Compute(original, original);

        // Assert
        Assert.Equal(1.0, result, precision: 10);
    }

    [Fact]
    public void Compute_FloatType_WorksCorrectly()
    {
        // Arrange
        var metric = new SSIMMetric<float>();
        var tensor = new Tensor<float>(new[] { 2, 2 }, new float[] { 100f, 150f, 200f, 250f });

        // Act
        float result = metric.Compute(tensor, tensor);

        // Assert
        Assert.Equal(1.0f, result, precision: 6);
    }

    [Fact]
    public void Compute_BrightnessShift_MaintainsHighSSIM()
    {
        // Arrange: SSIM should be robust to brightness changes
        var metric = new SSIMMetric<double>();
        var original = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            100, 150, 200,
            120, 180, 220,
            110, 160, 210
        });
        // Uniformly brighter (same structure)
        var brighter = new Tensor<double>(new[] { 3, 3 }, new double[]
        {
            120, 170, 220,
            140, 200, 240,
            130, 180, 230
        });

        // Act
        double result = metric.Compute(brighter, original);

        // Assert
        // SSIM should be high because structure is preserved
        Assert.True(result > 0.8, $"Expected SSIM > 0.8 for brightness shift, got {result}");
    }
}
```

#### Step 3.3: Create WERMetric Tests

**Location:** `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Evaluation\WERMetricTests.cs`

```csharp
using AiDotNet.Evaluation.Metrics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

/// <summary>
/// Unit tests for WERMetric.
/// </summary>
public class WERMetricTests
{
    [Fact]
    public void Compute_IdenticalStrings_ReturnsZero()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat on the mat";
        string hypothesis = "the cat sat on the mat";

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(0.0, result);
    }

    [Fact]
    public void Compute_OneSubstitution_ReturnsCorrectWER()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat on the mat";  // 6 words
        string hypothesis = "the cat sat on a mat";   // "the" → "a" = 1 substitution

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // WER = 1 / 6 ≈ 0.1667
        Assert.Equal(1.0 / 6.0, result, precision: 4);
    }

    [Fact]
    public void Compute_OneDeletion_ReturnsCorrectWER()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat on the mat";  // 6 words
        string hypothesis = "cat sat on the mat";     // "the" deleted = 1 deletion

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // WER = 1 / 6 ≈ 0.1667
        Assert.Equal(1.0 / 6.0, result, precision: 4);
    }

    [Fact]
    public void Compute_OneInsertion_ReturnsCorrectWER()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat on the mat";          // 6 words
        string hypothesis = "the cat sat happily on the mat"; // "happily" inserted = 1 insertion

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // WER = 1 / 6 ≈ 0.1667
        Assert.Equal(1.0 / 6.0, result, precision: 4);
    }

    [Fact]
    public void Compute_MultipleErrors_ReturnsCorrectWER()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat on the mat";     // 6 words
        string hypothesis = "a dog sits on mat";         // Multiple errors

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // Errors:
        // - "the" → "a" (substitution)
        // - "cat" → "dog" (substitution)
        // - "sat" → "sits" (substitution)
        // - "the" deleted
        // Total: 4 errors / 6 words = 0.6667
        Assert.True(result > 0.5 && result < 0.8, $"Expected WER between 0.5 and 0.8, got {result}");
    }

    [Fact]
    public void Compute_EmptyReference_ReturnsZeroForEmptyHypothesis()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "";
        string hypothesis = "";

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(0.0, result);
    }

    [Fact]
    public void Compute_EmptyReference_ReturnsWordCountForNonEmptyHypothesis()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "";
        string hypothesis = "hello world test";  // 3 words

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        Assert.Equal(3.0, result);  // 3 insertions / 0 reference words
    }

    [Fact]
    public void Compute_EmptyHypothesis_ReturnsOne()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat";  // 3 words
        string hypothesis = "";            // 0 words

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // 3 deletions / 3 reference words = 1.0 (100% error)
        Assert.Equal(1.0, result);
    }

    [Fact]
    public void Compute_CaseSensitive_DifferentCaseIsMistake()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "hello world";
        string hypothesis = "Hello World";  // Different case

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // Both words are substitutions (case-sensitive comparison)
        // 2 errors / 2 words = 1.0
        Assert.Equal(1.0, result);
    }

    [Fact]
    public void Compute_MultipleSpaces_HandledCorrectly()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the  cat   sat";      // Multiple spaces
        string hypothesis = "the cat sat";        // Single spaces

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // Should treat both as ["the", "cat", "sat"] -> WER = 0
        Assert.Equal(0.0, result);
    }

    [Fact]
    public void Compute_NullPrediction_ThrowsArgumentNullException()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "test";

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => metric.Compute(null!, reference));
    }

    [Fact]
    public void Compute_NullTarget_ThrowsArgumentNullException()
    {
        // Arrange
        var metric = new WERMetric();
        string prediction = "test";

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => metric.Compute(prediction, null!));
    }

    [Fact]
    public void Compute_CompletelyWrongTranscription_ReturnsHighWER()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "the cat sat on the mat";
        string hypothesis = "totally different words here";

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // Should be very high error rate (close to or > 1.0)
        Assert.True(result > 0.8, $"Expected WER > 0.8 for completely different text, got {result}");
    }

    [Fact]
    public void Compute_WERCanExceedOne_WithManyInsertions()
    {
        // Arrange
        var metric = new WERMetric();
        string reference = "hi";  // 1 word
        string hypothesis = "hello there my friend how are you doing today";  // 9 words

        // Act
        double result = metric.Compute(hypothesis, reference);

        // Assert
        // Many insertions and substitutions / 1 reference word -> WER > 1.0
        Assert.True(result > 1.0, $"Expected WER > 1.0 for many insertions, got {result}");
    }
}
```

**Testing Coverage Summary:**

1. **PSNRMetric Tests:**
   - Identical images → Infinity
   - Black vs white → Near 0 dB
   - Slightly different → High PSNR (> 50 dB)
   - Different shapes → Exception
   - Null inputs → Exceptions
   - Invalid maxValue → Exception
   - Normalized images (maxValue=1.0)
   - Generic type (float)

2. **SSIMMetric Tests:**
   - Identical images → 1.0
   - Similar images → High SSIM (> 0.99)
   - Very different → Low SSIM (< 0.5)
   - Different shapes → Exception
   - Null inputs → Exceptions
   - Invalid parameters → Exceptions
   - Normalized images (L=1.0)
   - Generic type (float)
   - Brightness shift → Robust (high SSIM maintained)

3. **WERMetric Tests:**
   - Identical → 0.0
   - One substitution → 1/6 ≈ 0.167
   - One deletion → 1/6 ≈ 0.167
   - One insertion → 1/6 ≈ 0.167
   - Multiple errors → Correct calculation
   - Empty strings → Correct handling
   - Case sensitivity → Different case is error
   - Multiple spaces → Handled correctly
   - Null inputs → Exceptions
   - Completely wrong → High WER (> 0.8)
   - Many insertions → WER > 1.0

---

## Testing Strategy

### Unit Testing Requirements

**Minimum 80% code coverage** is required for all new code.

### Test Categories

1. **Happy Path Tests:**
   - Valid inputs that should succeed
   - Known input/output pairs
   - Edge cases within valid range

2. **Error Handling Tests:**
   - Null inputs
   - Mismatched shapes
   - Invalid parameters
   - Out-of-range values

3. **Edge Case Tests:**
   - Empty tensors
   - Single-element tensors
   - Very large tensors
   - Boundary values

4. **Type Generic Tests:**
   - Test with `double`
   - Test with `float`
   - Ensure NumOps handles conversions

### Running Tests

```bash
# Run all tests
dotnet test

# Run specific test class
dotnet test --filter "FullyQualifiedName~PSNRMetricTests"

# Run with coverage
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover
```

---

## Common Pitfalls

### 1. Using Hardcoded Types

```csharp
// ❌ WRONG
public double Compute(Tensor<double> predictions, Tensor<double> targets)
{
    double sum = 0.0;
    return sum / predictions.Size;
}

// ✅ CORRECT
public T Compute(Tensor<T> predictions, Tensor<T> targets)
{
    T sum = NumOps.Zero;
    T size = NumOps.FromInt32(predictions.Size);
    return NumOps.Divide(sum, size);
}
```

### 2. Using default(T) or default!

```csharp
// ❌ WRONG
T zero = default(T);
string name = default!;

// ✅ CORRECT
T zero = NumOps.Zero;
string name = string.Empty;
```

### 3. Direct Arithmetic Operations

```csharp
// ❌ WRONG
T sum = a + b;
T product = a * b;

// ✅ CORRECT
T sum = NumOps.Add(a, b);
T product = NumOps.Multiply(a, b);
```

### 4. Incorrect File Organization

```csharp
// ❌ WRONG
// src/Evaluation/Metrics/IMetric.cs  (interface in subfolder)

// ✅ CORRECT
// src/Interfaces/IMetric.cs  (interface in root Interfaces folder)
```

### 5. Missing Validation

```csharp
// ❌ WRONG
public T Compute(Tensor<T> predictions, Tensor<T> targets)
{
    return CalculateMSE(predictions, targets);
}

// ✅ CORRECT
public T Compute(Tensor<T> predictions, Tensor<T> targets)
{
    if (predictions == null)
        throw new ArgumentNullException(nameof(predictions));
    if (targets == null)
        throw new ArgumentNullException(nameof(targets));
    if (!predictions.Shape.SequenceEqual(targets.Shape))
        throw new ArgumentException("Shapes must match");

    return CalculateMSE(predictions, targets);
}
```

### 6. Not Testing Edge Cases

Always test:
- Null inputs
- Empty tensors
- Mismatched dimensions
- Boundary values (0, infinity, max value)
- Different generic types

### 7. Poor Documentation

```csharp
// ❌ WRONG
/// <summary>Calculates PSNR</summary>
public T Compute(Tensor<T> a, Tensor<T> b)

// ✅ CORRECT
/// <summary>
/// Computes the PSNR between two images.
/// </summary>
/// <param name="predictions">The predicted or compressed image tensor.</param>
/// <param name="targets">The ground truth or original image tensor.</param>
/// <returns>The PSNR value in decibels (dB).</returns>
/// <remarks>
/// <para><b>For Beginners:</b> [Detailed explanation]</para>
/// </remarks>
public T Compute(Tensor<T> predictions, Tensor<T> targets)
```

---

## Resources

### Research Papers

1. **PSNR and MSE:**
   - Widely used in image processing literature
   - Reference: Huynh-Thu & Ghanbari, "Scope of validity of PSNR in image/video quality assessment" (2008)

2. **SSIM:**
   - Original paper: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE TIP, 2004
   - Link: https://www.cns.nyu.edu/~lcv/ssim/

3. **WER:**
   - Standard metric in speech recognition
   - Reference: NIST Speech Recognition Scoring Toolkit
   - Used in: Kaldi, ESPnet, wav2vec2 papers

### AiDotNet Documentation

- `PROJECT_RULES.md`: Architecture and coding standards
- `USER_STORY_ARCHITECTURAL_REQUIREMENTS.md`: Implementation patterns
- Existing metric implementations: `src/RetrievalAugmentedGeneration/Evaluation/`

### External Learning Resources

1. **PSNR Tutorial:**
   - https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
   - Includes formula derivation and examples

2. **SSIM Tutorial:**
   - https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
   - Visual examples of SSIM vs MSE

3. **WER and Levenshtein Distance:**
   - https://en.wikipedia.org/wiki/Levenshtein_distance
   - https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510

4. **Dynamic Programming:**
   - https://www.youtube.com/watch?v=Qc2ieXRgp0U (Edit Distance explained)
   - Helps understand the WER algorithm

### Tools

1. **Image Quality Assessment:**
   - Python: scikit-image (skimage.metrics.structural_similarity)
   - MATLAB: Image Processing Toolbox

2. **Speech Recognition Evaluation:**
   - Python: jiwer library for WER calculation
   - Kaldi: Speech recognition toolkit

---

## Checklist

Before submitting your PR, ensure:

- [ ] All three interfaces created (`IMetric<T>`, `ITextMetric`)
- [ ] All three metrics implemented (`PSNRMetric<T>`, `SSIMMetric<T>`, `WERMetric`)
- [ ] All files in correct locations (interfaces in `src/Interfaces/`)
- [ ] All classes use `INumericOperations<T>` correctly
- [ ] No use of `default(T)` or `default!`
- [ ] All properties properly initialized
- [ ] Comprehensive XML documentation with beginner remarks
- [ ] All unit tests passing
- [ ] Test coverage >= 80%
- [ ] Edge cases tested (null, empty, mismatched shapes)
- [ ] Multiple generic types tested (double, float)
- [ ] No hardcoded numeric types
- [ ] Constructor parameters have defaults with documented rationale
- [ ] All exceptions properly thrown and tested

---

## Getting Help

If you get stuck:

1. **Check existing code** for similar patterns (e.g., `RAGMetricBase`)
2. **Review PROJECT_RULES.md** for architectural guidance
3. **Ask questions** in the issue comments
4. **Reference the test cases** - they show expected behavior
5. **Use the debugger** to step through calculations

Good luck with your implementation!
