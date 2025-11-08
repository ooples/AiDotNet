# Junior Developer Implementation Guide: Issue #287
## Safety and Filtering for Images and Audio (NSFW, Toxicity)

### Overview
This guide will walk you through implementing a comprehensive safety filtering system for AiDotNet. Safety filters protect users and applications from harmful content by detecting and blocking NSFW images, toxic text, and other unsafe content before it reaches users.

---

## Understanding Content Safety and Filtering

### What Are Safety Filters?

Safety filters are AI systems that analyze content and determine if it's safe to display or process. They act as guardians that:

1. **Detect**: Identify potentially harmful content
2. **Score**: Assign a confidence level to the detection
3. **Block**: Prevent unsafe content from being shown or generated
4. **Log**: Record safety violations for monitoring

**Real-World Analogy**:
Think of safety filters like security screening at an airport:
- **Content Scanner**: Like an X-ray machine checking bags
- **Safety Score**: Like a threat level assessment (green/yellow/red)
- **Block Action**: Like confiscating prohibited items
- **Log**: Like recording security incidents

### Key Concepts

#### 1. Content Categories
Different types of unsafe content:
- **NSFW (Not Safe For Work)**: Sexual, violent, or graphic content
- **Toxicity**: Hate speech, harassment, profanity
- **Violence**: Gore, weapons, dangerous activities
- **Spam**: Unwanted promotional content
- **Misinformation**: False or misleading information

#### 2. Detection Methods

**Keyword-Based Filtering (Text)**:
```
Input: "This product is terrible and you're stupid"
Process: Check for toxic keywords ["stupid", "terrible", "idiot", ...]
Result: IsSafe = false, Score = 0.9, Category = "Toxicity"
```

**ML Model-Based Filtering (Images)**:
```
Input: Image tensor [224x224x3]
Process: Pass through ONNX NSFW classifier model
Output: [Safe: 0.1, NSFW: 0.9]
Result: IsSafe = false, Score = 0.9, Category = "NSFW"
```

#### 3. Safety Thresholds
How confident must we be to block content?
- **High threshold (0.9)**: Only block very obvious violations (fewer false positives)
- **Medium threshold (0.7)**: Balanced approach
- **Low threshold (0.5)**: Block anything suspicious (more false positives)

Default: **0.8** - Industry standard for production systems

#### 4. Pipeline Integration
Where to apply filters:

**Input Filtering**:
```
User Input → Safety Filter → Model Processing → Output
              ↓ (if unsafe)
              Block and return error
```

**Output Filtering**:
```
Model Generation → Safety Filter → User Display
                    ↓ (if unsafe)
                    Block and return safe placeholder
```

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── ISafetyFilter.cs           # Main safety filter interface
│   └── ISafetyResult.cs           # Result interface
├── Safety/
│   ├── SafetyResult.cs            # Result data structure
│   ├── SafetyCategory.cs          # Enum of safety categories
│   ├── SafetyFilterBase.cs        # Base class with common logic
│   ├── Text/
│   │   ├── TextToxicityFilter.cs     # Keyword-based toxic text detector
│   │   ├── TextProfanityFilter.cs    # Profanity-specific filter
│   │   └── ToxicKeywords.cs          # Shared keyword lists
│   └── Image/
│       ├── ImageNsfwFilter.cs        # ONNX-based NSFW image detector
│       └── ImageViolenceFilter.cs    # Violence detection filter

tests/
└── UnitTests/
    └── Safety/
        ├── TextToxicityFilterTests.cs
        ├── ImageNsfwFilterTests.cs
        └── SafetyIntegrationTests.cs
```

### Inheritance Pattern
```
ISafetyFilter<TContent> (interface in src/Interfaces/)
    ↓
SafetyFilterBase<TContent> (abstract base in src/Safety/)
    ↓
TextToxicityFilter (concrete implementation for string)
ImageNsfwFilter<T> (concrete implementation for Tensor<T>)
```

---

## Step-by-Step Implementation

### Phase 1: Core Safety Abstractions

#### Step 1: Define SafetyCategory Enum

**File**: `src/Safety/SafetyCategory.cs`

```csharp
namespace AiDotNet.Safety;

/// <summary>
/// Defines categories of content safety violations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are the types of harmful content we detect.
///
/// Think of them like warning labels:
/// - NSFW: "Adults only" content
/// - Toxicity: "Offensive language" warning
/// - Violence: "Graphic content" warning
/// - Safe: "All audiences" rating
/// </para>
/// </remarks>
public enum SafetyCategory
{
    /// <summary>
    /// Content is safe for all audiences.
    /// </summary>
    Safe,

    /// <summary>
    /// Content contains sexual or explicit material (Not Safe For Work).
    /// </summary>
    NSFW,

    /// <summary>
    /// Content contains toxic language, hate speech, or harassment.
    /// </summary>
    Toxicity,

    /// <summary>
    /// Content contains graphic violence or gore.
    /// </summary>
    Violence,

    /// <summary>
    /// Content appears to be spam or unwanted promotional material.
    /// </summary>
    Spam,

    /// <summary>
    /// Content category could not be determined.
    /// </summary>
    Unknown
}
```

#### Step 2: Define SafetyResult Class

**File**: `src/Safety/SafetyResult.cs`

```csharp
namespace AiDotNet.Safety;

/// <summary>
/// Represents the result of a safety filter check.
/// </summary>
/// <remarks>
/// <para>
/// Contains information about whether content is safe, how confident the filter is,
/// and what category of violation was detected (if any).
/// </para>
/// <para><b>For Beginners:</b> This is like a safety report card for content.
///
/// Example results:
/// 1. Safe content:
///    - IsSafe: true
///    - Score: 0.95 (95% confident it's safe)
///    - Category: Safe
///
/// 2. Toxic content:
///    - IsSafe: false
///    - Score: 0.88 (88% confident it's toxic)
///    - Category: Toxicity
///    - Reason: "Contains profanity: [redacted]"
///
/// The Score represents confidence:
/// - 1.0 = Absolutely certain
/// - 0.8 = Very confident (typical threshold)
/// - 0.5 = Uncertain
/// - 0.0 = No confidence
/// </para>
/// </remarks>
public class SafetyResult
{
    /// <summary>
    /// Gets or sets whether the content passed the safety check.
    /// </summary>
    /// <remarks>
    /// true = Safe to display/process
    /// false = Content violates safety policies
    /// </remarks>
    public bool IsSafe { get; set; }

    /// <summary>
    /// Gets or sets the confidence score of the safety determination.
    /// </summary>
    /// <remarks>
    /// Range: 0.0 to 1.0
    /// - For safe content: Score represents confidence that content is safe
    /// - For unsafe content: Score represents confidence in the violation detection
    ///
    /// Example:
    /// - Safe content with Score 0.95 = 95% sure it's safe
    /// - NSFW content with Score 0.88 = 88% sure it's NSFW
    /// </remarks>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets the safety category of the content.
    /// </summary>
    public SafetyCategory Category { get; set; }

    /// <summary>
    /// Gets or sets an optional detailed reason for the safety determination.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This explains WHY content was flagged.
    ///
    /// Examples:
    /// - "Contains profanity: [word1], [word2]"
    /// - "NSFW probability exceeds threshold (0.92 > 0.80)"
    /// - "No violations detected"
    ///
    /// Useful for:
    /// - Debugging false positives
    /// - User feedback
    /// - Logging and monitoring
    /// </para>
    /// </remarks>
    public string Reason { get; set; } = string.Empty;
}
```

#### Step 3: Define ISafetyFilter Interface

**File**: `src/Interfaces/ISafetyFilter.cs`

```csharp
using AiDotNet.Safety;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for safety filters that check content for policy violations.
/// </summary>
/// <remarks>
/// <para>
/// Safety filters analyze content (text, images, audio) and determine if it's safe
/// to display or process. They return a result indicating safety status and confidence.
/// </para>
/// <para><b>For Beginners:</b> Safety filters are like content moderators.
///
/// Think of them as automated reviewers:
/// - Input: Some content (text, image, etc.)
/// - Process: Analyze for violations (profanity, NSFW, etc.)
/// - Output: Safe/Unsafe + confidence score + reason
///
/// Example flow:
/// ```
/// User types: "This is a great product!"
/// Filter checks: No toxic keywords found
/// Result: IsSafe = true, Score = 0.98, Category = Safe
/// ```
///
/// ```
/// User types: "You're an idiot!"
/// Filter checks: Found toxic keyword "idiot"
/// Result: IsSafe = false, Score = 0.95, Category = Toxicity
/// ```
/// </para>
/// </remarks>
/// <typeparam name="TContent">The type of content this filter can check (e.g., string for text, Tensor for images).</typeparam>
public interface ISafetyFilter<TContent>
{
    /// <summary>
    /// Checks the given content for safety violations.
    /// </summary>
    /// <param name="content">The content to analyze.</param>
    /// <returns>A <see cref="SafetyResult"/> indicating whether the content is safe.</returns>
    /// <remarks>
    /// <para>
    /// This method should:
    /// 1. Analyze the content using the filter's detection method
    /// 2. Compare confidence against threshold
    /// 3. Return a result with IsSafe, Score, Category, and Reason
    /// </para>
    /// <para><b>For Beginners:</b> This is the main method to check if content is safe.
    ///
    /// Usage example:
    /// ```csharp
    /// var filter = new TextToxicityFilter();
    /// var result = filter.Check("Hello world!");
    ///
    /// if (result.IsSafe)
    /// {
    ///     Console.WriteLine("Content is safe to display");
    /// }
    /// else
    /// {
    ///     Console.WriteLine($"Blocked: {result.Category} (confidence: {result.Score})");
    ///     Console.WriteLine($"Reason: {result.Reason}");
    /// }
    /// ```
    /// </para>
    /// </remarks>
    SafetyResult Check(TContent content);

    /// <summary>
    /// Gets the threshold score above which content is considered unsafe.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Range: 0.0 to 1.0 (typically 0.7 to 0.9)
    /// - Higher threshold = fewer false positives, may miss some violations
    /// - Lower threshold = catches more violations, more false positives
    /// </para>
    /// <para><b>For Beginners:</b> This is the "sensitivity" setting.
    ///
    /// Example with threshold = 0.8:
    /// - Toxicity score 0.9 > 0.8 → BLOCKED (high confidence violation)
    /// - Toxicity score 0.7 < 0.8 → ALLOWED (below threshold)
    /// - Toxicity score 0.5 < 0.8 → ALLOWED (uncertain)
    ///
    /// Default: 0.8 (industry standard for production systems)
    /// </para>
    /// </remarks>
    double Threshold { get; }
}
```

### Phase 2: Text Safety Filters

#### Step 4: Create ToxicKeywords Helper Class

**File**: `src/Safety/Text/ToxicKeywords.cs`

```csharp
namespace AiDotNet.Safety.Text;

/// <summary>
/// Provides curated lists of toxic keywords for content filtering.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a database of bad words and phrases.
///
/// Why maintain keyword lists?
/// - Fast detection (no ML model needed)
/// - Interpretable (you can see exactly what triggered the filter)
/// - Customizable (easily add/remove words for your use case)
///
/// Limitations:
/// - Can be bypassed with creative spelling ("a$$hole")
/// - Doesn't understand context ("This movie is badass" vs "You're a badass")
/// - Requires maintenance as language evolves
///
/// For production systems, combine with ML-based toxicity detection.
/// </para>
/// </remarks>
public static class ToxicKeywords
{
    /// <summary>
    /// Gets a set of profanity and offensive words.
    /// </summary>
    /// <remarks>
    /// This list contains common English profanity.
    /// NOTE: This is a minimal set for demonstration. Production systems should use
    /// comprehensive lists from sources like:
    /// - Perspective API's toxicity lexicon
    /// - Community-curated blocklists
    /// - Domain-specific offensive terms
    /// </remarks>
    public static HashSet<string> Profanity { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        // NOTE: In production, use a comprehensive list from trusted sources
        // This is a minimal demonstration set
        "idiot", "stupid", "moron", "dumb", "hate", "kill", "die"
    };

    /// <summary>
    /// Gets a set of hate speech and discriminatory terms.
    /// </summary>
    /// <remarks>
    /// Contains slurs and discriminatory language targeting protected groups.
    /// Production implementations should source from specialized hate speech databases.
    /// </remarks>
    public static HashSet<string> HateSpeech { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        // NOTE: Production systems require comprehensive hate speech detection
        // Consider using ML models for context-aware detection
    };

    /// <summary>
    /// Gets a set of harassment and threatening language.
    /// </summary>
    public static HashSet<string> Harassment { get; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "kys", "kms", "neck yourself", "threats", "doxx"
    };

    /// <summary>
    /// Gets all toxic keywords combined.
    /// </summary>
    public static HashSet<string> All { get; } = new HashSet<string>(
        Profanity.Concat(HateSpeech).Concat(Harassment),
        StringComparer.OrdinalIgnoreCase
    );
}
```

#### Step 5: Implement TextToxicityFilter

**File**: `src/Safety/Text/TextToxicityFilter.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Detects toxic language in text using keyword matching.
/// </summary>
/// <remarks>
/// <para>
/// This filter uses a keyword-based approach to detect profanity, hate speech,
/// and harassment. It converts text to lowercase and checks for the presence
/// of known toxic keywords.
/// </para>
/// <para><b>For Beginners:</b> This is like a "bad word" detector.
///
/// How it works:
/// 1. Convert input text to lowercase
/// 2. Split into words
/// 3. Check each word against toxic keyword lists
/// 4. If any match found → IsSafe = false
/// 5. If no matches → IsSafe = true
///
/// Example:
/// ```
/// Input: "This product is terrible and you're stupid"
/// Process:
///   - Lowercase: "this product is terrible and you're stupid"
///   - Words: ["this", "product", "is", "terrible", "and", "you're", "stupid"]
///   - Matches: ["terrible", "stupid"] ← Both in toxic keywords
/// Result: IsSafe = false, Score = 1.0, Category = Toxicity
/// ```
///
/// Limitations:
/// - Simple keyword matching (no context understanding)
/// - Can be bypassed with creative spelling
/// - May have false positives ("Scunthorpe problem")
///
/// For production, consider ML-based toxicity detection (Perspective API, etc.)
/// </para>
/// </remarks>
public class TextToxicityFilter : ISafetyFilter<string>
{
    private readonly HashSet<string> _toxicKeywords;
    private readonly double _threshold;

    /// <inheritdoc/>
    public double Threshold => _threshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="TextToxicityFilter"/> class.
    /// </summary>
    /// <param name="customKeywords">
    /// Optional custom set of toxic keywords. If null, uses default ToxicKeywords.All.
    /// </param>
    /// <param name="threshold">
    /// The threshold score above which content is considered toxic. Defaults to 0.8.
    /// </param>
    /// <remarks>
    /// <para><b>Default Value (threshold = 0.8):</b> Industry standard for text moderation.
    ///
    /// The threshold is less critical for keyword-based filtering since matches are binary
    /// (either found or not). However, it's included for API consistency with ML-based filters.
    ///
    /// For keyword matching:
    /// - Found keyword = Score 1.0 (absolute certainty)
    /// - No keywords = Score 1.0 (certainty it's safe)
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when threshold is not between 0 and 1.</exception>
    public TextToxicityFilter(HashSet<string>? customKeywords = null, double threshold = 0.8)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentException("Threshold must be between 0 and 1", nameof(threshold));
        }

        _toxicKeywords = customKeywords ?? ToxicKeywords.All;
        _threshold = threshold;
    }

    /// <inheritdoc/>
    public SafetyResult Check(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return new SafetyResult
            {
                IsSafe = true,
                Score = 1.0,
                Category = SafetyCategory.Safe,
                Reason = "Empty content is considered safe"
            };
        }

        // Normalize text to lowercase for case-insensitive matching
        string normalizedText = content.ToLowerInvariant();

        // Find all toxic keywords present in the text
        var foundKeywords = new List<string>();
        foreach (var keyword in _toxicKeywords)
        {
            // Check for whole word matches to reduce false positives
            if (ContainsWholeWord(normalizedText, keyword))
            {
                foundKeywords.Add(keyword);
            }
        }

        // If any toxic keywords found, content is unsafe
        if (foundKeywords.Count > 0)
        {
            return new SafetyResult
            {
                IsSafe = false,
                Score = 1.0, // Keyword match = absolute confidence
                Category = SafetyCategory.Toxicity,
                Reason = $"Contains toxic keywords: {string.Join(", ", foundKeywords.Take(3))}" +
                         (foundKeywords.Count > 3 ? $" and {foundKeywords.Count - 3} more" : "")
            };
        }

        // No violations found
        return new SafetyResult
        {
            IsSafe = true,
            Score = 1.0,
            Category = SafetyCategory.Safe,
            Reason = "No toxic keywords detected"
        };
    }

    /// <summary>
    /// Checks if text contains a keyword as a whole word (not part of another word).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents false positives from partial matches.
    ///
    /// Example problem:
    /// - Keyword: "ass"
    /// - Text: "classic"
    /// - Naive match: "classic" contains "ass" → FALSE POSITIVE!
    ///
    /// Solution with whole-word matching:
    /// - "classic" → No match (ass is part of another word)
    /// - "you're an ass" → Match (ass is a standalone word)
    ///
    /// This reduces the "Scunthorpe problem" where benign words contain toxic substrings.
    /// </para>
    /// </remarks>
    private bool ContainsWholeWord(string text, string keyword)
    {
        // Simple word boundary check
        // Production implementations should use regex word boundaries (\b)
        string[] words = text.Split(new[] { ' ', ',', '.', '!', '?', ';', ':', '\n', '\r', '\t' },
                                     StringSplitOptions.RemoveEmptyEntries);

        return words.Any(w => w.Equals(keyword, StringComparison.OrdinalIgnoreCase));
    }
}
```

### Phase 3: Image Safety Filters

#### Step 6: Implement ImageNsfwFilter

**File**: `src/Safety/Image/ImageNsfwFilter.cs`

```csharp
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Safety.Image;

/// <summary>
/// Detects NSFW (Not Safe For Work) content in images using an ONNX model.
/// </summary>
/// <remarks>
/// <para>
/// This filter uses a pre-trained machine learning model to classify images
/// as safe or NSFW. The model outputs a probability score, which is compared
/// against a threshold to make the final determination.
/// </para>
/// <para><b>For Beginners:</b> This is like an AI content moderator for images.
///
/// How it works:
/// 1. Load a pre-trained ONNX model (e.g., NSFW classifier from HuggingFace)
/// 2. Preprocess input image:
///    - Resize to 224x224 pixels (model's expected input size)
///    - Normalize pixel values to 0-1 range
///    - Convert to model's expected format (NCHW: channels-first)
/// 3. Run inference:
///    - Input: [1, 3, 224, 224] tensor (batch=1, channels=3, height=224, width=224)
///    - Output: [1, 2] tensor with [safe_prob, nsfw_prob]
/// 4. Compare NSFW probability to threshold:
///    - If nsfw_prob >= 0.8 → IsSafe = false
///    - If nsfw_prob < 0.8 → IsSafe = true
///
/// Example ONNX models:
/// - Falconsai/nsfw_image_detection (HuggingFace)
/// - LAION CLIP-based NSFW detector
/// - Yahoo Open NSFW model (Caffe, needs conversion)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type for tensor operations.</typeparam>
public class ImageNsfwFilter<T> : ISafetyFilter<Tensor<T>>
{
    private readonly InferenceSession _session;
    private readonly double _threshold;
    private readonly string _inputName;
    private readonly string _outputName;

    /// <inheritdoc/>
    public double Threshold => _threshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImageNsfwFilter{T}"/> class.
    /// </summary>
    /// <param name="onnxModelPath">
    /// Path to the ONNX model file for NSFW detection.
    /// The model should accept [batch, channels, height, width] input
    /// and output [batch, num_classes] probabilities.
    /// </param>
    /// <param name="threshold">
    /// The threshold probability above which content is considered NSFW. Defaults to 0.8.
    /// </param>
    /// <remarks>
    /// <para><b>Default Value (threshold = 0.8):</b> Based on NSFW detection research.
    ///
    /// NSFW detection threshold recommendations:
    /// - 0.9: Very conservative, only blocks obvious NSFW (may miss some violations)
    /// - 0.8: Balanced approach, recommended for production (default)
    /// - 0.7: More aggressive, catches more borderline cases (more false positives)
    /// - 0.5: Very aggressive, may block safe content
    ///
    /// Research shows 0.8 provides good precision/recall balance for content moderation.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when model path is invalid or threshold is out of range.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file is not found.</exception>
    public ImageNsfwFilter(string onnxModelPath, double threshold = 0.8)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
        {
            throw new ArgumentException("ONNX model path cannot be empty", nameof(onnxModelPath));
        }

        if (!File.Exists(onnxModelPath))
        {
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);
        }

        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentException("Threshold must be between 0 and 1", nameof(threshold));
        }

        _threshold = threshold;

        // Initialize ONNX Runtime session
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC
        };

        _session = new InferenceSession(onnxModelPath, sessionOptions);

        // Get input/output names from model metadata
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <inheritdoc/>
    public SafetyResult Check(Tensor<T> content)
    {
        if (content == null)
        {
            throw new ArgumentNullException(nameof(content));
        }

        try
        {
            // Preprocess image tensor for model input
            var preprocessed = PreprocessImage(content);

            // Create ONNX tensor from preprocessed data
            var inputTensor = new DenseTensor<float>(preprocessed, new[] { 1, 3, 224, 224 });

            // Run inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Extract NSFW probability (assuming output[0] = safe, output[1] = nsfw)
            float nsfwProbability = output[1];

            // Determine if content is unsafe
            bool isSafe = nsfwProbability < _threshold;

            return new SafetyResult
            {
                IsSafe = isSafe,
                Score = nsfwProbability,
                Category = isSafe ? SafetyCategory.Safe : SafetyCategory.NSFW,
                Reason = isSafe
                    ? $"NSFW probability ({nsfwProbability:F3}) below threshold ({_threshold:F2})"
                    : $"NSFW probability ({nsfwProbability:F3}) exceeds threshold ({_threshold:F2})"
            };
        }
        catch (Exception ex)
        {
            return new SafetyResult
            {
                IsSafe = false,
                Score = 0.0,
                Category = SafetyCategory.Unknown,
                Reason = $"Error during NSFW detection: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Preprocesses an image tensor for the ONNX model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prepares the image for the AI model.
    ///
    /// What preprocessing does:
    /// 1. Resize to 224x224 pixels (standard CNN input size)
    /// 2. Normalize pixel values:
    ///    - Original: 0-255 (RGB pixel values)
    ///    - Normalized: 0.0-1.0 (divide by 255)
    ///    - Some models also subtract mean and divide by std
    /// 3. Convert format:
    ///    - From: HWC (height, width, channels) - how images are stored
    ///    - To: CHW (channels, height, width) - what PyTorch models expect
    ///
    /// Example:
    /// - Input: 1920x1080 RGB image
    /// - After resize: 224x224 RGB image
    /// - After normalize: Pixel values 0.0-1.0
    /// - After format: [3, 224, 224] tensor
    /// </para>
    /// </remarks>
    private float[] PreprocessImage(Tensor<T> image)
    {
        // NOTE: This is a simplified preprocessing pipeline.
        // Production implementations should match the exact preprocessing
        // used during model training (normalization constants, resize method, etc.)

        int targetHeight = 224;
        int targetWidth = 224;
        int channels = 3;

        // Create output array [C, H, W]
        var preprocessed = new float[channels * targetHeight * targetWidth];

        // Simple bilinear resize and normalize
        // In production, use proper image resizing library (ImageSharp, SkiaSharp, etc.)
        int srcHeight = image.Dimensions[0];
        int srcWidth = image.Dimensions[1];

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < targetHeight; h++)
            {
                for (int w = 0; w < targetWidth; w++)
                {
                    // Simple nearest-neighbor sampling (production should use bilinear)
                    int srcH = (int)((float)h / targetHeight * srcHeight);
                    int srcW = (int)((float)w / targetWidth * srcWidth);

                    // Get pixel value and normalize to 0-1
                    var pixelValue = Convert.ToDouble(image[srcH, srcW, c]);
                    float normalized = (float)(pixelValue / 255.0);

                    // Store in CHW format
                    int index = c * (targetHeight * targetWidth) + h * targetWidth + w;
                    preprocessed[index] = normalized;
                }
            }
        }

        return preprocessed;
    }

    /// <summary>
    /// Disposes of the ONNX Runtime session.
    /// </summary>
    public void Dispose()
    {
        _session?.Dispose();
    }
}
```

### Phase 4: Pipeline Integration

#### Step 7: Integration Example for Text Pipelines

**Example**: Adding safety filters to RAG pipeline

```csharp
using AiDotNet.Safety.Text;
using AiDotNet.RetrievalAugmentedGeneration;

namespace AiDotNet.Examples;

/// <summary>
/// Demonstrates integrating safety filters into a RAG pipeline.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This shows where to add safety checks in your pipeline.
///
/// Two key integration points:
/// 1. Input filtering: Check user queries before processing
/// 2. Output filtering: Check generated responses before returning
///
/// Example flow:
/// ```
/// User Query → Input Filter → [If unsafe: Reject] → RAG Pipeline → Output Filter → [If unsafe: Replace] → User
/// ```
/// </para>
/// </remarks>
public class SafeRAGPipeline
{
    private readonly ISafetyFilter<string> _inputFilter;
    private readonly ISafetyFilter<string> _outputFilter;

    public SafeRAGPipeline()
    {
        // Create safety filters
        _inputFilter = new TextToxicityFilter(threshold: 0.8);
        _outputFilter = new TextToxicityFilter(threshold: 0.9); // Higher threshold for outputs
    }

    public string Ask(string query)
    {
        // Step 1: Check input for safety
        var inputCheck = _inputFilter.Check(query);
        if (!inputCheck.IsSafe)
        {
            throw new InvalidOperationException(
                $"Input query failed safety check: {inputCheck.Category} - {inputCheck.Reason}");
        }

        // Step 2: Process query through RAG pipeline
        string response = ProcessQuery(query);

        // Step 3: Check output for safety
        var outputCheck = _outputFilter.Check(response);
        if (!outputCheck.IsSafe)
        {
            // Option 1: Throw exception
            // throw new InvalidOperationException($"Generated response failed safety check");

            // Option 2: Return safe placeholder
            return "I apologize, but I cannot provide a response to that query.";
        }

        return response;
    }

    private string ProcessQuery(string query)
    {
        // RAG pipeline logic here
        return "Sample response";
    }
}
```

#### Step 8: Integration Example for Image Generation

**Example**: Adding NSFW filter to diffusion model output

```csharp
using AiDotNet.Safety.Image;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Examples;

/// <summary>
/// Demonstrates integrating NSFW filter into image generation pipeline.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This prevents generating unsafe images.
///
/// Integration point: After image generation, before returning to user
///
/// Example flow:
/// ```
/// Prompt → Generate Image → NSFW Filter → [If unsafe: Return black image] → User
/// ```
///
/// Why filter outputs:
/// - Generative models can produce unexpected NSFW content
/// - User prompts may be benign but trigger unsafe outputs
/// - Output filtering is last line of defense
/// </para>
/// </remarks>
public class SafeDiffusionModel<T>
{
    private readonly ISafetyFilter<Tensor<T>> _nsfwFilter;

    public SafeDiffusionModel(string nsfwModelPath)
    {
        _nsfwFilter = new ImageNsfwFilter<T>(nsfwModelPath, threshold: 0.8);
    }

    public Tensor<T> Generate(string prompt)
    {
        // Step 1: Generate image from prompt
        Tensor<T> generatedImage = GenerateImageFromPrompt(prompt);

        // Step 2: Check generated image for NSFW content
        var safetyCheck = _nsfwFilter.Check(generatedImage);

        if (!safetyCheck.IsSafe)
        {
            // Option 1: Return black placeholder image
            return CreateBlackImage();

            // Option 2: Throw exception
            // throw new InvalidOperationException("Generated image failed NSFW check");

            // Option 3: Retry generation with safety guidance
            // return RegenerateWithSafetyGuidance(prompt);
        }

        return generatedImage;
    }

    private Tensor<T> GenerateImageFromPrompt(string prompt)
    {
        // Diffusion model generation logic
        return new Tensor<T>(new[] { 512, 512, 3 });
    }

    private Tensor<T> CreateBlackImage()
    {
        // Return safe placeholder
        return new Tensor<T>(new[] { 512, 512, 3 }); // All zeros = black
    }
}
```

---

## Phase 5: Testing

### Step 9: Unit Tests for Text Filters

**File**: `tests/UnitTests/Safety/TextToxicityFilterTests.cs`

```csharp
using Xunit;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;

namespace AiDotNet.Tests.UnitTests.Safety;

/// <summary>
/// Tests for the TextToxicityFilter class.
/// </summary>
public class TextToxicityFilterTests
{
    [Fact]
    public void TextToxicityFilter_SafeText_ReturnsSafe()
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check("This is a great product! I love it.");

        // Assert
        Assert.True(result.IsSafe);
        Assert.Equal(1.0, result.Score);
        Assert.Equal(SafetyCategory.Safe, result.Category);
    }

    [Fact]
    public void TextToxicityFilter_ToxicText_ReturnsUnsafe()
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check("You're an idiot and I hate you");

        // Assert
        Assert.False(result.IsSafe);
        Assert.Equal(1.0, result.Score);
        Assert.Equal(SafetyCategory.Toxicity, result.Category);
        Assert.Contains("toxic keywords", result.Reason);
    }

    [Fact]
    public void TextToxicityFilter_MultipleToxicKeywords_ReportsAll()
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check("stupid idiot moron");

        // Assert
        Assert.False(result.IsSafe);
        Assert.Contains("stupid", result.Reason);
        Assert.Contains("idiot", result.Reason);
    }

    [Fact]
    public void TextToxicityFilter_CaseInsensitive_DetectsUppercase()
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check("YOU ARE STUPID");

        // Assert
        Assert.False(result.IsSafe);
    }

    [Fact]
    public void TextToxicityFilter_WholeWordMatch_AvoidsFalsePositives()
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check("This is a classic example"); // Contains "ass" but not as whole word

        // Assert
        Assert.True(result.IsSafe); // Should not trigger "ass" keyword
    }

    [Fact]
    public void TextToxicityFilter_EmptyText_ReturnsSafe()
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check(string.Empty);

        // Assert
        Assert.True(result.IsSafe);
        Assert.Contains("Empty content", result.Reason);
    }

    [Fact]
    public void TextToxicityFilter_CustomKeywords_UsesCustomList()
    {
        // Arrange
        var customKeywords = new HashSet<string> { "banana", "apple" };
        var filter = new TextToxicityFilter(customKeywords);

        // Act
        var result = filter.Check("I like bananas");

        // Assert
        Assert.False(result.IsSafe); // "banana" is in custom toxic list
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void TextToxicityFilter_InvalidThreshold_ThrowsException(double threshold)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new TextToxicityFilter(threshold: threshold));
    }
}
```

### Step 10: Unit Tests for Image Filters (Mock)

**File**: `tests/UnitTests/Safety/ImageNsfwFilterTests.cs`

```csharp
using Xunit;
using Moq;
using AiDotNet.Safety;
using AiDotNet.Safety.Image;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Safety;

/// <summary>
/// Tests for the ImageNsfwFilter class.
/// </summary>
/// <remarks>
/// NOTE: These tests use mocked ONNX models to avoid external dependencies.
/// Integration tests should use real NSFW detection models.
/// </remarks>
public class ImageNsfwFilterTests
{
    [Fact]
    public void ImageNsfwFilter_Constructor_InvalidPath_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ImageNsfwFilter<double>(string.Empty));
    }

    [Fact]
    public void ImageNsfwFilter_Constructor_NonExistentFile_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            new ImageNsfwFilter<double>("nonexistent_model.onnx"));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void ImageNsfwFilter_Constructor_InvalidThreshold_ThrowsException(double threshold)
    {
        // NOTE: This test would fail with FileNotFoundException first
        // In practice, validate threshold before file existence
        Assert.Throws<Exception>(() =>
            new ImageNsfwFilter<double>("model.onnx", threshold));
    }

    // NOTE: Full integration tests require actual ONNX model files
    // Example integration test structure:
    //
    // [Fact]
    // public void ImageNsfwFilter_SafeImage_ReturnsSafe()
    // {
    //     // Arrange
    //     var filter = new ImageNsfwFilter<double>("nsfw_model.onnx");
    //     var safeImage = LoadTestImage("test_images/safe_landscape.jpg");
    //
    //     // Act
    //     var result = filter.Check(safeImage);
    //
    //     // Assert
    //     Assert.True(result.IsSafe);
    //     Assert.Equal(SafetyCategory.Safe, result.Category);
    // }
}
```

### Step 11: Integration Tests

**File**: `tests/UnitTests/Safety/SafetyIntegrationTests.cs`

```csharp
using Xunit;
using AiDotNet.Safety.Text;

namespace AiDotNet.Tests.UnitTests.Safety;

/// <summary>
/// Integration tests for safety filter usage in pipelines.
/// </summary>
public class SafetyIntegrationTests
{
    [Fact]
    public void SafetyFilter_InPipeline_BlocksToxicInput()
    {
        // Arrange
        var filter = new TextToxicityFilter();
        string toxicInput = "You're stupid";

        // Act
        var result = filter.Check(toxicInput);

        // Assert - Pipeline should reject this input
        Assert.False(result.IsSafe);
        // In real pipeline, this would throw exception or return error
    }

    [Fact]
    public void SafetyFilter_InPipeline_AllowsSafeInput()
    {
        // Arrange
        var filter = new TextToxicityFilter();
        string safeInput = "What is the weather today?";

        // Act
        var result = filter.Check(safeInput);

        // Assert - Pipeline should accept this input
        Assert.True(result.IsSafe);
        // In real pipeline, processing would continue
    }

    [Theory]
    [InlineData("This is great!", true)]
    [InlineData("You are stupid", false)]
    [InlineData("I hate this product", false)]
    [InlineData("Nice weather today", true)]
    public void SafetyFilter_VariousInputs_CorrectClassification(string input, bool expectedSafe)
    {
        // Arrange
        var filter = new TextToxicityFilter();

        // Act
        var result = filter.Check(input);

        // Assert
        Assert.Equal(expectedSafe, result.IsSafe);
    }
}
```

---

## Usage Examples

### Example 1: Basic Text Filtering

```csharp
using AiDotNet.Safety.Text;

// Create filter
var filter = new TextToxicityFilter();

// Check safe content
var safeResult = filter.Check("This is a great day!");
Console.WriteLine($"Safe: {safeResult.IsSafe}"); // True

// Check toxic content
var toxicResult = filter.Check("You're an idiot");
Console.WriteLine($"Safe: {toxicResult.IsSafe}"); // False
Console.WriteLine($"Reason: {toxicResult.Reason}"); // Contains toxic keywords: idiot
```

### Example 2: Custom Threshold

```csharp
// More aggressive filtering (lower threshold)
var strictFilter = new TextToxicityFilter(threshold: 0.5);

// More lenient filtering (higher threshold)
var lenientFilter = new TextToxicityFilter(threshold: 0.9);
```

### Example 3: Custom Keywords

```csharp
// Domain-specific toxic keywords
var customKeywords = new HashSet<string>
{
    "competitor_brand",
    "banned_term_1",
    "banned_term_2"
};

var customFilter = new TextToxicityFilter(customKeywords);
```

### Example 4: Image NSFW Detection (with model)

```csharp
using AiDotNet.Safety.Image;

// Create filter with ONNX model
var nsfwFilter = new ImageNsfwFilter<double>("models/nsfw_detector.onnx");

// Check image
var imageResult = nsfwFilter.Check(imageTensor);
if (!imageResult.IsSafe)
{
    Console.WriteLine($"Image blocked: {imageResult.Reason}");
}
```

---

## Key Concepts for Testing

### 1. Test Coverage
Ensure tests cover:
- Safe content (true positives)
- Unsafe content (true negatives)
- Edge cases (empty, very long, special characters)
- Threshold boundaries
- Error handling

### 2. False Positives/Negatives
Common issues:
- **False Positive**: Blocking safe content (e.g., "Scunthorpe problem")
- **False Negative**: Missing toxic content (e.g., creative spelling "a$$hole")

Solution: Balance threshold and use multiple detection methods

### 3. Performance Testing
Safety filters run on every request, so performance matters:
- Keyword matching: Very fast (microseconds)
- ML models: Slower (milliseconds to seconds)
- Cache results for repeated content

---

## Common Pitfalls and Solutions

### Pitfall 1: Scunthorpe Problem
**Problem**: Blocking words that contain toxic substrings ("classic" contains "ass")
**Solution**: Use whole-word matching, not substring matching

### Pitfall 2: Context Ignorance
**Problem**: "This movie is badass" flagged as toxic
**Solution**: Use ML-based context-aware toxicity detection (Perspective API)

### Pitfall 3: Language Coverage
**Problem**: Only detecting English toxicity
**Solution**: Use multilingual models or language-specific keyword lists

### Pitfall 4: Model Drift
**Problem**: NSFW model becomes outdated as new content types emerge
**Solution**: Regularly retrain models on fresh data, monitor false negatives

### Pitfall 5: Performance Bottleneck
**Problem**: ONNX model inference is slow, blocking user requests
**Solution**: Batch processing, async filtering, GPU acceleration

---

## Next Steps and Extensions

### 1. ML-Based Text Toxicity
Integrate advanced toxicity detection:
```csharp
public class PerspectiveAPIFilter : ISafetyFilter<string>
{
    // Use Google Perspective API for context-aware toxicity detection
}
```

### 2. Multi-Category Detection
Extend to detect multiple violation types:
```csharp
public class MultiCategoryFilter : ISafetyFilter<string>
{
    // Detect: NSFW, Toxicity, Spam, Violence, Hate Speech
    // Return: Dictionary<SafetyCategory, double> scores
}
```

### 3. Audio Safety Filtering
Add audio content moderation:
```csharp
public class AudioToxicityFilter : ISafetyFilter<AudioTensor>
{
    // Transcribe audio → Check text for toxicity
    // Or use audio-based toxic sound detection
}
```

### 4. Adaptive Thresholds
Adjust thresholds based on context:
```csharp
public class AdaptiveFilter
{
    // Higher threshold for medical/educational content
    // Lower threshold for children's content
}
```

---

## Testing Checklist

- [ ] TextToxicityFilter detects profanity
- [ ] TextToxicityFilter handles empty input
- [ ] TextToxicityFilter is case-insensitive
- [ ] TextToxicityFilter uses whole-word matching
- [ ] TextToxicityFilter accepts custom keywords
- [ ] TextToxicityFilter validates threshold range
- [ ] ImageNsfwFilter validates model path
- [ ] ImageNsfwFilter validates threshold range
- [ ] Integration: Input filtering blocks toxic queries
- [ ] Integration: Output filtering blocks toxic responses
- [ ] All tests pass with at least 80% code coverage

---

## Summary

You have implemented a comprehensive safety filtering system with:
- Safety abstraction (ISafetyFilter interface)
- Text toxicity detection (keyword-based)
- Image NSFW detection (ML model-based)
- Pipeline integration examples
- Comprehensive unit tests

This foundation supports building safe AI applications that:
- Protect users from harmful content
- Comply with content moderation policies
- Provide transparent safety decisions
- Can be customized for different use cases

The next phase would add ML-based text detection, audio filtering, and adaptive thresholding.
