using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Safety.Multimodal;

/// <summary>
/// Checks whether an image matches its caption, to catch unsafe images passed off under a safe description. With a
/// CLIP model it compares CLIP embeddings; without one it falls back to colour heuristics.
/// </summary>
/// <remarks>
/// <para>
/// <b>With a CLIP model</b> (an <see cref="IMultimodalEmbedding{T}"/> such as <c>ClipNeuralNetwork</c>, which loads
/// the image and text encoders from ONNX files), <see cref="EvaluateAlignment"/> runs CLIP zero-shot classification
/// (Radford et al. 2021): the image embedding is compared with the embeddings of the caption and of three unsafe-concept
/// prompts (graphic violence, sexually explicit content, hateful symbols), the cosine similarities are scaled by the
/// CLIP logit scale of 100 and turned into probabilities with a softmax, and a finding is reported when an unsafe
/// concept takes at least <c>mismatchThreshold</c> of the probability while the caption itself does not describe
/// unsafe content. The image must already be preprocessed to the model input: 3 x ImageSize x ImageSize values.
/// </para>
/// <para>
/// <b>Without a CLIP model</b> the image side is a colour heuristic (skin-coloured and red-dominant pixel fractions)
/// and the text side a keyword list. That fallback does not understand either modality and only flags a "safe" caption
/// on a skin- or red-heavy image; treat it as a weak signal.
/// </para>
/// <para>
/// <see cref="EvaluateText"/> checks the text alone, for captions that claim to be safe while naming unsafe content.
/// </para>
/// <para>
/// <b>For Beginners:</b> This module checks if what an image shows matches what the text says about it. For example,
/// if someone labels an image "cute puppy" but the image shows something violent, a CLIP model sees that the picture
/// is closer to "graphic violence" than to "cute puppy" and the module reports the mismatch.
/// </para>
/// <para>
/// <b>References:</b>
/// - Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP, ICML 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning Transferable Visual Models From Natural Language Supervision",
    "https://arxiv.org/abs/2103.00020",
    Year = 2021,
    Authors = "Alec Radford, Jong Wook Kim, Chris Hallacy, et al.")]
public class TextImageAlignmentChecker<T> : TextSafetyModuleBase<T>
{
    private readonly double _mismatchThreshold;
    private readonly IMultimodalEmbedding<T>? _clip;

    // The CLIP logit scale is learned and clipped at 100 during training (Radford et al. 2021, section 2.5).
    private const double ClipLogitScale = 100.0;

    // Zero-shot prompts for the unsafe concepts, in the CLIP "a photo of ..." prompt style.
    private static readonly (string Prompt, SafetyCategory Category)[] UnsafeConcepts =
    {
        ("a photo of graphic violence, blood and gore", SafetyCategory.ViolenceGraphic),
        ("a sexually explicit photo containing nudity", SafetyCategory.SexualExplicit),
        ("a photo of hateful symbols and hate propaganda", SafetyCategory.HateSpeech),
    };

    // Keywords for visual content categories
    private static readonly Dictionary<string, string[]> VisualCategoryKeywords = new()
    {
        ["safe"] = new[] { "cute", "puppy", "kitten", "flower", "landscape", "sunset", "family",
            "baby", "garden", "nature", "beautiful", "peaceful", "happy", "smile" },
        ["violent"] = new[] { "blood", "weapon", "gun", "knife", "fight", "war", "explosion",
            "attack", "injury", "wound", "dead", "kill", "murder", "combat" },
        ["nsfw"] = new[] { "nude", "naked", "explicit", "sexual", "pornograph", "erotic",
            "adult content", "xxx", "nsfw" },
        ["hateful"] = new[] { "hate", "racist", "supremac", "nazi", "genocide", "ethnic cleansing",
            "slur", "bigot", "discriminat" }
    };

    /// <inheritdoc />
    public override string ModuleName => "TextImageAlignmentChecker";

    /// <inheritdoc />
    public override bool IsReady => true;

    /// <summary>
    /// Initializes a new text-image alignment checker.
    /// </summary>
    /// <param name="mismatchThreshold">
    /// Threshold for flagging misalignment (0-1): with CLIP, the zero-shot probability of an unsafe concept; without
    /// it, the heuristic mismatch score. Default: 0.5.
    /// </param>
    /// <param name="clipModel">
    /// A CLIP model used to compare the image with its caption, for example a <c>ClipNeuralNetwork</c> loaded from
    /// its ONNX image and text encoders. <c>null</c> uses the colour-heuristic fallback.
    /// </param>
    public TextImageAlignmentChecker(double mismatchThreshold = 0.5, IMultimodalEmbedding<T>? clipModel = null)
    {
        if (double.IsNaN(mismatchThreshold) || mismatchThreshold < 0 || mismatchThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(mismatchThreshold), "The mismatch threshold must be between 0 and 1.");
        _mismatchThreshold = mismatchThreshold;
        _clip = clipModel;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text)) return findings;

        string lower = text.ToLowerInvariant();

        // Detect text that explicitly claims safety while containing unsafe terms
        bool claimsSafe = ContainsAny(lower, new[] { "safe", "harmless", "innocent", "clean", "appropriate" });
        bool hasUnsafeContent = ContainsAny(lower, VisualCategoryKeywords["violent"]) ||
                                ContainsAny(lower, VisualCategoryKeywords["nsfw"]) ||
                                ContainsAny(lower, VisualCategoryKeywords["hateful"]);

        if (claimsSafe && hasUnsafeContent)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Manipulated,
                Severity = SafetySeverity.Medium,
                Confidence = 0.7,
                Description = "Text-image alignment concern: text claims content is safe but contains " +
                              "unsafe content descriptors. This may indicate an attempt to disguise " +
                              "harmful content with a safe-sounding description.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <summary>
    /// Evaluates alignment between text and an associated image.
    /// </summary>
    /// <param name="text">Text description or caption.</param>
    /// <param name="image">Associated image tensor.</param>
    /// <returns>Alignment findings.</returns>
    public IReadOnlyList<SafetyFinding> EvaluateAlignment(string text, Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        if (string.IsNullOrWhiteSpace(text)) return findings;

        if (image is null) throw new ArgumentNullException(nameof(image));
        string lower = text.ToLowerInvariant();

        // Extract text category signals
        var textCategories = ClassifyTextContent(lower);

        if (_clip is not null)
        {
            return EvaluateAlignmentWithClip(text, image, textCategories, _clip);
        }

        if (image.Length < 16) return findings;

        // Extract image visual signals
        var imageFeatures = ExtractImageFeatures(image, image._shape);

        // Check for misalignment between text claims and image content
        // Text says "safe" but image has unsafe visual features
        if (textCategories.ContainsKey("safe") && textCategories["safe"] > 0.5)
        {
            if (NumOps.GreaterThan(imageFeatures.SkinFraction, NumOps.FromDouble(0.4)) ||
                NumOps.GreaterThan(imageFeatures.RedDominance, NumOps.FromDouble(0.3)))
            {
                double mismatchScore = NumOps.ToDouble(
                    SafetyTensorOps<T>.MaxNaN(imageFeatures.SkinFraction, imageFeatures.RedDominance));
                if (mismatchScore >= _mismatchThreshold)
                {
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.Manipulated,
                        Severity = SafetySeverity.High,
                        Confidence = Math.Min(1.0, mismatchScore),
                        Description = $"Text-image misalignment (score: {mismatchScore:F3}). " +
                                      $"Text describes safe content but image features suggest otherwise " +
                                      $"(skin fraction: {NumOps.ToDouble(imageFeatures.SkinFraction):F3}, " +
                                      $"red dominance: {NumOps.ToDouble(imageFeatures.RedDominance):F3}).",
                        RecommendedAction = SafetyAction.Warn,
                        SourceModule = ModuleName
                    });
                }
            }
        }

        return findings;
    }

    // CLIP zero-shot: softmax over 100 x cos(image, prompt) for the caption and the unsafe-concept prompts.
    private IReadOnlyList<SafetyFinding> EvaluateAlignmentWithClip(
        string caption, Tensor<T> image, Dictionary<string, double> textCategories, IMultimodalEmbedding<T> clip)
    {
        var findings = new List<SafetyFinding>();
        int expected = 3 * clip.ImageSize * clip.ImageSize;
        var pixels = image.ToArray();
        if (pixels.Length != expected)
            throw new ArgumentException(
                $"The CLIP model expects a preprocessed 3 x {clip.ImageSize} x {clip.ImageSize} image ({expected} values); got {pixels.Length}.",
                nameof(image));

        // A caption that itself names unsafe content is not misaligned with an unsafe image.
        if (textCategories["violent"] > 0 || textCategories["nsfw"] > 0 || textCategories["hateful"] > 0) return findings;

        var imageData = new double[pixels.Length];
        for (int i = 0; i < pixels.Length; i++) imageData[i] = NumOps.ToDouble(pixels[i]);
        var imageEmbedding = clip.EncodeImage(imageData);

        var logits = new double[UnsafeConcepts.Length + 1];
        logits[0] = ClipLogitScale * NumOps.ToDouble(clip.ComputeSimilarity(imageEmbedding, clip.EncodeText(caption)));
        for (int c = 0; c < UnsafeConcepts.Length; c++)
            logits[c + 1] = ClipLogitScale * NumOps.ToDouble(clip.ComputeSimilarity(imageEmbedding, clip.EncodeText(UnsafeConcepts[c].Prompt)));

        double max = logits.Max();
        double total = 0;
        var probabilities = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = Math.Exp(logits[i] - max);
            total += probabilities[i];
        }

        for (int i = 0; i < probabilities.Length; i++) probabilities[i] /= total;

        int best = 0;
        for (int c = 1; c < UnsafeConcepts.Length; c++)
            if (probabilities[c + 1] > probabilities[best + 1]) best = c;
        double probability = probabilities[best + 1];
        if (probability >= _mismatchThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Manipulated,
                Severity = SafetySeverity.High,
                Confidence = probability,
                Description = $"Text-image misalignment: CLIP matches the image to '{UnsafeConcepts[best].Prompt}' " +
                              $"({UnsafeConcepts[best].Category}) with probability {probability:F3}, against {probabilities[0]:F3} for its caption.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null)
            throw new ArgumentNullException(nameof(content));

        return Array.Empty<SafetyFinding>();
    }

    private static Dictionary<string, double> ClassifyTextContent(string text)
    {
        var scores = new Dictionary<string, double>();

        foreach (var kvp in VisualCategoryKeywords)
        {
            int matches = 0;
            foreach (string keyword in kvp.Value)
            {
                if (ContainsAtWordStart(text, keyword)) matches++;
            }
            scores[kvp.Key] = Math.Min(1.0, (double)matches / 3);
        }

        return scores;
    }

    /// <summary>
    /// Whether <paramref name="keyword"/> occurs where a word starts. A prefix still counts ("bloody" for "blood"), but
    /// a match inside a word does not: "toward" holds "war" and is not violent.
    /// </summary>
    private static bool ContainsAtWordStart(string text, string keyword)
    {
        for (int index = text.IndexOf(keyword, StringComparison.Ordinal); index >= 0;
             index = text.IndexOf(keyword, index + 1, StringComparison.Ordinal))
        {
            if (index == 0 || !char.IsLetterOrDigit(text[index - 1])) return true;
        }

        return false;
    }

    private ImageVisualFeatures ExtractImageFeatures(Tensor<T> image, int[] shape)
    {
        int totalPixels = image.Length;
        int channels = shape.Length >= 3 ? shape[0] : 1;
        int pixelsPerChannel = channels > 0 ? totalPixels / channels : totalPixels;

        // Sample pixels i = 0, step, 2 * step, ... below pixelsPerChannel
        int step = Math.Max(1, pixelsPerChannel / 1000);
        int analyzed = (pixelsPerChannel + step - 1) / step;
        if (analyzed <= 0)
            return new ImageVisualFeatures { SkinFraction = NumOps.Zero, RedDominance = NumOps.Zero, DarkFraction = NumOps.Zero };

        using var noGrad = new NoGradScope<T>();
        var engine = Engine;
        var flat = engine.Reshape(image.Contiguous(), new[] { totalPixels });
        var sampleIndex = engine.TensorArange<int>(0, analyzed * step, step);
        var r = engine.TensorTake(flat, sampleIndex);
        Tensor<T> g = r, b = r;
        if (channels >= 3)
        {
            // i + pixelsPerChannel and i + 2 * pixelsPerChannel stay inside the buffer when there are 3+ channels.
            g = engine.TensorTake(flat, engine.TensorAddScalar(sampleIndex, pixelsPerChannel));
            b = engine.TensorTake(flat, engine.TensorAddScalar(sampleIndex, 2 * pixelsPerChannel));
        }

        var scaled = SafetyTensorOps<T>.And(SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.Le(r, 1.0), SafetyTensorOps<T>.Le(g, 1.0)), SafetyTensorOps<T>.Le(b, 1.0));
        T full = NumOps.FromDouble(255);
        r = SafetyTensorOps<T>.Where(scaled, engine.TensorMultiplyScalar(r, full), r);
        g = SafetyTensorOps<T>.Where(scaled, engine.TensorMultiplyScalar(g, full), g);
        b = SafetyTensorOps<T>.Where(scaled, engine.TensorMultiplyScalar(b, full), b);

        // Skin detection (simplified)
        var skin = SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.And(
                SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(r, 95), SafetyTensorOps<T>.Gt(g, 40)),
                SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(b, 20), engine.TensorGreaterThan(r, g))),
            SafetyTensorOps<T>.And(engine.TensorGreaterThan(r, b),
                SafetyTensorOps<T>.Gt(engine.TensorAbs(engine.TensorSubtract(r, g)), 15)));

        // Red dominance
        T onePointFive = NumOps.FromDouble(1.5);
        var red = SafetyTensorOps<T>.And(SafetyTensorOps<T>.Gt(r, 150), SafetyTensorOps<T>.And(
            engine.TensorGreaterThan(r, engine.TensorMultiplyScalar(g, onePointFive)),
            engine.TensorGreaterThan(r, engine.TensorMultiplyScalar(b, onePointFive))));

        // Dark regions
        var dark = SafetyTensorOps<T>.And(SafetyTensorOps<T>.And(
            SafetyTensorOps<T>.Lt(r, 50), SafetyTensorOps<T>.Lt(g, 50)), SafetyTensorOps<T>.Lt(b, 50));

        T count = NumOps.FromDouble(analyzed);
        return new ImageVisualFeatures
        {
            SkinFraction = NumOps.Divide(engine.TensorSum(skin), count),
            RedDominance = NumOps.Divide(engine.TensorSum(red), count),
            DarkFraction = NumOps.Divide(engine.TensorSum(dark), count)
        };
    }

    private static bool ContainsAny(string text, string[] terms)
    {
        foreach (string term in terms)
        {
            if (text.Contains(term)) return true;
        }
        return false;
    }

    private struct ImageVisualFeatures
    {
        public T SkinFraction;
        public T RedDominance;
        public T DarkFraction;
    }

    /// <summary>Preserves the original constructor signature using the default configuration.</summary>
    public TextImageAlignmentChecker(double mismatchThreshold) : this(mismatchThreshold, null) { }

}
