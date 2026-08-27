using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for <see cref="AiDotNet.ComputerVision.OCR.EndToEnd.ABCNet{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Defaults follow Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin and Liangwei Wang,
/// "ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network" (CVPR 2020 oral,
/// arXiv:2002.10200): a cubic Bezier boundary per edge, BezierAlign to 8x32, and a deliberately
/// lightweight recognition branch.
/// </para>
/// <para><b>For Beginners:</b> These settings control how ABCNet describes the shape of curved text
/// (with a smooth curve along its top and bottom edges), how large a straightened image it cuts out of
/// each piece of text, and how many characters it can tell apart.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class ABCNetOptions<T> : NeuralNetworkOptions
{
    /// <summary>Gets or sets the input image height. Default 256.</summary>
    public int InputHeight { get; set; } = 256;

    /// <summary>Gets or sets the input image width. Default 256.</summary>
    public int InputWidth { get; set; } = 256;

    /// <summary>Gets or sets the input channel count. Default 3 (RGB).</summary>
    public int InputChannels { get; set; } = 3;

    /// <summary>
    /// Gets or sets the shared feature width fed to both heads. Default 256, the paper's FPN width.
    /// </summary>
    public int FeatureChannels { get; set; } = 256;

    /// <summary>
    /// Gets or sets the total downsampling from image to detection feature map. Default 4.
    /// </summary>
    /// <remarks>
    /// Control points are regressed and BezierAlign samples in FEATURE-MAP coordinates, so this factor
    /// is what converts between the two. Getting it wrong scales every predicted curve by a constant and
    /// still produces plausible-looking curves, which is why it is explicit rather than implied.
    /// </remarks>
    public int FeatureStride { get; set; } = 4;

    /// <summary>
    /// Gets or sets the rectified height BezierAlign produces. Default 8, the paper's value.
    /// </summary>
    public int BezierSampleHeight { get; set; } = 8;

    /// <summary>
    /// Gets or sets the rectified width BezierAlign produces. Default 32, the paper's value.
    /// </summary>
    /// <remarks>
    /// The rectified width also bounds the CTC output length, so it caps how many characters an instance
    /// can hold. 32 columns comfortably covers the word-level instances the benchmarks contain.
    /// </remarks>
    public int BezierSampleWidth { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of character classes INCLUDING the CTC blank. Default 97.
    /// </summary>
    /// <remarks>
    /// The paper's alphabet is 96 symbols (digits, upper and lower case, punctuation); CTC needs one
    /// additional blank label, hence 97 rather than 96.
    /// </remarks>
    public int NumCharacterClasses { get; set; } = 97;

    /// <summary>
    /// Gets or sets the recurrent width of the recognition branch. Default 256.
    /// </summary>
    public int RecognitionHiddenSize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the score above which a feature-map position is treated as a text instance.
    /// Default 0.5.
    /// </summary>
    public double ConfidenceThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the cap on instances recognized per image. Default 100.
    /// </summary>
    /// <remarks>
    /// The recognition branch runs once per instance, so an unbounded count makes cost depend on the
    /// image. When the cap truncates, the retained instances are the highest scoring ones and the
    /// truncation is reported on the result rather than being silent.
    /// </remarks>
    public int MaxInstances { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate used when the model builds its own optimizer.
    /// </summary>
    /// <value>Defaults to 0.01, the rate the ABCNet paper trains with (arXiv:2002.10200).</value>
    /// <remarks>
    /// <para>
    /// The paper's 0.01 is an <b>SGD with momentum</b> rate, and the default optimizer is now that
    /// same SGD-with-momentum, so the published value and the optimizer it is applied to match.
    /// They previously did not: the model built an <c>AdamOptimizer</c>, whose step is roughly the
    /// learning rate itself, making 0.01 about two orders of magnitude too large and driving the
    /// shared backbone's activations negative until every input produced the same detection map.
    /// Lower this to roughly 1e-4 if you inject an Adam-family optimizer instead.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Momentum coefficient for the default SGD optimizer.
    /// </summary>
    /// <remarks>
    /// The paper trains with SGD at momentum 0.9. Exposed for the same reason the learning rate is:
    /// a published default is only useful if a caller can retune it without having to rebuild the
    /// whole optimizer.
    /// </remarks>
    public double Momentum { get; set; } = 0.9;

    /// <summary>
    /// Validates the configuration, throwing on values that cannot describe a working model.
    /// </summary>
    public void Validate()
    {
        if (InputHeight <= 0) throw new ArgumentOutOfRangeException(nameof(InputHeight), InputHeight, "InputHeight must be positive.");
        if (InputWidth <= 0) throw new ArgumentOutOfRangeException(nameof(InputWidth), InputWidth, "InputWidth must be positive.");
        if (InputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(InputChannels), InputChannels, "InputChannels must be positive.");
        if (FeatureChannels <= 0) throw new ArgumentOutOfRangeException(nameof(FeatureChannels), FeatureChannels, "FeatureChannels must be positive.");
        if (FeatureStride <= 0) throw new ArgumentOutOfRangeException(nameof(FeatureStride), FeatureStride, "FeatureStride must be positive.");
        if (BezierSampleHeight <= 0) throw new ArgumentOutOfRangeException(nameof(BezierSampleHeight), BezierSampleHeight, "BezierSampleHeight must be positive.");
        if (BezierSampleWidth <= 0) throw new ArgumentOutOfRangeException(nameof(BezierSampleWidth), BezierSampleWidth, "BezierSampleWidth must be positive.");
        if (RecognitionHiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(RecognitionHiddenSize), RecognitionHiddenSize, "RecognitionHiddenSize must be positive.");
        if (MaxInstances <= 0) throw new ArgumentOutOfRangeException(nameof(MaxInstances), MaxInstances, "MaxInstances must be positive.");

        // ConfidenceThreshold is a probability compared against a score at ABCNet.cs:541
        // (`if (score < _options.ConfidenceThreshold) continue;`). Above 1.0 that test is true at
        // every position, so detection returns zero instances on every image and the model looks
        // trained-but-useless rather than misconfigured; below 0.0 every position survives. It was
        // the one numeric option here that nothing bounded, and the only one that fails silently.
        if (ConfidenceThreshold < 0.0 || ConfidenceThreshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(ConfidenceThreshold), ConfidenceThreshold,
                "ConfidenceThreshold must be within [0, 1]. Above 1 no detection can ever pass it; "
                + "below 0 every position passes.");
        }

        if (LearningRate <= 0.0 || double.IsNaN(LearningRate) || double.IsInfinity(LearningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate,
                "LearningRate must be a positive, finite number.");
        }

        if (Momentum < 0.0 || Momentum >= 1.0 || double.IsNaN(Momentum))
        {
            throw new ArgumentOutOfRangeException(nameof(Momentum), Momentum,
                "Momentum must be finite and within [0, 1).");
        }

        // WHAT THIS CHECK ACTUALLY ENFORCES: the alphabet must hold at least one real symbol plus
        // the CTC blank. Nothing more.
        //
        // The comment that used to sit here claimed the rectified width had to "leave room for" the
        // alphabet, and no such relation is implemented -- nor should it be, because the reasoning
        // was inverted. CTC's time-step count bounds the length of a decoded LABEL SEQUENCE, not
        // the SIZE of the alphabet those labels are drawn from: BezierSampleWidth time steps can
        // emit at most that many labels whether the alphabet holds 97 characters or 97,000. The
        // genuine width constraint is about the longest word to be recognized, which is a property
        // of the data rather than of this configuration, so it is not checkable here.
        if (NumCharacterClasses < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(NumCharacterClasses), NumCharacterClasses,
                "NumCharacterClasses must be at least 2 — one symbol plus the CTC blank.");
        }

        if (InputHeight % FeatureStride != 0 || InputWidth % FeatureStride != 0)
        {
            throw new ArgumentException(
                $"InputHeight ({InputHeight}) and InputWidth ({InputWidth}) must both be divisible by "
                + $"FeatureStride ({FeatureStride}); otherwise feature-map coordinates do not map back "
                + "onto whole image pixels and every predicted curve is offset.");
        }
    }
}
