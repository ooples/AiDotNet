using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the LayoutLMv2 document model.
/// </summary>
public class LayoutLMv2Options : DocumentNeuralNetworkOptions
{

    /// <summary>
    /// Gets or sets the learning rate. Default 2e-5 — LayoutLMv2 Appendix B trains with Adam at this rate.
    /// It is a FINE-TUNING rate for an already-pretrained backbone, so training a randomly initialized model
    /// from scratch needs a larger one. Previously hardcoded in CreatePaperDefaultOptimizer, which left callers
    /// no way to change it without supplying an entire optimizer. (#1789)
    /// </summary>
    public double LearningRate { get; set; } = 2e-5;

    /// <summary>
    /// Gets or sets the decoupled weight decay. Default 1e-2, per LayoutLMv2 Appendix B. Previously hardcoded.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-2;
}
