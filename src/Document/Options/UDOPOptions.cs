using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the UDOP document model.
/// </summary>
public class UDOPOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="UDOPOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public UDOPOptions()
    {
        NumClasses = 16;
        ImageSize = 224;
        MaxSequenceLength = 2048;
        HiddenDim = 1024;
        NumEncoderLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 16;
        VocabSize = 50000;
    }

    /// <summary>
    /// Initial learning rate. Defaults to the paper's 5e-5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// UDOP (Tang et al., arXiv:2212.02623 S4.1) trains with "learning rate 5e-5, 1000 warmup
    /// steps, batch size 512, weight decay of 1e-2, beta1 = 0.9, and beta2 = 0.98". The model
    /// built its optimizer with no options at all, so it ran at Adam's own 1e-3 default -- twenty
    /// times the paper's rate, with beta2 at 0.999 and no weight decay.
    /// </para>
    /// <para>
    /// The paper's 1000-step warmup is deliberately NOT applied by default. It is calibrated to
    /// batch-512 pretraining, and over a short run it would hold the rate near zero for the whole
    /// run; callers reproducing the paper's schedule can attach a warmup scheduler explicitly.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Too big and
    /// training gets worse the longer it runs.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 5e-5;


    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
