using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DCRNN (Diffusion Convolutional Recurrent Neural Network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// DCRNN combines diffusion convolution with sequence-to-sequence architecture
/// for spatial-temporal forecasting on graph-structured data.
/// </para>
/// <para><b>For Beginners:</b> DCRNN was specifically designed for traffic forecasting
/// by combining two powerful ideas:
///
/// <b>The Key Insight:</b>
/// Traffic flow on road networks can be modeled as a diffusion process - like how
/// congestion spreads through a network. DCRNN captures this with diffusion convolution
/// while using an encoder-decoder architecture for multi-step prediction.
///
/// <b>What Problems Does DCRNN Solve?</b>
/// - Traffic speed/flow prediction on road networks
/// - Air quality forecasting across sensor networks
/// - Subway ridership prediction
/// - Any spatial-temporal forecasting where diffusion dynamics matter
///
/// <b>How DCRNN Works:</b>
/// 1. <b>Diffusion Convolution:</b> Models spatial dependencies as bidirectional random walks
/// 2. <b>Diffusion GRU:</b> Replaces matrix multiplications in GRU with diffusion convolution
/// 3. <b>Encoder-Decoder:</b> Encoder captures history, decoder generates predictions
/// 4. <b>Scheduled Sampling:</b> Gradually transitions from ground truth to predictions during training
///
/// <b>DCRNN Architecture:</b>
/// - Encoder: Stacked DCGRU layers that encode input sequence
/// - Decoder: Stacked DCGRU layers that generate output sequence
/// - Diffusion: D_O^(K) + D_I^(K) bidirectional diffusion matrices
/// - Output: Linear projection to forecast values
///
/// <b>Key Benefits:</b>
/// - Captures spatial dependencies through diffusion process (not just adjacency)
/// - Multi-step prediction through encoder-decoder architecture
/// - Scheduled sampling prevents exposure bias during training
/// - Bidirectional diffusion captures both upstream and downstream effects
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.
/// https://arxiv.org/abs/1707.01926
/// </para>
/// </remarks>
public class DCRNNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DCRNNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default DCRNN configuration optimized for
    /// traffic forecasting with diffusion convolution and encoder-decoder architecture.
    /// </para>
    /// </remarks>
    public DCRNNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DCRNNOptions(DCRNNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumNodes = other.NumNodes;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        DiffusionSteps = other.DiffusionSteps;
        DropoutRate = other.DropoutRate;
        UseScheduledSampling = other.UseScheduledSampling;
        ScheduledSamplingDecaySteps = other.ScheduledSamplingDecaySteps;
        MinTeacherForcingRatio = other.MinTeacherForcingRatio;
        MaxDiffusionStep = other.MaxDiffusionStep;
        FilterType = other.FilterType;
        NumSamples = other.NumSamples;
    }

    /// <summary>
    /// Gets or sets the sequence length (input time steps).
    /// </summary>
    /// <value>The sequence length, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps to use as input.
    /// For traffic data with 5-minute intervals, 12 steps = 1 hour of history.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 12;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>The forecast horizon, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future time steps to predict.
    /// The decoder generates this many predictions autoregressively.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of nodes in the graph.
    /// </summary>
    /// <value>The number of nodes, defaulting to 207 (METR-LA dataset size).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many locations/sensors in the network.
    /// Each node is a traffic sensor on a road network.
    /// </para>
    /// </remarks>
    public int NumNodes { get; set; } = 207;

    /// <summary>
    /// Gets or sets the number of input features per node.
    /// </summary>
    /// <value>The number of features, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many measurements at each node per time step.
    /// Common features: (speed, time_of_day) or just speed alone.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 2;

    /// <summary>
    /// Gets or sets the hidden dimension for the DCGRU cells.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal state size of each diffusion GRU cell.
    /// Larger values capture more complex patterns but need more computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of encoder DCGRU layers.
    /// </summary>
    /// <value>The number of encoder layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many stacked diffusion GRU layers in the encoder.
    /// The encoder processes the input sequence to create a context representation.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder DCGRU layers.
    /// </summary>
    /// <value>The number of decoder layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many stacked diffusion GRU layers in the decoder.
    /// The decoder uses the encoder's context to generate predictions step-by-step.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of diffusion steps (K).
    /// </summary>
    /// <value>The diffusion steps, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many hops of diffusion to compute.
    /// K=2 means using the transition matrix up to the 2nd power.
    /// Higher K captures longer-range spatial dependencies.
    /// </para>
    /// </remarks>
    public int DiffusionSteps { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting by randomly dropping connections.
    /// DCRNN paper uses 0, but small dropout can help generalization.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use scheduled sampling during training.
    /// </summary>
    /// <value>True to use scheduled sampling; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scheduled sampling helps bridge the gap between
    /// training (where ground truth is available) and inference (where it isn't).
    /// It gradually reduces teacher forcing during training.
    /// </para>
    /// </remarks>
    public bool UseScheduledSampling { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of training steps for scheduled sampling decay.
    /// </summary>
    /// <value>The decay steps, defaulting to 2000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training steps it takes to transition
    /// from 100% teacher forcing to the minimum ratio. Uses inverse sigmoid decay.
    /// </para>
    /// </remarks>
    public int ScheduledSamplingDecaySteps { get; set; } = 2000;

    /// <summary>
    /// Gets or sets the minimum teacher forcing ratio.
    /// </summary>
    /// <value>The minimum teacher forcing ratio, defaulting to 0.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The final probability of using ground truth
    /// instead of predictions during decoder training. 0 means fully autoregressive.
    /// </para>
    /// </remarks>
    public double MinTeacherForcingRatio { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the maximum diffusion step power.
    /// </summary>
    /// <value>The maximum diffusion step, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The highest power of the transition matrix used.
    /// With max_step=2, we use: I, P, P^2 for diffusion.
    /// </para>
    /// </remarks>
    public int MaxDiffusionStep { get; set; } = 2;

    /// <summary>
    /// Gets or sets the type of filter for diffusion.
    /// </summary>
    /// <value>The filter type, defaulting to "dual_random_walk".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to construct the diffusion filters.
    /// - "dual_random_walk": Bidirectional (forward + backward) random walk
    /// - "laplacian": Use normalized Laplacian
    /// - "random_walk": Single direction only
    /// </para>
    /// </remarks>
    public string FilterType { get; set; } = "dual_random_walk";

    /// <summary>
    /// Gets or sets the number of samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For probabilistic forecasting with MC Dropout.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;
}
