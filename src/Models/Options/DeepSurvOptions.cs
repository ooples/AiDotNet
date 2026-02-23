namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DeepSurv survival analysis model.
/// </summary>
/// <remarks>
/// <para>
/// DeepSurv extends the Cox Proportional Hazards model using a deep neural network
/// to learn the relationship between covariates and survival outcomes. It optimizes
/// the negative partial log-likelihood of the Cox model.
/// </para>
/// <para>
/// <b>For Beginners:</b> Survival analysis is used when you want to predict "time until
/// an event happens" - like:
/// - How long until a machine fails?
/// - How long until a customer cancels their subscription?
/// - How long until a patient experiences disease recurrence?
///
/// What makes survival analysis special is that some observations are "censored" -
/// meaning the event hasn't happened yet when the study ends. For example, if you're
/// studying customer churn and a customer is still subscribed when the study ends,
/// you know they survived at least that long, but you don't know when (or if) they'll churn.
///
/// DeepSurv uses a neural network to learn complex patterns in your data while properly
/// handling this censoring. It outputs a "risk score" - higher values mean higher risk
/// of the event happening sooner.
/// </para>
/// </remarks>
public class DeepSurvOptions
{
    /// <summary>
    /// Gets or sets the number of hidden layers.
    /// </summary>
    /// <value>Default is 2.</value>
    public int NumHiddenLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of neurons in each hidden layer.
    /// </summary>
    /// <value>Default is 32.</value>
    public int HiddenLayerSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the learning rate for optimization.
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Default is 100.</value>
    public int Epochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <value>Default is 32.</value>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the L2 regularization strength.
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double L2Regularization { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the activation function type.
    /// </summary>
    /// <value>Default is SELU.</value>
    public DeepSurvActivation Activation { get; set; } = DeepSurvActivation.SELU;

    /// <summary>
    /// Gets or sets whether to use batch normalization.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseBatchNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets the patience for early stopping.
    /// </summary>
    /// <value>Default is 10. Set to null to disable early stopping.</value>
    public int? EarlyStoppingPatience { get; set; } = 10;
}

/// <summary>
/// Activation functions for DeepSurv.
/// </summary>
public enum DeepSurvActivation
{
    /// <summary>
    /// Rectified Linear Unit: max(0, x).
    /// </summary>
    ReLU,

    /// <summary>
    /// Scaled Exponential Linear Unit - self-normalizing.
    /// </summary>
    SELU,

    /// <summary>
    /// Exponential Linear Unit.
    /// </summary>
    ELU,

    /// <summary>
    /// Hyperbolic tangent.
    /// </summary>
    Tanh,

    /// <summary>
    /// Leaky ReLU with small negative slope.
    /// </summary>
    LeakyReLU
}
