namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Financial PPO (Proximal Policy Optimization) trading agent.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PPO is a robust policy gradient algorithm that prevents
/// large policy updates, leading to stable training. These options extend the base
/// trading agent options with PPO-specific parameters.
/// </para>
/// </remarks>
public class FinancialPPOAgentOptions<T> : TradingAgentOptions<T>
{
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialPPOAgentOptions()
    {
        // Schulman et al. 2017, section 6.1: "a fully-connected MLP with two hidden layers of 64 units, and
        // tanh nonlinearities". The agent applies tanh; this is the paper's width.
        HiddenLayers = new[] { 64, 64 };
    }

    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public FinancialPPOAgentOptions(FinancialPPOAgentOptions<T> other) : base(other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        NumEpochs = other.NumEpochs;
        NumMiniBatches = other.NumMiniBatches;
    }

    /// <summary>
    /// Number of optimization epochs per batch of data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many times to iterate over each batch of experiences.
    /// More epochs extract more learning but risk overfitting to the batch.
    /// </para>
    /// </remarks>
    public int NumEpochs { get; set; } = 4;

    /// <summary>
    /// Number of mini-batches for each update.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Splitting data into mini-batches adds noise that can
    /// help generalization. More mini-batches = noisier but more diverse updates.
    /// </para>
    /// </remarks>
    public int NumMiniBatches { get; set; } = 4;
}
