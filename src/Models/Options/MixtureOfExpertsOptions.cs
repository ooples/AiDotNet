namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Mixture-of-Experts (MoE) neural network model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// Mixture-of-Experts is a neural network architecture that employs multiple specialist networks (experts)
/// and a gating mechanism to route inputs to the most appropriate experts. This approach enables:
/// - Increased model capacity without proportional compute cost (sparse activation)
/// - Specialization of different experts on different aspects of the problem
/// - Improved scalability for large-scale problems
/// </para>
/// <para><b>For Beginners:</b> Mixture-of-Experts (MoE) is like having a team of specialists rather than one generalist.
///
/// Imagine you're running a hospital:
/// - Instead of one doctor handling everything, you have specialists (cardiologist, neurologist, etc.)
/// - A triage system (gating network) decides which specialist(s) should see each patient
/// - Each specialist only handles cases they're best suited for
///
/// In a MoE neural network:
/// - Multiple "expert" networks specialize in different patterns in your data
/// - A "gating network" learns to route each input to the best expert(s)
/// - Only a few experts process each input (sparse activation), making it efficient
/// - The final prediction combines the outputs from the selected experts
///
/// This class lets you configure:
/// - How many expert networks to use
/// - How many experts process each input (Top-K)
/// - Dimensions of the expert networks
/// - Whether to use load balancing to ensure all experts are utilized
/// </para>
/// </remarks>
public class MixtureOfExpertsOptions<T> : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the number of expert networks in the mixture.
    /// </summary>
    /// <value>The number of experts, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many specialist networks the model contains. More experts allow for
    /// greater specialization and model capacity but increase computational and memory requirements. Research
    /// suggests that 4-16 experts provides a good balance for most applications. The number of experts should
    /// be chosen based on the complexity of the problem domain and available computational resources.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many specialist networks you want in your model.
    ///
    /// The default value of 4 means you'll have 4 different expert networks:
    /// - Each expert can learn to specialize in different types of patterns
    /// - Think of it like having 4 specialists on your team
    ///
    /// You might want more experts if:
    /// - Your problem has many distinct types of patterns or sub-tasks
    /// - You have a large dataset and lots of computing power
    /// - You want maximum model capacity
    ///
    /// You might want fewer experts if:
    /// - Your problem is relatively simple
    /// - You have limited computing resources or data
    /// - You want faster training and inference
    ///
    /// Typical values range from 2-16 experts. Start with 4-8 for most problems.
    /// </para>
    /// </remarks>
    public int NumExperts { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of experts to activate for each input (Top-K routing).
    /// </summary>
    /// <value>The Top-K value, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many experts process each input in a sparse activation pattern. Only
    /// the K experts with the highest routing probabilities are activated for each input, while others are
    /// skipped. This sparse activation is key to the efficiency of MoE models. The value should typically
    /// be much smaller than the total number of experts. Common choices are 1-2 for efficiency or 2-4 for
    /// better quality. TopK must be less than or equal to NumExperts.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how many experts actually process each input.
    ///
    /// The default value of 2 means:
    /// - For each input, only the 2 best-suited experts are activated
    /// - The other experts are skipped (saving computation)
    /// - The gating network learns which experts are best for each input
    ///
    /// Think of it like consulting specialists:
    /// - You don't need to see all 4 specialists for every case
    /// - The triage system picks the 2 most relevant ones
    /// - You get expert opinions while saving time
    ///
    /// TopK = 1: Fastest, each input goes to only one expert
    /// TopK = 2: Good balance (recommended default)
    /// TopK = 4+: More experts per input, higher quality but slower
    ///
    /// The value must be at least 1 and at most equal to NumExperts.
    /// </para>
    /// </remarks>
    public int TopK { get; set; } = 2;

    /// <summary>
    /// Gets or sets the input dimension for each expert network.
    /// </summary>
    /// <value>The input dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// This parameter specifies the dimensionality of the input that each expert receives. For MoE layers
    /// embedded within larger networks, this typically matches the hidden layer size of the network. The
    /// input dimension determines the size of the expert networks and should match the output dimension of
    /// the previous layer in the network architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the input data that goes into each expert.
    ///
    /// The default value of 128 means:
    /// - Each expert receives 128-dimensional input vectors
    /// - This is a common size for neural network hidden layers
    ///
    /// This value should match:
    /// - The output size of the previous layer in your network, OR
    /// - The size of your input features if MoE is the first layer
    ///
    /// Larger dimensions:
    /// - Can capture more complex patterns
    /// - Require more memory and computation
    /// - Need more training data
    ///
    /// Typical values range from 64 to 512 depending on your problem complexity.
    /// </para>
    /// </remarks>
    public int InputDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the output dimension for each expert network.
    /// </summary>
    /// <value>The output dimension, defaulting to 128.</value>
    /// <remarks>
    /// <para>
    /// This parameter specifies the dimensionality of the output produced by each expert. In many
    /// architectures, this matches the input dimension to allow for residual connections and easier
    /// stacking of MoE layers. However, it can differ if the MoE layer is designed to change the
    /// dimensionality of the representations.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the output from each expert.
    ///
    /// The default value of 128 means:
    /// - Each expert produces 128-dimensional output vectors
    /// - Often set equal to InputDim for symmetry
    ///
    /// Common patterns:
    /// - Same as InputDim (128→128): Maintains dimension, good for stacking multiple MoE layers
    /// - Different size: If you want to compress (128→64) or expand (64→128) the representation
    ///
    /// This value should match:
    /// - The input size expected by the next layer in your network, OR
    /// - The final output size if MoE is the last hidden layer
    ///
    /// For most applications, keeping InputDim == OutputDim (both 128) works well.
    /// </para>
    /// </remarks>
    public int OutputDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the hidden layer expansion factor for each expert's feed-forward network.
    /// </summary>
    /// <value>The expansion factor, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the size of the hidden layer within each expert relative to the input
    /// dimension. Following the Transformer architecture convention, experts typically use a feed-forward
    /// network with a hidden layer that is 4x the input dimension. The hidden layer size equals
    /// InputDim * HiddenExpansion. Larger values increase expert capacity but also increase computational cost.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much each expert can expand internally for processing.
    ///
    /// The default value of 4 means:
    /// - Each expert has a hidden layer that's 4 times the input size
    /// - With InputDim=128, the hidden layer is 512 neurons
    /// - This follows the proven Transformer architecture design
    ///
    /// How it works:
    /// - Input (128) → Hidden Layer (512) → Output (128)
    /// - The expansion allows experts to learn more complex transformations
    /// - Then compression back to output size
    ///
    /// Typical values:
    /// - 4: Standard choice (recommended, from Transformer research)
    /// - 2-3: More efficient, less capacity
    /// - 6-8: More capacity, higher cost
    ///
    /// Most users should keep the default value of 4.
    /// </para>
    /// </remarks>
    public int HiddenExpansion { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to enable auxiliary load balancing loss.
    /// </summary>
    /// <value>True to enable load balancing, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the model adds an auxiliary loss term that encourages balanced utilization of all
    /// experts. Without load balancing, the model may learn to use only a subset of experts, wasting
    /// capacity. The load balancing loss measures the deviation from uniform expert usage and penalizes
    /// imbalanced routing decisions. This is crucial for effective MoE training, especially with larger
    /// numbers of experts. The strength of this loss is controlled by LoadBalancingWeight.
    /// </para>
    /// <para><b>For Beginners:</b> This prevents some experts from being ignored during training.
    ///
    /// The default value of true means load balancing is enabled:
    /// - Without this, the model might only use 1-2 experts and ignore the rest
    /// - Load balancing encourages all experts to be utilized
    /// - Think of it like ensuring all team members contribute, not just a few favorites
    ///
    /// Why this matters:
    /// - You're paying the cost (memory, parameters) for all experts
    /// - If only some are used, you're wasting resources
    /// - Balanced usage leads to better model capacity and performance
    ///
    /// When to disable (UseLoadBalancing = false):
    /// - Only for experimentation or debugging
    /// - Generally, you should keep this enabled
    ///
    /// Recommended: Keep the default value of true for nearly all applications.
    /// </para>
    /// </remarks>
    public bool UseLoadBalancing { get; set; } = true;

    /// <summary>
    /// Gets or sets the weight of the auxiliary load balancing loss.
    /// </summary>
    /// <value>The load balancing loss weight, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the strength of the load balancing regularization. The total loss becomes:
    /// TotalLoss = PrimaryLoss + (LoadBalancingWeight * LoadBalancingLoss). A value of 0.01 provides gentle
    /// encouragement toward balanced expert usage without overwhelming the primary task loss. Values that are
    /// too high can hurt task performance by forcing artificial balance, while values too low may not
    /// effectively prevent expert collapse. Only used when UseLoadBalancing is true.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly the model tries to balance expert usage.
    ///
    /// The default value of 0.01 means:
    /// - Load balancing contributes 1% as much as the main task loss
    /// - It gently encourages balance without dominating training
    /// - This is a research-proven default from the Switch Transformer paper
    ///
    /// Think of it like priorities:
    /// - Main task (99% weight): Make accurate predictions
    /// - Load balancing (1% weight): Keep expert usage balanced
    ///
    /// Typical values:
    /// - 0.001: Very gentle balancing (for when balance isn't critical)
    /// - 0.01: Standard choice (recommended default)
    /// - 0.1: Strong balancing (if you notice severe imbalance)
    ///
    /// If you see in your logs that only 1-2 experts are being used, try increasing this to 0.05 or 0.1.
    /// If training seems unstable, try decreasing to 0.001.
    ///
    /// For most applications, the default 0.01 works well.
    /// </para>
    /// </remarks>
    public double LoadBalancingWeight { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the random seed for expert initialization.
    /// </summary>
    /// <value>The random seed, defaulting to null for non-deterministic initialization.</value>
    /// <remarks>
    /// <para>
    /// When set, this seed ensures deterministic initialization of expert networks and the gating network,
    /// making training reproducible. When null, initialization uses a non-deterministic seed, leading to
    /// different results across runs. Reproducibility is important for research, debugging, and production
    /// systems where consistent behavior is required.
    /// </para>
    /// <para><b>For Beginners:</b> This controls whether training produces the same results every time.
    ///
    /// The default value of null means:
    /// - Each training run will produce slightly different results
    /// - Initial weights are randomly chosen each time
    ///
    /// Set a specific number (e.g., 42) for reproducibility:
    /// - Same seed = same initial weights = same training trajectory
    /// - Useful for debugging, comparing changes, or research
    ///
    /// Example usage:
    /// - RandomSeed = null: Different results each time (fine for production)
    /// - RandomSeed = 42: Same results each time (good for debugging/research)
    ///
    /// Note: This only controls initialization. Other factors like data shuffling may still introduce variability.
    /// </para>
    /// </remarks>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Validates that all option values are within acceptable ranges.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option value is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method checks all configuration parameters to ensure they meet the requirements for a valid
    /// MoE model. It verifies dimensional constraints, expert counts, routing parameters, and other settings.
    /// Calling this method before model construction helps catch configuration errors early.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks that all your settings make sense together.
    ///
    /// It validates things like:
    /// - You have at least 1 expert
    /// - TopK isn't larger than the number of experts
    /// - Dimensions are positive numbers
    /// - Load balancing weight is non-negative
    ///
    /// This catches mistakes before they cause problems during training.
    /// You don't need to call this manually - it's called automatically when creating the model.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (NumExperts < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(NumExperts),
                $"NumExperts must be at least 1, but got {NumExperts}.");
        }

        if (TopK < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(TopK),
                $"TopK must be at least 1, but got {TopK}.");
        }

        if (TopK > NumExperts)
        {
            throw new ArgumentOutOfRangeException(nameof(TopK),
                $"TopK ({TopK}) cannot be greater than NumExperts ({NumExperts}).");
        }

        if (InputDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(InputDim),
                $"InputDim must be at least 1, but got {InputDim}.");
        }

        if (OutputDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(OutputDim),
                $"OutputDim must be at least 1, but got {OutputDim}.");
        }

        if (HiddenExpansion < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(HiddenExpansion),
                $"HiddenExpansion must be at least 1, but got {HiddenExpansion}.");
        }

        if (LoadBalancingWeight < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(LoadBalancingWeight),
                $"LoadBalancingWeight must be non-negative, but got {LoadBalancingWeight}.");
        }
    }
}
