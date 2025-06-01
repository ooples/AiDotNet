namespace AiDotNet.Enums;

/// <summary>
/// Defines strategies for automated neural architecture search (NAS).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Neural Architecture Search automatically designs neural networks for you. 
/// Instead of manually choosing layers and connections, these strategies help find the best 
/// architecture for your specific problem.
/// </para>
/// </remarks>
public enum NeuralArchitectureSearchStrategy
{
    /// <summary>
    /// No architecture search (manual design).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> You design the network yourself - traditional approach where you 
    /// decide how many layers, neurons, and connections to use.
    /// </remarks>
    None,

    /// <summary>
    /// Random search through architecture space.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tries random network designs and keeps the best ones - simple but 
    /// can be surprisingly effective given enough time.
    /// </remarks>
    RandomSearch,

    /// <summary>
    /// Grid search through predefined options.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Systematically tries all combinations from a list of options - like 
    /// trying every item on a menu to find your favorite.
    /// </remarks>
    GridSearch,

    /// <summary>
    /// Evolutionary algorithms for architecture evolution.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses natural selection principles - good architectures "reproduce" 
    /// and "mutate" to create better offspring architectures.
    /// </remarks>
    Evolutionary,

    /// <summary>
    /// Reinforcement learning-based search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Trains an AI agent that learns to design good networks - the agent 
    /// gets rewards for creating architectures that perform well.
    /// </remarks>
    ReinforcementLearning,

    /// <summary>
    /// Gradient-based architecture search (DARTS).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses calculus to efficiently search for architectures - much faster 
    /// than trying random designs but more complex to implement.
    /// </remarks>
    GradientBased,

    /// <summary>
    /// Bayesian optimization for architecture search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses probability to intelligently guess which architectures to try 
    /// next - learns from previous attempts to make better choices.
    /// </remarks>
    BayesianOptimization,

    /// <summary>
    /// Weight sharing approach (ENAS).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Shares weights between different architectures to speed up search - 
    /// like reusing building blocks instead of starting from scratch each time.
    /// </remarks>
    WeightSharing,

    /// <summary>
    /// One-shot architecture search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Trains one large "super network" that contains all possible 
    /// architectures, then extracts the best sub-network.
    /// </remarks>
    OneShot,

    /// <summary>
    /// Progressive search starting simple.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Starts with simple networks and gradually makes them more complex - 
    /// like learning to walk before trying to run.
    /// </remarks>
    Progressive,

    /// <summary>
    /// Differentiable architecture search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Makes the architecture itself trainable using the same methods used 
    /// to train weights - very efficient but mathematically complex.
    /// </remarks>
    Differentiable,

    /// <summary>
    /// Early stopping based search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Quickly evaluates architectures by training them partially - saves 
    /// time by not fully training bad architectures.
    /// </remarks>
    EarlyStopping,

    /// <summary>
    /// Multi-objective architecture search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Searches for architectures that balance multiple goals - like finding 
    /// networks that are both accurate and fast.
    /// </remarks>
    MultiObjective,

    /// <summary>
    /// Hardware-aware architecture search.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Designs networks specifically for your hardware - ensures the network 
    /// will run efficiently on your target device.
    /// </remarks>
    HardwareAware,

    /// <summary>
    /// Transfer learning from previous searches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses knowledge from previous architecture searches to speed up new 
    /// searches - like using experience from past projects.
    /// </remarks>
    TransferLearning,

    /// <summary>
    /// Custom search strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to implement your own architecture search method for 
    /// special requirements.
    /// </remarks>
    Custom
}