using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Transformer neural network architecture, which is particularly effective for
/// sequence-based tasks like natural language processing.
/// </summary>
/// <remarks>
/// <para>
/// The Transformer architecture is a type of neural network design that uses self-attention mechanisms
/// instead of recurrence or convolution. This approach allows the model to weigh the importance of 
/// different parts of the input sequence when producing each part of the output sequence.
/// </para>
/// <para>
/// The key components of a Transformer include:
/// - Multi-head attention layers: Allow the model to focus on different parts of the input
/// - Feed-forward networks: Process the attended information
/// - Layer normalization: Stabilize the network during training
/// - Residual connections: Help information flow through the network
/// </para>
/// <para><b>For Beginners:</b> A Transformer is a modern type of neural network that excels at 
/// understanding sequences of data, like sentences or time series.
/// 
/// Think of it like reading a book:
/// - When you read a sentence, some words are more important than others for understanding the meaning
/// - A Transformer can "pay attention" to different words based on their importance
/// - It can look at the entire context at once, rather than reading one word at a time
/// 
/// For example, in the sentence "The animal didn't cross the street because it was too wide",
/// the Transformer can figure out that "it" refers to "the street" by paying attention to the
/// relationship between these words.
/// 
/// Transformers are behind many recent AI advances, including large language models like GPT and BERT.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new TransformerOptions { HiddenSize = 512, NumHeads = 8, NumLayers = 6 };
/// var model = new Transformer&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 128, 512 });
/// var output = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Attention Is All You Need", "https://arxiv.org/abs/1706.03762", Year = 2017, Authors = "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin")]
public class Transformer<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly TransformerOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets or sets whether auxiliary loss (attention regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention regularization aggregates auxiliary losses from all MultiHeadAttentionLayers in the network.
    /// This includes both entropy regularization and head diversity penalties.
    /// </para>
    /// <para><b>For Beginners:</b> This controls attention quality across the entire Transformer.
    ///
    /// When enabled, the Transformer:
    /// - Collects regularization from all attention layers
    /// - Prevents attention collapse across the network
    /// - Encourages diverse attention patterns at all levels
    ///
    /// This is especially important for:
    /// - Deep transformers (many layers)
    /// - Models with many attention heads
    /// - Tasks requiring robust attention patterns
    ///
    /// The auxiliary loss helps maintain attention quality throughout training.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the attention regularization auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much network-level attention regularization contributes to the total loss.
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much to encourage good attention throughout the network.
    ///
    /// Common values:
    /// - 0.005 (default): Balanced network-level regularization
    /// - 0.001-0.003: Light regularization
    /// - 0.008-0.01: Strong regularization
    ///
    /// Higher values enforce stronger attention quality constraints.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastAttentionRegularizationLoss;

    /// <summary>
    /// The configuration settings for this Transformer network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the architecture configuration which defines the structure and properties
    /// of this Transformer network, including settings like embedding size, number of attention 
    /// heads, and feed-forward dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the blueprint for our Transformer.
    /// 
    /// It contains all the important settings that determine how the Transformer works:
    /// - How many attention mechanisms to use
    /// - How large each part of the network should be
    /// - How information flows through the network
    /// 
    /// Just like a house blueprint defines the structure of a house, this architecture
    /// defines the structure of our Transformer neural network.
    /// </para>
    /// </remarks>
    private readonly TransformerArchitecture<T> _transformerArchitecture;

    /// <summary>
    /// Gets or sets the attention mask used in the Transformer.
    /// </summary>
    /// <remarks>
    /// This mask is used to control which positions are attended to in the self-attention layers.
    /// It's particularly useful for tasks like sequence generation where future tokens should be masked.
    /// </remarks>
    public Tensor<T>? AttentionMask { get; set; }

    /// <summary>
    /// The optimizer used to update the Transformer's parameters during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The optimizer determines how the Transformer's parameters should be adjusted based on the calculated gradients.
    /// It's responsible for the learning process, controlling how quickly and in what manner the model improves.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is like a coach for the Transformer.
    /// 
    /// Think of training the Transformer as teaching it to play a sport:
    /// - The optimizer decides how to adjust the Transformer's technique (its parameters)
    /// - It looks at how the Transformer performed (the loss) and suggests improvements
    /// - Different optimizers have different strategies, like focusing on quick improvements or steady, consistent progress
    /// 
    /// The choice of optimizer can significantly affect how well and how quickly the Transformer learns.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Overrides the base hook so any caller of <see cref="NeuralNetworkBase{T}.SetBaseTrainOptimizer"/>
    /// — typically <c>AiModelBuilder.ConfigureOptimizer</c> — also updates
    /// the subclass-private <see cref="_optimizer"/> field. Without this
    /// override, the field stayed pinned to whatever was constructed in
    /// the ctor, while training routed through the base optimizer slot —
    /// so <see cref="GetModelMetadata"/> and <see cref="SerializeNetworkSpecificData"/>
    /// (which read <see cref="_optimizer"/>) would report and persist the
    /// stale ctor optimizer instead of the live training one. The current
    /// effective optimizer is now the single source of truth for both
    /// reads and writes.
    /// </summary>
    internal override void SetBaseTrainOptimizer(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
    {
        base.SetBaseTrainOptimizer(optimizer);
        // The previous version only updated _optimizer when the new
        // optimizer was non-null, leaving the field pointing at the
        // OLD optimizer when a caller cleared the base slot. That made
        // GetModelMetadata / SerializeNetworkSpecificData report the
        // stale optimizer instead of "no optimizer configured" — exactly
        // the staleness the override is supposed to prevent. Mirror the
        // base's null-clears-the-slot semantic by falling back to the
        // ctor-supplied default optimizer (so subsequent training calls
        // still have something usable) rather than holding onto a stale
        // reference. Closes review-comment #1270.vhmE.
        if (optimizer is not null)
        {
            _optimizer = optimizer;
        }
        else
        {
            // null = "reset to default". Reconstruct the same Vaswani-2017
            // recipe the ctor would build (Adam β₁=0.9, β₂=0.98, ε=1e-9
            // + NoamSchedule on _transformerArchitecture.ModelDimension /
            // WarmupSteps, stepped per batch). Mirrors the deserialization
            // fallback at line 794-806 so behaviour is consistent across
            // the two "no caller-supplied optimizer" entry points.
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = 1e-3,
                    Beta2 = 0.98,
                    Epsilon = 1e-9,
                    LearningRateScheduler = new LearningRateSchedulers.NoamSchedule(
                        modelDimension: _transformerArchitecture.ModelDimension,
                        warmupSteps: _transformerArchitecture.WarmupSteps),
                    SchedulerStepMode = LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                });
        }
    }

    /// <summary>
    /// Creates a new Transformer neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">
    /// The architecture configuration that defines how this Transformer will be structured.
    /// This includes settings like embedding size, number of attention heads, and feed-forward dimensions.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Transformer neural network with the provided architecture
    /// configuration. It passes the architecture to the base class constructor and also stores it
    /// for use in initializing the Transformer-specific layers.
    /// </para>
    /// <para><b>For Beginners:</b> This is where we create our Transformer network.
    /// 
    /// When you create a new Transformer, you provide a blueprint (the architecture) that specifies:
    /// - How many layers it should have
    /// - How attention works in the network
    /// - How large the various components should be
    /// 
    /// This is similar to how you might specify the size, number of rooms, and layout when building a house.
    /// </para>
    /// </remarks>
    public Transformer(TransformerArchitecture<T> architecture, ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, TransformerOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new TransformerOptions();
        Options = _options;
        _transformerArchitecture = architecture;
        // Default optimizer for Transformer is Adam, not vanilla GradientDescent.
        // Vaswani 2017 ("Attention Is All You Need") trains Transformers with Adam
        // (β₁=0.9, β₂=0.98, ε=1e-9) — vanilla SGD does not produce competitive
        // models on attention architectures because attention's softmax + LayerNorm
        // gradient surface has very different scales across parameters and SGD's
        // single-rate update can't accommodate that without per-parameter adaptation.
        // Every other neural-net family in this library defaults to Adam via
        // GetOrCreateBaseOptimizer(); Transformer was the lone outlier defaulting
        // to GradientDescent, which made byte-LM training silently fail to converge
        // (see ooples/AiDotNet#1264 for the V=256 reproducer that demonstrates this).
        // LR=1e-3 follows PyTorch's torch.optim.Adam default and lines up with the
        // Adam paper's recommendation; consumers needing the Vaswani-2017 schedule
        // (warmup + inverse-sqrt decay) can pass an explicit optimizer + scheduler.
        if (optimizer is null)
        {
            // Vaswani 2017 §5.3 hyperparameters, applied AS A RECIPE
            // (β₁=0.9, β₂=0.98, ε=1e-9). When the NoamSchedule is attached
            // below, GradientBasedOptimizerBase uses the scheduler's
            // CurrentLearningRate AS THE ABSOLUTE LR for each step
            // (InitialLearningRate=1e-3 acts only as the base ctor's
            // positive-lr-guard sentinel — it is bypassed entirely once a
            // scheduler is present). Effective LR per batch equals
            // NoamSchedule(t) directly, NOT InitialLearningRate × NoamSchedule(t).
            // Closes review-comment #1270.vhm-.
            //
            // β₂=0.98 (not the library default 0.999) is paired with the
            // inverse-sqrt warmup schedule below: the small β₂ tracks
            // attention/embedding gradients that change rapidly during
            // training, while warmup keeps the early-step LR small enough
            // that the (still-stabilizing) second-moment estimates don't
            // produce mis-scaled updates. Applying β₂=0.98 WITHOUT the
            // schedule is broken — the previous default (β₂=0.999, no
            // schedule) was a workaround for that mismatch. We now apply
            // them together so the recipe matches the paper exactly.
            //
            // The lazy-init path of NeuralNetworkBase only resolves
            // ModelDimension after the first forward pass, but the Vaswani
            // schedule needs d_model upfront to compute its peak LR. The
            // architecture exposes ModelDimension at construction time, so
            // we bind the schedule to that value here.
            var defaultAdamOpts = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-3,
                Beta2 = 0.98,
                Epsilon = 1e-9,
                LearningRateScheduler = new LearningRateSchedulers.NoamSchedule(
                    modelDimension: architecture.ModelDimension,
                    warmupSteps: architecture.WarmupSteps),
                SchedulerStepMode = LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            };
            _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, defaultAdamOpts);
        }
        else
        {
            _optimizer = optimizer;
        }

        // Mirror our optimizer into the base class's training-optimizer
        // slot so AiModelBuilder.ConfigureOptimizer (which calls
        // NeuralNetworkBase.SetBaseTrainOptimizer before nn.Train) and our
        // Train override resolve to the SAME optimizer instance. Without
        // this, the base's GetOrCreateBaseOptimizer would lazy-construct a
        // separate default Adam the first time anything outside our Train
        // override touched it (e.g. via TrainBatched on the base path),
        // and the streaming-loader optimizer override couldn't reach us.
        SetBaseTrainOptimizer(_optimizer);

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAttentionRegularizationLoss = NumOps.Zero;

        InitializeLayers();
    }

    /// <summary>
    /// Sets up the layers of the Transformer network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided by the user or creates default Transformer layers.
    /// A typical Transformer consists of attention mechanisms, normalization layers, and feed-forward networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the Transformer.
    /// 
    /// It works in one of two ways:
    /// - If you've provided your own custom layers, it uses those
    /// - Otherwise, it creates a standard set of Transformer layers
    /// 
    /// These layers typically include:
    /// - Attention layers (which let the model focus on relevant parts of the input)
    /// - Normalization layers (which keep the numbers from getting too large or small)
    /// - Feed-forward layers (which process the information)
    /// 
    /// It's like assembling the rooms and sections of a house according to the blueprint.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default transformer layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultTransformerLayers(_transformerArchitecture));
        }
    }

    /// <summary>
    /// Ensures that custom layers provided for the Transformer meet the minimum requirements.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <remarks>
    /// <para>
    /// A valid Transformer must include at least one attention layer and one normalization layer.
    /// Attention layers allow the model to focus on different parts of the input sequence.
    /// Normalization layers help stabilize training by normalizing the activations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if your custom layers will actually work as a Transformer.
    /// 
    /// For a Transformer to function properly, it needs at minimum:
    /// - An attention layer (which helps the model focus on important parts of the input)
    /// - A normalization layer (which keeps the numbers stable during training)
    /// 
    /// If either of these is missing, it's like trying to build a house without walls or a foundation - it won't work!
    /// 
    /// This method checks for these essential components and raises an error if they're missing.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the custom layers don't include required layer types.
    /// </exception>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        bool hasAttentionLayer = false;
        bool hasLayerNorm = false;

        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is MultiHeadAttentionLayer<T>)
            {
                hasAttentionLayer = true;
            }
            else if (layers[i] is LayerNormalizationLayer<T>)
            {
                hasLayerNorm = true;
            }
        }

        if (!hasAttentionLayer)
        {
            throw new InvalidOperationException("Custom Transformer must include at least one MultiHeadAttentionLayer.");
        }

        if (!hasLayerNorm)
        {
            throw new InvalidOperationException("Custom Transformer must include at least one LayerNormalizationLayer.");
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for attention regularization across all attention layers.
    /// </summary>
    /// <returns>The computed attention regularization auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method aggregates auxiliary losses from all MultiHeadAttentionLayers in the Transformer.
    /// It collects both entropy regularization and head diversity penalties from each attention layer.
    /// Formula: L = (1/N) * Σ_layers auxloss_i where N = number of attention layers
    /// </para>
    /// <para><b>For Beginners:</b> This calculates network-wide attention quality.
    ///
    /// Transformer attention regularization works by:
    /// 1. Finding all attention layers in the network
    /// 2. Computing auxiliary loss for each layer (if enabled)
    /// 3. Averaging these losses across all layers
    /// 4. Returning the network-level regularization penalty
    ///
    /// This helps because:
    /// - Maintains attention quality throughout the entire network
    /// - Prevents attention collapse at any level
    /// - Encourages diverse attention patterns across all layers
    /// - Improves interpretability and robustness
    ///
    /// The auxiliary loss is added to the main task loss during training.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastAttentionRegularizationLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        T totalAuxLoss = NumOps.Zero;
        int attentionLayerCount = 0;

        // Aggregate auxiliary losses from all attention layers
        foreach (var layer in Layers)
        {
            if (layer is IAuxiliaryLossLayer<T> auxLayer && auxLayer.UseAuxiliaryLoss)
            {
                T layerAuxLoss = auxLayer.ComputeAuxiliaryLoss();
                totalAuxLoss = NumOps.Add(totalAuxLoss, layerAuxLoss);
                attentionLayerCount++;
            }
        }

        // Average over all attention layers
        if (attentionLayerCount > 0)
        {
            totalAuxLoss = NumOps.Divide(totalAuxLoss, NumOps.FromDouble(attentionLayerCount));
        }

        _lastAttentionRegularizationLoss = totalAuxLoss;
        return totalAuxLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the attention regularization auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about network-level attention regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about attention regularization across the Transformer,
    /// including aggregated losses, layer counts, and configuration parameters.
    /// This information is useful for monitoring training progress and debugging attention issues.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how attention works across the network.
    ///
    /// The diagnostics include:
    /// - Total attention regularization loss (averaged across layers)
    /// - Weight applied to the regularization
    /// - Number of attention layers with regularization enabled
    /// - Whether network-level regularization is enabled
    ///
    /// This helps you:
    /// - Monitor attention quality throughout the network
    /// - Debug issues with attention collapse
    /// - Understand the impact of regularization at the network level
    /// - Track which layers are contributing to regularization
    ///
    /// You can use this information to adjust regularization settings for better training results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "TotalAttentionRegularizationLoss", _lastAttentionRegularizationLoss?.ToString() ?? "0" },
            { "AttentionRegularizationWeight", AuxiliaryLossWeight?.ToString() ?? "0.005" },
            { "UseAttentionRegularization", UseAuxiliaryLoss.ToString() }
        };

        // Count attention layers with regularization enabled
        int attentionLayerCount = Layers.OfType<IAuxiliaryLossLayer<T>>()
            .Count(l => l.UseAuxiliaryLoss);
        diagnostics["AttentionLayersWithRegularization"] = attentionLayerCount.ToString();

        // Total attention layers
        int totalAttentionLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        diagnostics["TotalAttentionLayers"] = totalAttentionLayers.ToString();

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Updates the parameters of all layers in the Transformer network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer based on their parameter counts.
    /// It's typically used during training when applying gradient updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the Transformer's internal values during training.
    /// 
    /// Think of parameters as the "settings" of the Transformer:
    /// - Each layer needs a certain number of parameters to function
    /// - During training, these parameters are constantly adjusted to improve performance
    /// - This method takes a big list of new parameter values and gives each layer its share
    /// 
    /// It's like distributing updated parts to each section of a machine so it works better.
    /// Each layer gets exactly the number of parameters it needs.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the Transformer network to generate predictions.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing the predictions.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input through each layer of the Transformer sequentially.
    /// It handles both the encoder and decoder parts of the Transformer if present.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data and runs it through the entire Transformer.
    /// 
    /// It's like sending a message through a complex machine:
    /// - The input goes through each part of the Transformer in order
    /// - Each layer processes the data in its own way (attention, normalization, etc.)
    /// - The final output is the Transformer's prediction or transformation of your input
    /// 
    /// This is used when you want to use a trained Transformer to process new data.
    /// </para>
    /// </remarks>
    /// <inheritdoc />
    /// <remarks>
    /// Transformer's eager forward routes the encoder/decoder
    /// cross-attention pattern: <see cref="DecoderLayer{T}"/> takes the
    /// encoder's output as a second input, <see cref="AttentionLayer{T}"/>
    /// takes the attention mask as a second input, and other layers are
    /// invoked through the standard single-input
    /// <see cref="ILayer{T}.Forward(Tensor{T})"/>. The encoder output is
    /// captured at the last <see cref="MultiHeadAttentionLayer{T}"/>
    /// before any decoder layer in the chain. The public
    /// <see cref="NeuralNetworkBase{T}.Predict"/> wrapper handles
    /// training-mode toggle, no-grad scope, batch-dim promotion, and
    /// output squeeze — see that method's remarks for the inference-
    /// scaffolding contract (issue #1221).
    /// </remarks>
    protected override Tensor<T> PredictEager(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> output = input;
        Tensor<T>? encoderOutput = null;
        bool seenDecoder = false;
        Tensor<T> mask = AttentionMask ?? Tensor<T>.CreateDefault(input._shape, NumOps.One); // Default to all ones if no mask is provided

        // Process all layers sequentially
        // The layer list structure: input projection, positional encoding, dropout, then encoder/decoder blocks
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is DecoderLayer<T> decoderLayer)
            {
                seenDecoder = true;
                // Decoder layer with cross-attention needs encoder output
                output = decoderLayer.Forward(output, encoderOutput ?? output, mask);
            }
            else if (Layers[i] is AttentionLayer<T> attentionLayer)
            {
                output = attentionLayer.Forward(output, mask);
            }
            else
            {
                output = Layers[i].Forward(output);
            }

            // Track encoder output for cross-attention in decoders.
            // The encoder output we want is the LAST MultiHeadAttention
            // output before any DecoderLayer in the chain — for an
            // encoder stack with multiple blocks, decoder cross-attention
            // should consume the fully-encoded representation, not the
            // first encoder block's output. Keep updating `encoderOutput`
            // on every encoder-block attention until we hit the first
            // decoder; after that, freeze it (subsequent decoder blocks
            // share the same encoder context).
            if (!seenDecoder && Layers[i] is MultiHeadAttentionLayer<T>)
            {
                // Only capture if there's a decoder downstream — otherwise
                // there's no consumer for this state and we'd be retaining
                // a tensor reference unnecessarily.
                for (int j = i + 1; j < Layers.Count; j++)
                {
                    if (Layers[j] is DecoderLayer<T>)
                    {
                        encoderOutput = output;
                        break;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Trains the Transformer on a single sample <i>or</i> a batched tensor of samples.
    /// </summary>
    /// <param name="input">
    /// Either a single-sample input (e.g. <c>[ctxLen]</c> / <c>[ctxLen, F]</c>) or an
    /// explicitly batched tensor (e.g. <c>[B, ctxLen]</c> / <c>[B, ctxLen, F]</c>).
    /// Single-sample inputs are auto-promoted to <c>[1, …]</c>; batched inputs are
    /// passed through unchanged.
    /// </param>
    /// <param name="expectedOutput">Per-sample target with matching batch arity.</param>
    /// <remarks>
    /// <para>
    /// One forward pass + tape-backward + optimizer step. The configured optimizer is
    /// Adam by default (LR = 1e-3) — see the constructor docs for why and how to override.
    /// </para>
    /// <para>
    /// <b>Important — per-sample vs. batched training:</b> calling this method per-sample
    /// in a Python-style for-loop converges very slowly on language-model-style tasks
    /// (V ≥ 32 vocabularies). Each step's gradient must compete against <c>V − 1</c>
    /// negative classes; the cumulative noise stalls the model at the unigram-prior
    /// accuracy in any practical step budget. For byte-LM, character-LM, token
    /// classification, or any task with a vocabulary head, **batch your training**:
    /// either pass a <c>[B, ctxLen, …]</c> tensor directly, or use the
    /// <see cref="NeuralNetworkBase{T}.TrainBatched"/> convenience overload that stacks
    /// an array of single-sample tensors for you. Practical batch sizes for Transformers:
    /// 16–64. The gradient averaging across a batch reduces noise by <c>√B</c>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the Transformer using example data.
    /// The process works like this:
    /// 1. The Transformer makes a prediction based on the input
    /// 2. We compare this prediction to the expected output
    /// 3. We calculate how wrong the prediction was (the "loss")
    /// 4. We adjust the Transformer's internal values to make it a little more accurate next time
    ///
    /// This process is repeated many times with different examples. For language models
    /// (which is most Transformer use cases), give the Transformer a BATCH of examples
    /// at once rather than one example at a time — TrainBatched does this for you, or
    /// pass a tensor whose first dimension is the batch size.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            // Resolve via GetOrCreateBaseOptimizer so a builder-side
            // SetBaseTrainOptimizer override (e.g. AiModelBuilder.ConfigureOptimizer
            // → AdamW with a learning-rate scheduler) actually drives this
            // training step. The ctor seeded the base optimizer with our
            // _optimizer instance, so when no override is in effect this
            // resolves to the same Vaswani Adam we'd have used pre-refactor.
            TrainWithTape(input, expectedOutput, GetOrCreateBaseOptimizer());
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Sets the attention mask for the Transformer.
    /// </summary>
    /// <param name="mask">The attention mask to be used in self-attention layers.</param>
    /// <remarks>
    /// Call this method before training or prediction to set a mask for controlling attention.
    /// </remarks>
    public void SetAttentionMask(Tensor<T> mask)
    {
        AttentionMask = mask;
    }

    /// <summary>
    /// Retrieves metadata about the Transformer model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the Transformer.</returns>
    /// <remarks>
    /// <para>
    /// This method collects and returns various pieces of information about the Transformer,
    /// including its type, architecture details, and current state.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a summary of the Transformer's current state and structure.
    /// 
    /// It's like creating a report card for the Transformer, including:
    /// - What type of model it is (Transformer)
    /// - How it's structured (number of layers, size of each layer, etc.)
    /// - Its current training progress
    /// - Other important details about its configuration
    /// 
    /// This information is useful for keeping track of different models, especially when you're
    /// experimenting with multiple configurations.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumHeads", _transformerArchitecture.NumHeads },
                { "NumEncoderLayers", _transformerArchitecture.NumEncoderLayers },
                { "NumDecoderLayers", _transformerArchitecture.NumDecoderLayers },
                { "MaxSequenceLength", _transformerArchitecture.MaxSequenceLength },
                { "VocabularySize", _transformerArchitecture.VocabularySize },
                { "DropoutRate", _transformerArchitecture.DropoutRate },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() },
                { "LossFunction", LossFunction.GetType().Name },
                { "Optimizer", _optimizer.GetType().Name }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Transformer-specific data to a binary stream.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes Transformer-specific configuration and state data to a binary stream.
    /// It allows the Transformer's current state to be saved and later reconstructed.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important details about the Transformer to a file.
    /// 
    /// It's like taking a snapshot of the Transformer's current state, including:
    /// - Its configuration (how it's set up)
    /// - Its current parameter values (what it has learned so far)
    /// 
    /// This allows you to save your trained Transformer and use it again later without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write Transformer-specific architecture details
        writer.Write(_transformerArchitecture.NumHeads);
        writer.Write(_transformerArchitecture.NumEncoderLayers);
        writer.Write(_transformerArchitecture.NumDecoderLayers);
        writer.Write(_transformerArchitecture.MaxSequenceLength);
        writer.Write(_transformerArchitecture.VocabularySize);
        writer.Write(Convert.ToDouble(_transformerArchitecture.DropoutRate));

        // Write loss function and optimizer types
        SerializationHelper<T>.SerializeInterface(writer, LossFunction);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes Transformer-specific data from a binary stream.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads Transformer-specific configuration and state data from a binary stream.
    /// It reconstructs the Transformer's state from previously serialized data.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads all the important details about a Transformer from a file.
    /// 
    /// It's like reconstructing the Transformer from a saved snapshot, including:
    /// - Rebuilding its configuration (how it was set up)
    /// - Restoring its parameter values (what it had learned)
    /// 
    /// This allows you to load a previously trained Transformer and use it immediately without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read Transformer-specific architecture details
        int numHeads = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int maxSequenceLength = reader.ReadInt32();
        int vocabularySize = reader.ReadInt32();
        T dropoutRate = NumOps.FromDouble(reader.ReadDouble());

        // Read and reconstruct loss function and optimizer (must match serialization order).
        LossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader)
            ?? LossFunction;

        // Match the constructor's default-optimizer policy: Vaswani 2017
        // recipe (β₁=0.9, β₂=0.98, ε=1e-9, lr=1e-3 + NoamSchedule). Stale
        // state-dicts written before this fix didn't serialize their
        // optimizer; reading null-optimizer back must produce the SAME
        // optimizer the ctor would, otherwise the deserialized model silently
        // regresses to non-converging vanilla SGD or a different (non-paper)
        // Adam configuration.
        _optimizer = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = 1e-3,
                    Beta2 = 0.98,
                    Epsilon = 1e-9,
                    LearningRateScheduler = new LearningRateSchedulers.NoamSchedule(
                        modelDimension: _transformerArchitecture.ModelDimension,
                        warmupSteps: _transformerArchitecture.WarmupSteps),
                    SchedulerStepMode = LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                });

        // Keep the base optimizer slot in sync after deserialization too
        // — Train() now resolves through GetOrCreateBaseOptimizer, so a
        // load-then-resume-training flow needs the deserialized optimizer
        // installed on both sides.
        SetBaseTrainOptimizer(_optimizer);
    }

    /// <summary>
    /// Creates a new instance of the Transformer with the same architecture and configuration.
    /// </summary>
    /// <returns>A new instance of the Transformer with the same configuration as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new Transformer neural network with the same architecture, loss function,
    /// and optimizer as the current instance. The new instance has freshly initialized parameters,
    /// making it useful for creating separate instances with identical configurations or for
    /// resetting a network while preserving its structure.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a brand new Transformer with the same setup.
    /// 
    /// Think of it like creating a blueprint copy:
    /// - It has the same architecture (number of layers, attention heads, etc.)
    /// - It uses the same loss function to measure performance
    /// - It uses the same optimizer to learn from data
    /// - But it starts with fresh parameters (weights and biases)
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh network but keep the same design
    /// - Create multiple networks with identical settings for comparison
    /// - Reset a network to its initial state
    /// 
    /// The new Transformer will need to be trained from scratch, as it doesn't
    /// inherit any of the learned knowledge from the original.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Transformer<T>(
            _transformerArchitecture,
            LossFunction,
            _optimizer);
    }
}
