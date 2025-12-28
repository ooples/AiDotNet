using AiDotNet.Autodiff;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Squeeze-and-Excitation layer that recalibrates channel-wise feature responses adaptively.
/// </summary>
/// <remarks>
/// <para>
/// A Squeeze-and-Excitation layer enhances the representational power of a network by explicitly modeling the 
/// interdependencies between channels. It does this by performing two operations:
/// 1. "Squeeze" - aggregating feature maps across spatial dimensions to produce a channel descriptor
/// 2. "Excitation" - using this descriptor to recalibrate the original feature maps channel-wise
/// </para>
/// <para><b>For Beginners:</b> This layer helps the neural network focus on the most important features.
/// 
/// Think of it like how your brain works when looking at a picture:
/// - First, you get a rough idea of what's in the image (the "squeeze" step)
/// - Then, you decide which parts to pay more attention to (the "excitation" step)
/// - Finally, you look at the image again with this focused attention
/// 
/// For example, if the network is processing an image of a cat, the Squeeze-and-Excitation layer might:
/// - First compress all the information to understand "this is probably a cat"
/// - Then decide to pay more attention to features that look like ears, whiskers, and fur
/// - Finally enhance those important features in the original image data
/// 
/// This helps the network become more accurate and efficient by focusing on what matters most.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SqueezeAndExcitationLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>, IChainableComputationGraph<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer computes a channel attention regularization loss that encourages balanced
    /// channel importance. This helps prevent the layer from over-relying on specific channels.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the layer uses an additional learning signal.
    ///
    /// When enabled (true):
    /// - The layer encourages balanced attention across channels
    /// - This helps prevent over-reliance on specific features
    /// - Training may be more stable and produce more robust representations
    ///
    /// When disabled (false):
    /// - Only the main task loss is used for training
    /// - This is the default setting
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the auxiliary loss contribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines how much the channel attention regularization contributes to the total loss.
    /// The default value of 0.01 provides a good balance between the main task and regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much importance to give to the channel attention regularization.
    ///
    /// The weight affects training:
    /// - Higher values (e.g., 0.05) make the network prioritize balanced channel attention more strongly
    /// - Lower values (e.g., 0.001) make the regularization less important
    /// - The default (0.01) works well for most computer vision tasks
    ///
    /// If your network is over-fitting to specific channels, increase this value.
    /// If the main task is more important, you might decrease it.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Stores the last computed channel attention regularization loss for diagnostic purposes.
    /// </summary>
    private T _lastChannelAttentionLoss;

    /// <summary>
    /// Caches the excitation weights from the forward pass for auxiliary loss computation.
    /// Shape: [batchSize, channels]
    /// </summary>
    private Tensor<T>? _lastExcitationWeights;
    private Tensor<T>? _lastSqueezed;
    private Tensor<T>? _lastFc1Biased;
    private Tensor<T>? _lastFc1Activated;
    private Tensor<T>? _lastFc2Biased;

    /// <summary>
    /// The number of input and output channels in the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the number of channels in the input tensor, which also corresponds to the number of output channels.
    /// In neural networks, channels typically represent different feature maps or types of information.
    /// </para>
    /// <para><b>For Beginners:</b> Think of channels as different types of information.
    /// 
    /// For example, in an image:
    /// - One channel might detect edges
    /// - Another might detect colors
    /// - Another might detect textures
    /// 
    /// The neural network processes each of these information types separately before combining them.
    /// </para>
    /// </remarks>
    private readonly int _channels;

    /// <summary>
    /// The number of channels in the bottleneck (reduced dimension).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the reduced number of channels used in the intermediate representation of the Squeeze-and-Excitation block.
    /// It is calculated as _channels divided by the reduction ratio, creating a bottleneck that forces the network to focus on the most important features.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a compression rate for information.
    /// 
    /// If you have 64 channels (types of information) and _reducedChannels is 16:
    /// - The layer has to compress all 64 types down to just 16
    /// - This forces it to keep only the most important information
    /// - It's like summarizing a long story into a few key points
    /// 
    /// The smaller this number, the more aggressive the compression, which can help the network
    /// focus but might also cause it to miss some details.
    /// </para>
    /// </remarks>
    private readonly int _reducedChannels;

    /// <summary>
    /// The weights for the first fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the weights that transform the squeezed representation from _channels dimensions to _reducedChannels dimensions.
    /// These weights are learned during training to identify important patterns in the feature maps.
    /// </para>
    /// <para><b>For Beginners:</b> These are the adjustable values that transform the compressed information.
    /// 
    /// Think of these weights like knobs that the network can turn:
    /// - Each weight determines how much influence one feature has on another
    /// - The network adjusts these values during training to get better results
    /// - This matrix helps reduce the information from many channels to fewer channels
    /// 
    /// This is part of the "squeeze" operation that compresses information.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights1;

    /// <summary>
    /// The bias values for the first fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the bias values that are added after applying the _weights1 transformation.
    /// Biases allow the network to shift the activation function, providing more flexibility in learning.
    /// </para>
    /// <para><b>For Beginners:</b> These are baseline values added after the transformation.
    ///
    /// Think of biases like default settings:
    /// - They provide starting values that get added regardless of the input
    /// - They help the network represent patterns more easily
    /// - Without biases, every feature would have to start from zero
    ///
    /// For example, if a certain feature should usually be active, a positive bias
    /// means it starts with some activation even before the input is considered.
    /// </para>
    /// </remarks>
    private Tensor<T> _bias1;

    /// <summary>
    /// The weights for the second fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the weights that transform from the reduced representation back to the original channel dimensions.
    /// These weights determine how much attention to give to each channel in the original input.
    /// </para>
    /// <para><b>For Beginners:</b> These values determine how important each feature is.
    ///
    /// After compressing the information:
    /// - These weights expand it back to the original size
    /// - However, they now contain "importance scores" for each feature
    /// - Features deemed more important get higher weights
    ///
    /// This is part of the "excitation" operation that decides which features to emphasize.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights2;

    /// <summary>
    /// The bias values for the second fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the bias values that are added after applying the _weights2 transformation.
    /// These biases help determine the baseline attention given to each channel.
    /// </para>
    /// <para><b>For Beginners:</b> These are baseline attention values for each feature.
    ///
    /// Similar to the first set of biases:
    /// - They provide default attention levels for each feature
    /// - They're added to the calculated importance scores
    /// - They help ensure certain features get at least some attention
    ///
    /// For example, if color information is usually important, its bias might be higher
    /// so it receives attention even when the specific input doesn't strongly suggest it.
    /// </para>
    /// </remarks>
    private Tensor<T> _bias2;

    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass. It is used during the backward pass
    /// to calculate gradients. The value is null if no forward pass has been performed yet or after ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like short-term memory of what the layer just processed.
    /// 
    /// During training:
    /// - The layer needs to remember what input it received
    /// - This helps it calculate how to improve itself
    /// - It's like keeping your work when solving a math problem, so you can check your steps
    /// 
    /// This value is temporarily stored during training and is cleared when moving to a new sample.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor from the most recent forward pass. It is used during the backward pass
    /// to calculate gradients. The value is null if no forward pass has been performed yet or after ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is a record of what the layer just produced.
    /// 
    /// Similar to remembering the input:
    /// - The layer also needs to remember what output it generated
    /// - This helps determine how far off its prediction was
    /// - It's needed to calculate how to improve the weights and biases
    /// 
    /// This is another piece of temporary memory used during training.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The gradient of the loss with respect to _weights1 from the most recent backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient of the loss function with respect to _weights1. It indicates how _weights1
    /// should be adjusted to reduce the loss. The value is null if no backward pass has been performed yet or after ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how the first set of weights should change to improve performance.
    /// 
    /// During training:
    /// - The network calculates how much each weight contributed to any errors
    /// - This gradient indicates the direction and amount each weight should change
    /// - Larger values mean the weight needs more adjustment
    /// 
    /// Think of it like feedback saying "this weight should be a little higher" or
    /// "that weight should be much lower" to get better results next time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _weights1Gradient;

    /// <summary>
    /// The gradient of the loss with respect to _bias1 from the most recent backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient of the loss function with respect to _bias1. It indicates how _bias1
    /// should be adjusted to reduce the loss. The value is null if no backward pass has been performed yet or after ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how the first set of biases should change.
    ///
    /// Similar to the weight gradients:
    /// - This indicates how each bias value should be adjusted
    /// - It helps fine-tune the default settings for each feature
    /// - The network uses this to update the biases during learning
    ///
    /// These gradients help the network gradually improve its performance over time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _bias1Gradient;

    /// <summary>
    /// The gradient of the loss with respect to _weights2 from the most recent backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient of the loss function with respect to _weights2. It indicates how _weights2
    /// should be adjusted to reduce the loss. The value is null if no backward pass has been performed yet or after ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how the second set of weights should change.
    ///
    /// This is crucial because:
    /// - These weights determine which features get more attention
    /// - Adjusting them helps the network focus on the right things
    /// - It's like learning which parts of a picture are most important for identifying what's in it
    ///
    /// The network uses these gradients to gradually improve its "attention mechanism" over time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _weights2Gradient;

    /// <summary>
    /// The gradient of the loss with respect to _bias2 from the most recent backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient of the loss function with respect to _bias2. It indicates how _bias2
    /// should be adjusted to reduce the loss. The value is null if no backward pass has been performed yet or after ResetState is called.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how the second set of biases should change.
    ///
    /// These gradients:
    /// - Help adjust the default attention given to each feature
    /// - Allow the network to learn which features are generally more important
    /// - Fine-tune the "excitation" part of the layer
    ///
    /// Along with the other gradients, these help the network improve through training.
    /// </para>
    /// </remarks>
    private Tensor<T>? _bias2Gradient;

    /// <summary>
    /// Gets or sets the weight for L1 sparsity regularization on attention weights.
    /// </summary>
    /// <value>
    /// The weight to apply to the L1 sparsity loss. Default is 0.0001.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property controls the strength of L1 sparsity regularization applied to
    /// the channel attention weights. Higher values encourage more sparse attention
    /// (fewer active channels), while lower values allow more distributed attention.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly to encourage sparse attention.
    ///
    /// Sparsity regularization:
    /// - Encourages the network to focus on fewer, more important channels
    /// - Helps prevent overfitting by reducing model complexity
    /// - Can improve interpretability by making channel selection clearer
    ///
    /// Typical values range from 0.0001 to 0.01. Set to 0 to disable sparsity regularization.
    /// </para>
    /// </remarks>
    public T SparsityWeight { get; set; }

    /// <summary>
    /// The activation function applied after the first fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the scalar activation function that is applied element-wise after the first fully connected layer.
    /// The default is ReLU, which keeps positive values unchanged and sets negative values to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This is a mathematical function that adds non-linearity after the first transformation.
    /// 
    /// Activation functions:
    /// - Add non-linearity, allowing the network to learn complex patterns
    /// - ReLU (the default) keeps positive values and changes negative values to zero
    /// - This helps the network focus on active features and ignore inactive ones
    /// 
    /// Without activation functions, neural networks would only be able to learn linear relationships,
    /// severely limiting what they can do.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _firstActivation;

    /// <summary>
    /// The activation function applied after the second fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the scalar activation function that is applied element-wise after the second fully connected layer.
    /// The default is Sigmoid, which squashes values to be between 0 and 1, making them appropriate for channel attention weights.
    /// </para>
    /// <para><b>For Beginners:</b> This transforms the output into attention scores between 0 and 1.
    /// 
    /// The Sigmoid function (default):
    /// - Squashes any number to be between 0 and 1
    /// - 0 means "ignore this feature completely"
    /// - 1 means "give this feature full attention"
    /// - Values in between provide partial attention
    /// 
    /// This is perfect for the "excitation" step because we need to decide how much
    /// attention to give each feature, which is naturally expressed as a percentage.
    /// </para>
    /// </remarks>
    private readonly IActivationFunction<T>? _secondActivation;

    /// <summary>
    /// The vector activation function applied after the first fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the vector activation function that is applied to entire vectors after the first fully connected layer.
    /// It is used instead of _firstActivation when a vector activation is provided in the constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the scalar activation, but works on entire groups of values at once.
    /// 
    /// Vector activations:
    /// - Process entire rows of numbers together
    /// - Can capture relationships between different elements
    /// - Might normalize or transform values based on the entire group
    /// 
    /// This allows for more sophisticated transformations that consider how different
    /// features relate to each other, rather than processing each feature independently.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _firstVectorActivation;

    /// <summary>
    /// The vector activation function applied after the second fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the vector activation function that is applied to entire vectors after the second fully connected layer.
    /// It is used instead of _secondActivation when a vector activation is provided in the constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This transforms groups of values into attention scores.
    /// 
    /// Similar to scalar activation, but:
    /// - It can consider relationships between different channels
    /// - It might use competition between features to determine attention
    /// - It could normalize attention so the total adds up to 100%
    /// 
    /// For example, it might increase attention on texture features while decreasing
    /// attention on color features if it determines texture is more important for this specific input.
    /// </para>
    /// </remarks>
    private readonly IVectorActivationFunction<T>? _secondVectorActivation;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, as it contains trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the Squeeze-and-Excitation layer can be trained through backpropagation.
    /// Since this layer has trainable parameters (weights and biases), it supports training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (weights and biases) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// For this layer, the value is always true because it needs to learn which features 
    /// are most important to pay attention to.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SqueezeAndExcitationLayer{T}"/> class with scalar activation functions.
    /// </summary>
    /// <param name="channels">The number of input and output channels.</param>
    /// <param name="reductionRatio">The ratio by which to reduce the number of channels in the bottleneck.</param>
    /// <param name="firstActivation">The activation function for the first fully connected layer. Defaults to ReLU if not specified.</param>
    /// <param name="secondActivation">The activation function for the second fully connected layer. Defaults to Sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a Squeeze-and-Excitation layer with the specified number of channels and reduction ratio.
    /// The reduction ratio determines how much the channel dimension is compressed in the bottleneck.
    /// The activation functions control the non-linearities applied after each fully connected layer.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new Squeeze-and-Excitation layer.
    /// 
    /// The parameters you provide determine:
    /// - channels: How many different feature types the layer will process
    /// - reductionRatio: How much to compress the information (higher means more compression)
    /// - firstActivation: How to process information after the first step (defaults to ReLU, which keeps only positive values)
    /// - secondActivation: How to determine importance of each feature (defaults to Sigmoid, which outputs values between 0 and 1)
    /// 
    /// Think of it like this: if you have 64 channels (different types of features) and a reduction ratio of 16,
    /// the layer will compress those 64 channels down to just 4 during the middle step, forcing it to focus
    /// on only the most important patterns.
    /// </para>
    /// </remarks>
    public SqueezeAndExcitationLayer(int channels, int reductionRatio,
        IActivationFunction<T>? firstActivation = null,
        IActivationFunction<T>? secondActivation = null)
        : base([[channels]], [channels])
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastChannelAttentionLoss = NumOps.Zero;

        _channels = channels;
        _reducedChannels = channels / reductionRatio;
        _firstActivation = firstActivation ?? new ReLUActivation<T>();
        _secondActivation = secondActivation ?? new SigmoidActivation<T>();

        _weights1 = new Tensor<T>([_channels, _reducedChannels]);
        _bias1 = new Tensor<T>([_reducedChannels]);
        _weights2 = new Tensor<T>([_reducedChannels, _channels]);
        _bias2 = new Tensor<T>([_channels]);

        SparsityWeight = NumOps.FromDouble(0.0001);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SqueezeAndExcitationLayer{T}"/> class with vector activation functions.
    /// </summary>
    /// <param name="channels">The number of input and output channels.</param>
    /// <param name="reductionRatio">The ratio by which to reduce the number of channels in the bottleneck.</param>
    /// <param name="firstVectorActivation">The vector activation function for the first fully connected layer. Defaults to ReLU if not specified.</param>
    /// <param name="secondVectorActivation">The vector activation function for the second fully connected layer. Defaults to Sigmoid if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a Squeeze-and-Excitation layer with the specified number of channels and reduction ratio.
    /// It uses vector activation functions, which operate on entire vectors rather than individual elements.
    /// The reduction ratio determines how much the channel dimension is compressed in the bottleneck.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the previous one, but uses vector activations.
    ///
    /// Vector activations:
    /// - Process entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements
    /// - Allow for more complex transformations
    ///
    /// This version is useful when you need more sophisticated processing that considers
    /// how different features relate to each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public SqueezeAndExcitationLayer(int channels, int reductionRatio,
        IVectorActivationFunction<T>? firstVectorActivation = null,
        IVectorActivationFunction<T>? secondVectorActivation = null)
        : base([[channels]], [channels])
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);
        _lastChannelAttentionLoss = NumOps.Zero;

        _channels = channels;
        _reducedChannels = channels / reductionRatio;
        _firstVectorActivation = firstVectorActivation ?? new ReLUActivation<T>();
        _secondVectorActivation = secondVectorActivation ?? new SigmoidActivation<T>();

        _weights1 = new Tensor<T>([_channels, _reducedChannels]);
        _bias1 = new Tensor<T>([_reducedChannels]);
        _weights2 = new Tensor<T>([_reducedChannels, _channels]);
        _bias2 = new Tensor<T>([_channels]);

        SparsityWeight = NumOps.FromDouble(0.0001);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes the weights and biases of the layer with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes all weights and biases of the layer with small random values scaled by 0.1.
    /// Proper initialization is important for training neural networks effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets the starting values for the layer's parameters.
    /// 
    /// Good initialization is important because:
    /// - Starting with all zeros would make learning impossible
    /// - Starting with values that are too large can slow down or prevent learning
    /// - Small random values help the network start learning efficiently
    /// 
    /// The values are set to small random numbers (between -0.05 and 0.05) to give the
    /// network a good starting point for learning.
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        T scale = NumOps.FromDouble(0.1);
        InitializeTensor2D(_weights1, scale);
        InitializeTensor2D(_weights2, scale);
        InitializeTensor1D(_bias1, scale);
        InitializeTensor1D(_bias2, scale);
    }

    /// <summary>
    /// Initializes a 2D tensor with small random values scaled by the specified factor.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    private void InitializeTensor2D(Tensor<T> tensor, T scale)
    {
        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                tensor[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Initializes a 1D tensor with small random values scaled by the specified factor.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    private void InitializeTensor1D(Tensor<T> tensor, T scale)
    {
        int length = tensor.Shape[0];
        for (int i = 0; i < length; i++)
        {
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Performs the forward pass of the Squeeze-and-Excitation layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after Squeeze-and-Excitation processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the Squeeze-and-Excitation layer. It first applies global average pooling
    /// to "squeeze" spatial information into a channel descriptor. Then it passes this descriptor through two fully connected
    /// layers with activations to produce channel-wise scaling factors. Finally, it multiplies the original input by these
    /// scaling factors to recalibrate the feature maps.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through the Squeeze-and-Excitation steps.
    /// 
    /// The process works in three main steps:
    /// 
    /// 1. Squeeze: Compresses all spatial information into a single value per channel
    ///    - For each channel, all values are averaged together
    ///    - This creates a "summary" of each feature type
    /// 
    /// 2. Excitation: Determines the importance of each channel
    ///    - The summary passes through two neural layers with activations
    ///    - This produces an "importance score" between 0 and 1 for each channel
    /// 
    /// 3. Scaling: Adjusts the original input based on importance
    ///    - Each feature map is multiplied by its importance score
    ///    - Important features are kept or enhanced
    ///    - Less important features are reduced
    /// 
    /// This helps the network focus attention on the most useful features for the current input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        // height/width not needed for Engine operations

        // 1. Squeeze: Global Average Pooling [B, H, W, C] -> [B, C]
        // Use Engine.ReduceMean over spatial dims (1, 2)
        var squeezed = Engine.ReduceMean(input, new[] { 1, 2 }, keepDims: false);
        _lastSqueezed = squeezed;

        // 2. Excitation: FC1 + Activation
        // squeezed: [batchSize, channels], weights1: [channels, reducedChannels]
        var fc1Output = Engine.TensorMatMul(squeezed, _weights1);
        var fc1BiasReshaped = _bias1.Reshape(1, _reducedChannels);
        var fc1Biased = Engine.TensorBroadcastAdd(fc1Output, fc1BiasReshaped);
        _lastFc1Biased = fc1Biased;

        var activated1 = ApplyTensorActivation(fc1Biased, isFirstActivation: true);
        _lastFc1Activated = activated1;

        // Excitation: FC2 + Activation
        // activated1: [batchSize, reducedChannels], weights2: [reducedChannels, channels]
        var fc2Output = Engine.TensorMatMul(activated1, _weights2);
        var fc2BiasReshaped = _bias2.Reshape(1, _channels);
        var fc2Biased = Engine.TensorBroadcastAdd(fc2Output, fc2BiasReshaped);
        _lastFc2Biased = fc2Biased;

        var excitation = ApplyTensorActivation(fc2Biased, isFirstActivation: false);

        // Cache excitation weights for auxiliary loss computation
        _lastExcitationWeights = excitation;

        // 3. Scale: input * excitation
        // input: [B, H, W, C], excitation: [B, C]
        // Reshape excitation to [B, 1, 1, C] for broadcasting
        var excitationReshaped = excitation.Reshape(batchSize, 1, 1, _channels);

        var output = Engine.TensorBroadcastMultiply(input, excitationReshaped);

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Applies the appropriate activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to apply the activation to.</param>
    /// <param name="isFirstActivation">Indicates whether to use the first or second activation function.</param>
    /// <returns>The tensor after applying the activation function.</returns>
    private Tensor<T> ApplyTensorActivation(Tensor<T> input, bool isFirstActivation)
    {
        if (isFirstActivation)
        {
            if (_firstVectorActivation != null) return _firstVectorActivation.Activate(input);
            if (_firstActivation != null) return _firstActivation.Activate(input);
        }
        else
        {
            if (_secondVectorActivation != null) return _secondVectorActivation.Activate(input);
            if (_secondActivation != null) return _secondActivation.Activate(input);
        }

        // If no activation function is set, return the input as is
        return input;
    }

    /// <summary>
    /// Applies the derivative of the activation function for backpropagation.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="isFirstActivation">Indicates whether to use the first or second activation function.</param>
    /// <returns>The tensor with derivatives applied.</returns>
    private Tensor<T> ApplyTensorActivationDerivative(Tensor<T> input, bool isFirstActivation)
    {
        int rows = input.Shape[0];
        int cols = input.Shape[1];
        var result = new Tensor<T>(input.Shape);

        if (isFirstActivation)
        {
            if (_firstVectorActivation != null)
            {
                for (int i = 0; i < rows; i++)
                {
                    var row = new Vector<T>(cols);
                    for (int j = 0; j < cols; j++)
                        row[j] = input[i, j];
                    var gradMatrix = _firstVectorActivation.Derivative(row);
                    for (int j = 0; j < cols; j++)
                        result[i, j] = gradMatrix[j, j]; // Diagonal element
                }
                return result;
            }
            else if (_firstActivation != null)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        result[i, j] = _firstActivation.Derivative(input[i, j]);
                    }
                }
                return result;
            }
        }
        else
        {
            if (_secondVectorActivation != null)
            {
                for (int i = 0; i < rows; i++)
                {
                    var row = new Vector<T>(cols);
                    for (int j = 0; j < cols; j++)
                        row[j] = input[i, j];
                    var gradMatrix = _secondVectorActivation.Derivative(row);
                    for (int j = 0; j < cols; j++)
                        result[i, j] = gradMatrix[j, j]; // Diagonal element
                }
                return result;
            }
            else if (_secondActivation != null)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        result[i, j] = _secondActivation.Derivative(input[i, j]);
                    }
                }
                return result;
            }
        }

        // If no activation function, derivative is 1
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = NumOps.One;
            }
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass of the Squeeze-and-Excitation layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the Squeeze-and-Excitation layer, which is used during training to propagate
    /// error gradients back through the network. It calculates gradients for the input and for all trainable parameters
    /// (weights and biases).
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change (outputGradient)
    /// 2. It calculates how the original input should change to reduce error (inputGradient)
    /// 3. It calculates how its internal weights and biases should change to reduce error
    /// 
    /// This process follows the chain rule of calculus, working backward from the output to the input.
    /// It's essential for the "learning" part of deep learning, allowing the network to gradually
    /// improve its performance based on examples.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }


    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        // Build computation graph mirroring Forward
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // 1. Squeeze: Global Average Pooling
        // Input [B, H, W, C] -> [B, C]
        var axes = new int[] { 1, 2 };
        var squeezed = Autodiff.TensorOperations<T>.ReduceMean(inputNode, axes, keepDims: false);

        // 2. Excitation FC1
        var w1Node = Autodiff.TensorOperations<T>.Variable(_weights1, "w1", requiresGradient: true);
        var b1Node = Autodiff.TensorOperations<T>.Variable(_bias1, "b1", requiresGradient: true);
        var fc1 = Autodiff.TensorOperations<T>.MatrixMultiply(squeezed, w1Node);

        // Broadcast bias (assumed supported by Add)
        var fc1Biased = Autodiff.TensorOperations<T>.Add(fc1, b1Node);

        // Activation 1
        var act1 = ApplyActivationToGraphNode(fc1Biased, true);

        // Excitation FC2
        var w2Node = Autodiff.TensorOperations<T>.Variable(_weights2, "w2", requiresGradient: true);
        var b2Node = Autodiff.TensorOperations<T>.Variable(_bias2, "b2", requiresGradient: true);
        var fc2 = Autodiff.TensorOperations<T>.MatrixMultiply(act1, w2Node);
        var fc2Biased = Autodiff.TensorOperations<T>.Add(fc2, b2Node);

        // Activation 2
        var excitation = ApplyActivationToGraphNode(fc2Biased, false);

        // 3. Scale
        // excitation [B, C] -> [B, 1, 1, C]
        var reshapeShape = new int[] { batchSize, 1, 1, _channels };
        var excitationReshaped = Autodiff.TensorOperations<T>.Reshape(excitation, reshapeShape);

        var outputNode = Autodiff.TensorOperations<T>.ElementwiseMultiply(inputNode, excitationReshaped);

        // Backward
        outputNode.Gradient = outputGradient;

        // Inline topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients
        if (w1Node.Gradient != null) _weights1Gradient = w1Node.Gradient;
        if (b1Node.Gradient != null) _bias1Gradient = b1Node.Gradient;
        if (w2Node.Gradient != null) _weights2Gradient = w2Node.Gradient;
        if (b2Node.Gradient != null) _bias2Gradient = b2Node.Gradient;

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }

    private Autodiff.ComputationNode<T> ApplyActivationToGraphNode(Autodiff.ComputationNode<T> input, bool isFirst)
    {
        if (isFirst)
        {
            if (_firstVectorActivation != null && _firstVectorActivation.SupportsJitCompilation)
                return _firstVectorActivation.ApplyToGraph(input);
            if (_firstActivation != null && _firstActivation.SupportsJitCompilation)
                return _firstActivation.ApplyToGraph(input);
        }
        else
        {
            if (_secondVectorActivation != null && _secondVectorActivation.SupportsJitCompilation)
                return _secondVectorActivation.ApplyToGraph(input);
            if (_secondActivation != null && _secondActivation.SupportsJitCompilation)
                return _secondActivation.ApplyToGraph(input);
        }
        return input;
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastExcitationWeights == null || _lastSqueezed == null || _lastFc1Biased == null || _lastFc1Activated == null || _lastFc2Biased == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        var inputGradientDirect = Engine.TensorMultiply(outputGradient, _lastExcitationWeights.Reshape(batchSize, 1, 1, _channels));

        var excitationGradientSpatial = Engine.TensorMultiply(outputGradient, _lastInput);
        var excitationGradient = Engine.ReduceSum(excitationGradientSpatial, new[] { 1, 2 }, keepDims: false);

        var secondActivationDerivative = ApplyTensorActivationDerivative(_lastFc2Biased, isFirstActivation: false);
        var fc2OutputGradient = Engine.TensorMultiply(excitationGradient, secondActivationDerivative);

        _weights2Gradient = Engine.TensorMatMul(Engine.TensorTranspose(_lastFc1Activated), fc2OutputGradient);
        _bias2Gradient = Engine.ReduceSum(fc2OutputGradient, new[] { 0 }, keepDims: false);

        var fc1OutputGradient = Engine.TensorMatMul(fc2OutputGradient, Engine.TensorTranspose(_weights2));

        var firstActivationDerivative = ApplyTensorActivationDerivative(_lastFc1Biased, isFirstActivation: true);
        var fc1Gradient = Engine.TensorMultiply(fc1OutputGradient, firstActivationDerivative);

        _weights1Gradient = Engine.TensorMatMul(Engine.TensorTranspose(_lastSqueezed), fc1Gradient);
        _bias1Gradient = Engine.ReduceSum(fc1Gradient, new[] { 0 }, keepDims: false);

        var squeezedGradient = Engine.TensorMatMul(fc1Gradient, Engine.TensorTranspose(_weights1));
        var squeezeBackprop = Engine.ReduceMeanBackward(squeezedGradient, _lastInput.Shape, new[] { 1, 2 });

        var inputGradient = Engine.TensorAdd(inputGradientDirect, squeezeBackprop);
        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls the size of the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when trying to update parameters before calculating gradients.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer based on the gradients calculated during the backward pass.
    /// The learning rate controls the size of the updates, with larger values leading to faster but potentially less stable learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the layer's weights and biases to improve performance.
    /// 
    /// During training:
    /// - The backward pass calculates how each parameter should change to reduce errors
    /// - This method applies those changes to the actual parameters
    /// - The learning rate controls how big each adjustment is
    /// 
    /// Think of it like learning to ride a bike:
    /// - If you make very small adjustments (small learning rate), you learn slowly but steadily
    /// - If you make large adjustments (large learning rate), you might learn faster but risk overcorrecting
    /// 
    /// This process of gradual adjustment is how neural networks "learn" from examples.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weights1Gradient == null || _bias1Gradient == null || _weights2Gradient == null || _bias2Gradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var w1Update = Engine.TensorMultiplyScalar(_weights1Gradient, learningRate);
        var b1Update = Engine.TensorMultiplyScalar(_bias1Gradient, learningRate);
        var w2Update = Engine.TensorMultiplyScalar(_weights2Gradient, learningRate);
        var b2Update = Engine.TensorMultiplyScalar(_bias2Gradient, learningRate);

        _weights1 = Engine.TensorSubtract(_weights1, w1Update);
        _bias1 = Engine.TensorSubtract(_bias1, b1Update);
        _weights2 = Engine.TensorSubtract(_weights2, w2Update);
        _bias2 = Engine.TensorSubtract(_bias2, b2Update);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) of the layer and combines them into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all weights and biases from both fully connected layers
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _weights1.Shape[0] * _weights1.Shape[1] +
                          _bias1.Shape[0] +
                          _weights2.Shape[0] * _weights2.Shape[1] +
                          _bias2.Shape[0];

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy weights1
        for (int i = 0; i < _weights1.Shape[0]; i++)
        {
            for (int j = 0; j < _weights1.Shape[1]; j++)
            {
                parameters[index++] = _weights1[i, j];
            }
        }

        // Copy bias1
        for (int i = 0; i < _bias1.Shape[0]; i++)
        {
            parameters[index++] = _bias1[i];
        }

        // Copy weights2
        for (int i = 0; i < _weights2.Shape[0]; i++)
        {
            for (int j = 0; j < _weights2.Shape[1]; j++)
            {
                parameters[index++] = _weights2[i, j];
            }
        }

        // Copy bias2
        for (int i = 0; i < _bias2.Shape[0]; i++)
        {
            parameters[index++] = _bias2[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters (weights and biases) of the layer from a single vector.
    /// This is useful for loading saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The values are copied back into the layer's weights and biases
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _weights1.Shape[0] * _weights1.Shape[1] +
                          _bias1.Shape[0] +
                          _weights2.Shape[0] * _weights2.Shape[1] +
                          _bias2.Shape[0];

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weights1
        for (int i = 0; i < _weights1.Shape[0]; i++)
        {
            for (int j = 0; j < _weights1.Shape[1]; j++)
            {
                _weights1[i, j] = parameters[index++];
            }
        }

        // Set bias1
        for (int i = 0; i < _bias1.Shape[0]; i++)
        {
            _bias1[i] = parameters[index++];
        }

        // Set weights2
        for (int i = 0; i < _weights2.Shape[0]; i++)
        {
            for (int j = 0; j < _weights2.Shape[1]; j++)
            {
                _weights2[i, j] = parameters[index++];
            }
        }

        // Set bias2
        for (int i = 0; i < _bias2.Shape[0]; i++)
        {
            _bias2[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the Squeeze-and-Excitation layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the Squeeze-and-Excitation layer, including the cached inputs and outputs
    /// from the forward pass and the gradients calculated during the backward pass. This is useful when starting to process
    /// a new input after training or when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Calculated gradients are cleared
    /// - The layer forgets any information from previous inputs
    ///
    /// This is important for:
    /// - Processing a new, unrelated input
    /// - Starting a new training epoch
    /// - Preventing information from one input affecting another
    ///
    /// Think of it like wiping a whiteboard clean before starting a new problem.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastSqueezed = null;
        _lastFc1Biased = null;
        _lastFc1Activated = null;
        _lastFc2Biased = null;
        _weights1Gradient = null;
        _bias1Gradient = null;
        _weights2Gradient = null;
        _bias2Gradient = null;
        _lastExcitationWeights = null;
    }

    /// <summary>
    /// Computes the auxiliary loss for this layer based on channel attention regularization.
    /// </summary>
    /// <returns>The computed auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes a channel attention regularization loss. In a full implementation, this would encourage
    /// balanced channel attention by penalizing extreme attention values (all attention on one channel or uniform
    /// attention across all channels). The regularization can use L2 norm or entropy-based measures.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates a penalty to encourage balanced feature importance.
    ///
    /// Channel attention regularization:
    /// - Prevents the layer from relying too heavily on specific channels
    /// - Encourages the network to use information from multiple features
    /// - Helps create more robust and generalizable models
    ///
    /// Why this is useful:
    /// - In complex tasks, multiple types of features are usually important
    /// - Over-relying on one type of feature can lead to poor generalization
    /// - Balanced attention helps the network learn richer representations
    ///
    /// Example: In image classification, instead of only looking at edges (one channel),
    /// the network should also consider colors, textures, and shapes (other channels).
    ///
    /// <b>Note:</b> This is a placeholder implementation. For full functionality, the layer would need to
    /// cache the excitation weights (channel attention scores) during the forward pass. The formula would
    /// compute a regularization term based on these attention weights, such as:
    /// - L2 regularization: L = ||excitation||²
    /// - Entropy regularization: L = -Σ(p * log(p)) for normalized excitation weights
    /// - Variance penalty: encouraging variance in attention across channels
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastExcitationWeights == null)
        {
            _lastChannelAttentionLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute L2 regularization on excitation weights
        // This penalizes large excitation values and encourages sparse channel attention
        T attentionLoss = NumOps.Zero;
        int batchSize = _lastExcitationWeights.Shape[0];
        int channels = _lastExcitationWeights.Shape[1];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T weight = _lastExcitationWeights[b, c];
                // L2 regularization: sum of squared weights
                attentionLoss = NumOps.Add(attentionLoss, NumOps.Multiply(weight, weight));
            }
        }

        // Average across batch and channels
        int totalElements = batchSize * channels;
        attentionLoss = NumOps.Divide(attentionLoss, NumOps.FromDouble(totalElements));

        // Store unweighted loss for diagnostics
        _lastChannelAttentionLoss = attentionLoss;

        // Return weighted auxiliary loss
        return NumOps.Multiply(AuxiliaryLossWeight, attentionLoss);
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about the auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method returns diagnostic information that can be used to monitor the auxiliary loss during training.
    /// The diagnostics include the total channel attention loss, the weight applied to it, and whether auxiliary loss is enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information to help you understand how the auxiliary loss is working.
    ///
    /// The diagnostics show:
    /// - TotalChannelAttentionLoss: The computed penalty for imbalanced channel attention
    /// - ChannelAttentionWeight: How much this penalty affects the overall training
    /// - UseChannelAttention: Whether this penalty is currently enabled
    ///
    /// You can use this information to:
    /// - Monitor if channel attention is becoming more balanced over time
    /// - Debug training issues related to feature selection
    /// - Understand which features the network prioritizes
    ///
    /// Example: If TotalChannelAttentionLoss is high, it might indicate that the network is over-relying
    /// on specific channels, which could be a sign of overfitting or poor feature diversity.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalChannelAttentionLoss", System.Convert.ToString(_lastChannelAttentionLoss) ?? "0" },
            { "ChannelAttentionWeight", System.Convert.ToString(AuxiliaryLossWeight) ?? "0.01" },
            { "UseChannelAttention", UseAuxiliaryLoss.ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_weights1 == null || _weights2 == null || _bias1 == null || _bias2 == null)
            throw new InvalidOperationException("Layer weights not initialized. Initialize the layer before compiling.");

        // Create symbolic input tensor with batch dimension
        // SE blocks operate on [batch, height, width, channels] tensors
        var symbolicInput = new Tensor<T>(new int[] { 1, 1, 1, _channels });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // Squeeze: Global Average Pooling across spatial dimensions
        var squeezed = TensorOperations<T>.ReduceMean(inputNode, axes: new[] { 1, 2 }, keepDims: false);

        // Excitation: First fully connected layer (weights and biases are already Tensor<T>)
        var weights1Node = TensorOperations<T>.Constant(_weights1, $"{namePrefix}se_weights1");
        var bias1Node = TensorOperations<T>.Constant(_bias1, $"{namePrefix}se_bias1");

        var fc1Output = TensorOperations<T>.MatrixMultiply(squeezed, weights1Node);
        fc1Output = TensorOperations<T>.Add(fc1Output, bias1Node);

        // Apply first activation (default: ReLU)
        if (_firstActivation != null && _firstActivation.SupportsJitCompilation)
        {
            fc1Output = _firstActivation.ApplyToGraph(fc1Output);
        }
        else if (_firstVectorActivation == null)
        {
            fc1Output = TensorOperations<T>.ReLU(fc1Output);
        }

        // Excitation: Second fully connected layer (weights and biases are already Tensor<T>)
        var weights2Node = TensorOperations<T>.Constant(_weights2, $"{namePrefix}se_weights2");
        var bias2Node = TensorOperations<T>.Constant(_bias2, $"{namePrefix}se_bias2");

        var fc2Output = TensorOperations<T>.MatrixMultiply(fc1Output, weights2Node);
        fc2Output = TensorOperations<T>.Add(fc2Output, bias2Node);

        // Apply second activation (default: Sigmoid)
        if (_secondActivation != null && _secondActivation.SupportsJitCompilation)
        {
            fc2Output = _secondActivation.ApplyToGraph(fc2Output);
        }
        else if (_secondVectorActivation == null)
        {
            fc2Output = TensorOperations<T>.Sigmoid(fc2Output);
        }

        // Scale: Multiply input by excitation weights (with broadcasting)
        // fc2Output has shape [batch, channels], inputNode has shape [batch, height, width, channels]
        // ElementwiseMultiply should handle broadcasting automatically
        var scaledOutput = TensorOperations<T>.ElementwiseMultiply(inputNode, fc2Output);

        return scaledOutput;
    }

    public override bool SupportsJitCompilation =>
        _weights1 != null && _weights2 != null && _bias1 != null && _bias2 != null;
}
