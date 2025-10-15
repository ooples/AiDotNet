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
public class SqueezeAndExcitationLayer<T> : LayerBase<T>
{
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
    private Matrix<T> _weights1 = default!;

    /// <summary>
    /// The bias values for the first fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the bias values that are added after applying the _weights1 transformation.
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
    private Vector<T> _bias1 = default!;

    /// <summary>
    /// The weights for the second fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix contains the weights that transform from the reduced representation back to the original channel dimensions.
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
    private Matrix<T> _weights2 = default!;

    /// <summary>
    /// The bias values for the second fully connected layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the bias values that are added after applying the _weights2 transformation.
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
    private Vector<T> _bias2 = default!;

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
    private Matrix<T>? _weights1Gradient;

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
    private Vector<T>? _bias1Gradient;

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
    private Matrix<T>? _weights2Gradient;

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
    private Vector<T>? _bias2Gradient;

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
    /// Vector<double> activations:
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
        _channels = channels;
        _reducedChannels = channels / reductionRatio;
        _firstActivation = firstActivation ?? new ReLUActivation<T>();
        _secondActivation = secondActivation ?? new SigmoidActivation<T>();

        _weights1 = new Matrix<T>(_channels, _reducedChannels);
        _bias1 = new Vector<T>(_reducedChannels);
        _weights2 = new Matrix<T>(_reducedChannels, _channels);
        _bias2 = new Vector<T>(_channels);

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
    /// Vector<double> activations:
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
        _channels = channels;
        _reducedChannels = channels / reductionRatio;
        _firstVectorActivation = firstVectorActivation ?? new ReLUActivation<T>();
        _secondVectorActivation = secondVectorActivation ?? new SigmoidActivation<T>();

        _weights1 = new Matrix<T>(_channels, _reducedChannels);
        _bias1 = new Vector<T>(_reducedChannels);
        _weights2 = new Matrix<T>(_reducedChannels, _channels);
        _bias2 = new Vector<T>(_channels);

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
        InitializeMatrix(_weights1, NumOps.FromDouble(0.1));
        InitializeMatrix(_weights2, NumOps.FromDouble(0.1));
        InitializeVector(_bias1, NumOps.FromDouble(0.1));
        InitializeVector(_bias2, NumOps.FromDouble(0.1));
    }

    /// <summary>
    /// Initializes a matrix with small random values scaled by the specified factor.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given matrix with random values between -0.5 and 0.5, scaled by the specified factor.
    /// This is a common practice in neural network initialization to ensure that weights are neither too large nor too small.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets random starting values for a group of parameters.
    /// 
    /// It works by:
    /// - Generating random numbers between -0.5 and 0.5
    /// - Multiplying them by a scaling factor (typically small, like 0.1)
    /// - Storing these values in the matrix
    /// 
    /// This creates a good balance of positive and negative values that helps the network start learning.
    /// </para>
    /// </remarks>
    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Initializes a vector with small random values scaled by the specified factor.
    /// </summary>
    /// <param name="vector">The vector to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given vector with random values between -0.5 and 0.5, scaled by the specified factor.
    /// This approach helps ensure that the initial biases are neither too large nor too small.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets random starting values for a list of parameters.
    /// 
    /// Similar to initializing a matrix, it:
    /// - Generates random numbers between -0.5 and 0.5
    /// - Multiplies them by a scaling factor
    /// - Stores these values in the vector
    /// 
    /// Biases typically require similar initialization to weights to ensure balanced starting conditions.
    /// </para>
    /// </remarks>
    private void InitializeVector(Vector<T> vector, T scale)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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
        int height = input.Shape[1];
        int width = input.Shape[2];

        // Squeeze: Global Average Pooling
        var squeezed = new Matrix<T>(batchSize, _channels);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, input[b, h, w, c]);
                    }
                }

                squeezed[b, c] = NumOps.Divide(sum, NumOps.FromDouble(height * width));
            }
        }

        // Excitation: Two FC layers with activation
        var excitation1 = squeezed.Multiply(_weights1).AddVectorToEachRow(_bias1);
        excitation1 = ApplyActivation(excitation1, isFirstActivation: true);
        var excitation2 = excitation1.Multiply(_weights2).AddVectorToEachRow(_bias2);
        var excitation = ApplyActivation(excitation2, isFirstActivation: false);

        // Scale the input
        var output = new Tensor<T>(input.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < _channels; c++)
                    {
                        output[b, h, w, c] = NumOps.Multiply(input[b, h, w, c], excitation[b, c]);
                    }
                }
            }
        }

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Applies the appropriate activation function to the input matrix.
    /// </summary>
    /// <param name="input">The input matrix to apply the activation to.</param>
    /// <param name="isFirstActivation">Indicates whether to use the first or second activation function.</param>
    /// <returns>The matrix after applying the activation function.</returns>
    /// <remarks>
    /// <para>
    /// This method applies either the first or second activation function to the input matrix, depending on the value
    /// of the isFirstActivation parameter. It handles both scalar and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies a mathematical function to transform the values.
    /// 
    /// Activation functions:
    /// - Add non-linearity to the network (making it able to learn complex patterns)
    /// - Transform values in specific ways (like keeping only positive values or squashing values between 0 and 1)
    /// 
    /// This method chooses between:
    /// - The first activation (typically ReLU) if processing the first layer
    /// - The second activation (typically Sigmoid) if processing the second layer
    /// 
    /// It also handles two types of activations:
    /// - Vector<double> activations that process entire rows at once
    /// - Scalar activations that process each value individually
    /// </para>
    /// </remarks>
    private Matrix<T> ApplyActivation(Matrix<T> input, bool isFirstActivation)
    {
        if (isFirstActivation)
        {
            if (_firstVectorActivation != null)
            {
                return ApplyVectorActivation(input, _firstVectorActivation);
            }
            else if (_firstActivation != null)
            {
                return ApplyScalarActivation(input, _firstActivation);
            }
        }
        else
        {
            if (_secondVectorActivation != null)
            {
                return ApplyVectorActivation(input, _secondVectorActivation);
            }
            else if (_secondActivation != null)
            {
                return ApplyScalarActivation(input, _secondActivation);
            }
        }

        // If no activation function is set, return the input as is
        return input;
    }

    /// <summary>
    /// Applies a vector activation function to each row of the input matrix.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <param name="activationFunction">The vector activation function to apply.</param>
    /// <returns>A new matrix with the activation function applied to each row.</returns>
    /// <remarks>
    /// <para>
    /// This method applies a vector activation function to each row of the input matrix. Vector<double> activation functions
    /// operate on entire vectors rather than individual elements, allowing them to capture relationships between elements.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms entire rows of values at once.
    /// 
    /// Vector<double> activation functions:
    /// - Process an entire row of numbers together
    /// - Can consider relationships between different elements
    /// - Are more sophisticated than processing each number separately
    /// 
    /// For example, a vector activation might:
    /// - Normalize the values so they sum to 1 (useful for attention mechanisms)
    /// - Apply different transformations based on the pattern of values
    /// - Calculate how different elements relate to each other
    /// </para>
    /// </remarks>
    private static Matrix<T> ApplyVectorActivation(Matrix<T> input, IVectorActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            Vector<T> row = input.GetRow(i);
            Vector<T> activatedRow = activationFunction.Activate(row);
            result.SetRow(i, activatedRow);
        }

        return result;
    }

    /// <summary>
    /// Applies a scalar activation function to each element of the input matrix.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <param name="activationFunction">The scalar activation function to apply.</param>
    /// <returns>A new matrix with the activation function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This method applies a scalar activation function to each element of the input matrix. Scalar activation functions
    /// operate on individual elements independently.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms each value individually.
    /// 
    /// Scalar activation functions:
    /// - Process each number separately
    /// - Apply the same transformation to each value
    /// - Are simpler but often effective
    /// 
    /// Common scalar activations include:
    /// - ReLU: Keeps positive values unchanged, sets negative values to zero
    /// - Sigmoid: Squashes any value to be between 0 and 1
    /// - Tanh: Squashes any value to be between -1 and 1
    /// 
    /// For example, applying ReLU to [2, -3, 4, -1] would result in [2, 0, 4, 0].
    /// </para>
    /// </remarks>
    private static Matrix<T> ApplyScalarActivation(Matrix<T> input, IActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                result[i, j] = activationFunction.Activate(input[i, j]);
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
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int height = _lastInput.Shape[1];
        int width = _lastInput.Shape[2];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _weights1Gradient = new Matrix<T>(_weights1.Rows, _weights1.Columns);
        _bias1Gradient = new Vector<T>(_bias1.Length);
        _weights2Gradient = new Matrix<T>(_weights2.Rows, _weights2.Columns);
        _bias2Gradient = new Vector<T>(_bias2.Length);

        // Calculate gradients for scaling and input
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < _channels; c++)
                    {
                        T scaleFactor = NumOps.Divide(_lastOutput[b, h, w, c], _lastInput[b, h, w, c]);
                        inputGradient[b, h, w, c] = NumOps.Multiply(outputGradient[b, h, w, c], scaleFactor);
                    }
                }
            }
        }

        // Calculate gradients for excitation
        var excitationGradient = new Matrix<T>(batchSize, _channels);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _channels; c++)
            {
                T sum = NumOps.Zero;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(outputGradient[b, h, w, c], _lastInput[b, h, w, c]));
                    }
                }
                excitationGradient[b, c] = sum;
            }
        }

        // Backpropagate through FC layers
        var excitation2Gradient = excitationGradient;
        if (_secondVectorActivation != null)
        {
            excitation2Gradient = ApplyVectorActivationGradient(excitationGradient, _secondVectorActivation);
        }
        else if (_secondActivation != null)
        {
            excitation2Gradient = ApplyScalarActivationGradient(excitationGradient, _secondActivation);
        }

        _weights2Gradient = excitation2Gradient.Transpose().Multiply(excitationGradient);
        _bias2Gradient = excitationGradient.SumColumns();

        var excitation1Gradient = excitation2Gradient.Multiply(_weights2.Transpose());
        if (_firstVectorActivation != null)
        {
            excitation1Gradient = ApplyVectorActivationGradient(excitation1Gradient, _firstVectorActivation);
        }
        else if (_firstActivation != null)
        {
            excitation1Gradient = ApplyScalarActivationGradient(excitation1Gradient, _firstActivation);
        }

        _weights1Gradient = excitation1Gradient.Transpose().Multiply(excitationGradient);
        _bias1Gradient = excitation1Gradient.SumColumns();

        return inputGradient;
    }

    /// <summary>
    /// Applies the derivative of a vector activation function for backpropagation.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <param name="activationFunction">The vector activation function whose derivative to apply.</param>
    /// <returns>A new matrix with the activation function's derivative applied.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of a vector activation function to the input matrix, which is necessary during
    /// the backward pass of backpropagation. The derivative of the activation function determines how error gradients
    /// flow backward through the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes should flow through vector activations.
    /// 
    /// During backpropagation (the learning process):
    /// - We need to know how changes to the output affect the input of each function
    /// - This requires the derivative (rate of change) of the activation function
    /// - For vector activations, this is more complex as each output can depend on multiple inputs
    /// 
    /// This method:
    /// 1. Calculates the derivative matrix for each row
    /// 2. Extracts the diagonal elements (which represent direct effects)
    /// 3. Multiplies the input gradients by these derivatives
    /// 
    /// This helps the network adjust its parameters correctly during learning.
    /// </para>
    /// </remarks>
    private static Matrix<T> ApplyVectorActivationGradient(Matrix<T> input, IVectorActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            Vector<T> row = input.GetRow(i);
            Matrix<T> gradientMatrix = activationFunction.Derivative(row);
            Vector<T> gradientDiagonal = gradientMatrix.Diagonal();
        
            // Element-wise multiplication of the input row with the gradient diagonal
            Vector<T> gradientRow = row.ElementwiseMultiply(gradientDiagonal);
        
            result.SetRow(i, gradientRow);
        }

        return result;
    }

    /// <summary>
    /// Applies the derivative of a scalar activation function for backpropagation.
    /// </summary>
    /// <param name="input">The input matrix.</param>
    /// <param name="activationFunction">The scalar activation function whose derivative to apply.</param>
    /// <returns>A new matrix with the activation function's derivative applied.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of a scalar activation function to the input matrix, which is necessary during
    /// the backward pass of backpropagation. The derivative of the activation function determines how error gradients
    /// flow backward through the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes should flow through scalar activations.
    /// 
    /// Similar to the vector version, but simpler:
    /// - Each output only depends on a single input
    /// - We calculate the derivative for each value separately
    /// 
    /// For example:
    /// - ReLU's derivative is 1 for positive inputs and 0 for negative inputs
    /// - This means errors flow unchanged through positive values, but stop at negative values
    /// - This helps the network focus on updating parameters that have a positive effect
    /// 
    /// This is a key part of how neural networks learn from their mistakes.
    /// </para>
    /// </remarks>
    private static Matrix<T> ApplyScalarActivationGradient(Matrix<T> input, IActivationFunction<T> activationFunction)
    {
        var result = new Matrix<T>(input.Rows, input.Columns);
        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < input.Columns; j++)
            {
                result[i, j] = activationFunction.Derivative(input[i, j]);
            }
        }

        return result;
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

        _weights1 = _weights1.Subtract(_weights1Gradient.Multiply(learningRate));
        _bias1 = _bias1.Subtract(_bias1Gradient.Multiply(learningRate));
        _weights2 = _weights2.Subtract(_weights2Gradient.Multiply(learningRate));
        _bias2 = _bias2.Subtract(_bias2Gradient.Multiply(learningRate));
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
        int totalParams = _weights1.Rows * _weights1.Columns +
                          _bias1.Length +
                          _weights2.Rows * _weights2.Columns +
                          _bias2.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy weights1
        for (int i = 0; i < _weights1.Rows; i++)
        {
            for (int j = 0; j < _weights1.Columns; j++)
            {
                parameters[index++] = _weights1[i, j];
            }
        }
    
        // Copy bias1
        for (int i = 0; i < _bias1.Length; i++)
        {
            parameters[index++] = _bias1[i];
        }
    
        // Copy weights2
        for (int i = 0; i < _weights2.Rows; i++)
        {
            for (int j = 0; j < _weights2.Columns; j++)
            {
                parameters[index++] = _weights2[i, j];
            }
        }
    
        // Copy bias2
        for (int i = 0; i < _bias2.Length; i++)
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
        int totalParams = _weights1.Rows * _weights1.Columns +
                          _bias1.Length +
                          _weights2.Rows * _weights2.Columns +
                          _bias2.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set weights1
        for (int i = 0; i < _weights1.Rows; i++)
        {
            for (int j = 0; j < _weights1.Columns; j++)
            {
                _weights1[i, j] = parameters[index++];
            }
        }
    
        // Set bias1
        for (int i = 0; i < _bias1.Length; i++)
        {
            _bias1[i] = parameters[index++];
        }
    
        // Set weights2
        for (int i = 0; i < _weights2.Rows; i++)
        {
            for (int j = 0; j < _weights2.Columns; j++)
            {
                _weights2[i, j] = parameters[index++];
            }
        }
    
        // Set bias2
        for (int i = 0; i < _bias2.Length; i++)
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
        _weights1Gradient = null;
        _bias1Gradient = null;
        _weights2Gradient = null;
        _bias2Gradient = null;
    }
}