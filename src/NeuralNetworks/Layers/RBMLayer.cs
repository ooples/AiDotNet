using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Restricted Boltzmann Machine (RBM) layer for neural networks.
/// </summary>
/// <remarks>
/// <para>
/// An RBM layer is a stochastic neural network layer that learns a probability distribution over its inputs.
/// It consists of a visible layer and a hidden layer with no connections between nodes within the same layer.
/// </para>
/// <para><b>For Beginners:</b> An RBM layer is like a feature detector that can learn patterns in data.
/// 
/// Imagine you have a set of movie ratings:
/// - The visible layer represents the actual ratings
/// - The hidden layer represents abstract features (e.g., "likes action", "prefers comedy")
/// - The RBM learns to connect ratings to these abstract features
/// 
/// RBM layers are useful for:
/// - Finding underlying patterns in data
/// - Reducing the dimensionality of data
/// - Initializing weights for deep neural networks
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RBMLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the number of units in the visible layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The visible units are the input nodes of the RBM, representing observed data.
    /// This value determines the dimensionality of the input data that the RBM can process.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the visible units as the "input sensors" of the RBM.
    /// 
    /// For example, if you're working with 28x28 pixel images:
    /// - You would have 784 visible units (28 × 28 = 784)
    /// - Each visible unit represents one pixel in the image
    /// - The visible layer receives the actual data you want to analyze
    /// 
    /// This number must match the dimensionality of your input data.
    /// </para>
    /// </remarks>
    private readonly int _visibleUnits;

    /// <summary>
    /// Gets the number of units in the hidden layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hidden units are the nodes that learn to detect features or patterns in the input data.
    /// This value determines the capacity of the RBM to represent complex patterns.
    /// </para>
    /// <para><b>For Beginners:</b> Think of hidden units as "pattern detectors" that learn important features.
    /// 
    /// For example, in an image recognition task:
    /// - You might have 100-500 hidden units
    /// - Each hidden unit learns to detect a specific pattern (like edges, corners, or shapes)
    /// - More hidden units can recognize more complex patterns, but require more data to train properly
    /// 
    /// The number of hidden units is a hyperparameter you can adjust to balance between:
    /// - Too few: The RBM can't learn complex patterns
    /// - Too many: The RBM might overfit to the training data
    /// </para>
    /// </remarks>
    private readonly int _hiddenUnits;

    /// <summary>
    /// Gets or sets the weight matrix connecting visible and hidden units.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The weights tensor represents the strength of connections between visible and hidden units.
    /// Shape is [hiddenUnits, visibleUnits]. Each element W[j,i] represents the connection strength
    /// between visible unit i and hidden unit j.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the weights as "connection strengths" between units.
    ///
    /// The weight tensor:
    /// - Has dimensions [hiddenUnits × visibleUnits]
    /// - Each weight shows how strongly a visible unit influences a hidden unit
    /// - Positive weights mean "these units tend to be active together"
    /// - Negative weights mean "when one unit is active, the other tends to be inactive"
    ///
    /// During training, these weights are adjusted to capture patterns in your data.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights;

    /// <summary>
    /// Gets or sets the bias values for the visible units.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The visible biases represent the baseline activation level of each visible unit,
    /// independent of any inputs from hidden units. They allow the RBM to model
    /// the marginal distribution of the visible units.
    /// </para>
    /// <para><b>For Beginners:</b> Think of visible biases as the "default preference" of each visible unit.
    ///
    /// Visible biases:
    /// - One bias value per visible unit
    /// - Control how likely each visible unit is to be active by default
    /// - Positive bias: the unit prefers to be on (value of 1)
    /// - Negative bias: the unit prefers to be off (value of 0)
    ///
    /// For example, if most images in your dataset have dark backgrounds (pixels mostly off),
    /// the biases for those background pixels would become negative during training.
    /// </para>
    /// </remarks>
    private Tensor<T> _visibleBiases;

    /// <summary>
    /// Gets or sets the bias values for the hidden units.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The hidden biases represent the baseline activation level of each hidden unit,
    /// independent of any inputs from visible units. They allow the RBM to model
    /// the marginal distribution of the hidden units.
    /// </para>
    /// <para><b>For Beginners:</b> Think of hidden biases as the "default sensitivity" of each feature detector.
    ///
    /// Hidden biases:
    /// - One bias value per hidden unit
    /// - Control how easily each feature detector activates
    /// - Positive bias: the detector is more sensitive (activates easily)
    /// - Negative bias: the detector is less sensitive (requires stronger evidence to activate)
    ///
    /// These biases help the RBM learn features that occur with different frequencies in your data.
    /// </para>
    /// </remarks>
    private Tensor<T> _hiddenBiases;

    /// <summary>
    /// Gradient of the weights computed during backpropagation.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradient of the visible biases computed during backpropagation.
    /// </summary>
    private Tensor<T>? _visibleBiasesGradient;

    /// <summary>
    /// Gradient of the hidden biases computed during backpropagation.
    /// </summary>
    private Tensor<T>? _hiddenBiasesGradient;

    /// <summary>
    /// Stores the last input from the visible layer during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the original visible input tensor from the most recent forward pass.
    /// It is used during parameter updates in contrastive divergence training.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as saving the "before" snapshot of input data.
    ///
    /// During training:
    /// - The RBM needs to compare the original input with a reconstructed version
    /// - This field stores the original input data (the "before" snapshot)
    /// - It's used in the "positive phase" of contrastive divergence training
    ///
    /// This storage helps the RBM adjust its weights to make better reconstructions of the input data.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastVisibleInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores the last output from the hidden layer during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the hidden unit activations from the most recent forward pass.
    /// It is used during parameter updates in contrastive divergence training.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as saving what patterns the RBM detected in the input.
    ///
    /// During training:
    /// - The RBM activates hidden units based on the input
    /// - This field stores those activations (what features were detected)
    /// - It's used in the "positive phase" of contrastive divergence training
    ///
    /// This storage helps the RBM learn which features are actually present in the data
    /// versus which ones it incorrectly generates on its own.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastHiddenOutput;

    /// <summary>
    /// Stores the reconstructed visible layer activations during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the reconstructed visible unit activations during training.
    /// It is used during parameter updates in contrastive divergence training.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as the RBM's attempt to recreate the original input.
    ///
    /// During training:
    /// - After detecting features, the RBM tries to recreate the original input
    /// - This field stores that reconstruction (the "after" snapshot)
    /// - It's used in the "negative phase" of contrastive divergence training
    ///
    /// The difference between the original input and this reconstruction guides
    /// how the RBM updates its weights to improve future reconstructions.
    /// </para>
    /// </remarks>
    private Tensor<T>? _reconstructedVisible;

    /// <summary>
    /// Stores the reconstructed hidden layer activations during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the hidden unit activations based on the reconstructed visible units.
    /// It is used during parameter updates in contrastive divergence training.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as what patterns the RBM detects in its own reconstruction.
    ///
    /// During training:
    /// - After reconstructing the input, the RBM detects features in that reconstruction
    /// - This field stores those activations (what the RBM "thinks" should be in the data)
    /// - It's used in the "negative phase" of contrastive divergence training
    ///
    /// The difference between these activations and the ones from the real data helps
    /// the RBM learn to distinguish real patterns from ones it incorrectly imagines.
    /// </para>
    /// </remarks>
    private Tensor<T>? _reconstructedHidden;

    /// <summary>
    /// Initializes a new instance of the RBMLayer class with scalar activation.
    /// </summary>
    /// <param name="visibleUnits">The number of visible units.</param>
    /// <param name="hiddenUnits">The number of hidden units.</param>
    /// <param name="scalarActivation">The scalar activation function to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RBM layer with the specified number of visible and hidden units,
    /// using a scalar activation function that is applied element-wise to unit activations.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create an RBM layer with a standard activation function.
    /// 
    /// For example:
    /// ```csharp
    /// // Create an RBM with 784 visible units (28x28 image), 200 hidden units,
    /// // and sigmoid activation function
    /// var rbmLayer = new RBMLayer<float>(784, 200, new SigmoidActivation<float>());
    /// ```
    /// 
    /// The sigmoid activation is most common for RBMs because it produces probabilities
    /// between 0 and 1, which is what the RBM needs for its stochastic behavior.
    /// </para>
    /// </remarks>
    public RBMLayer(int visibleUnits, int hiddenUnits, IActivationFunction<T>? scalarActivation = null)
        : base([visibleUnits], [hiddenUnits], scalarActivation ?? new SigmoidActivation<T>())
    {
        _visibleUnits = visibleUnits;
        _hiddenUnits = hiddenUnits;
        _weights = new Tensor<T>([_hiddenUnits, _visibleUnits]);
        _visibleBiases = new Tensor<T>([_visibleUnits]);
        _hiddenBiases = new Tensor<T>([_hiddenUnits]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the RBMLayer class with vector activation.
    /// </summary>
    /// <param name="visibleUnits">The number of visible units.</param>
    /// <param name="hiddenUnits">The number of hidden units.</param>
    /// <param name="vectorActivation">The vector activation function to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RBM layer with the specified number of visible and hidden units,
    /// using a vector activation function that is applied to vectors of unit activations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a more advanced way to create an RBM that processes multiple units at once.
    /// 
    /// For example:
    /// ```csharp
    /// // Create an RBM with 784 visible units, 200 hidden units,
    /// // and a vectorized sigmoid activation function
    /// var rbmLayer = new RBMLayer<float>(784, 200, new VectorizedSigmoidActivation<float>());
    /// ```
    /// 
    /// This approach is functionally equivalent to using scalar activation, but can be more
    /// computationally efficient, especially when running on specialized hardware like GPUs.
    /// </para>
    /// </remarks>
    public RBMLayer(int visibleUnits, int hiddenUnits, IVectorActivationFunction<T>? vectorActivation = null)
        : base([visibleUnits], [hiddenUnits], vectorActivation ?? new SigmoidActivation<T>())
    {
        _visibleUnits = visibleUnits;
        _hiddenUnits = hiddenUnits;
        _weights = new Tensor<T>([_hiddenUnits, _visibleUnits]);
        _visibleBiases = new Tensor<T>([_visibleUnits]);
        _hiddenBiases = new Tensor<T>([_hiddenUnits]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the RBM layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the RBM parameters with appropriate starting values.
    /// Weights are set to small random values, while biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the RBM with starting values before training.
    /// 
    /// During initialization:
    /// - Weights are set to small random values (between -0.05 and 0.05)
    /// - Small random values help the RBM start learning gradually
    /// - Biases are set to zero (no initial preference)
    /// 
    /// Good initialization is important because:
    /// - Too large values can cause training to diverge
    /// - Too small values can cause very slow learning
    /// - The right balance helps the RBM train efficiently
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Initialize biases to zero using Fill
        _visibleBiases.Fill(NumOps.Zero);
        _hiddenBiases.Fill(NumOps.Zero);

        // Initialize weights with small random values using Engine operations
        // Create random tensor [0, 1], scale to [0, 0.1], shift to [-0.05, 0.05]
        var randomTensor = Tensor<T>.CreateRandom(_hiddenUnits, _visibleUnits);
        var scaledTensor = Engine.TensorMultiplyScalar(randomTensor, NumOps.FromDouble(0.1));
        var shiftTensor = new Tensor<T>([_hiddenUnits, _visibleUnits]);
        shiftTensor.Fill(NumOps.FromDouble(0.05));
        _weights = Engine.TensorSubtract(scaledTensor, shiftTensor);
    }

    /// <summary>
    /// Computes the forward pass of the RBM layer.
    /// </summary>
    /// <param name="input">The input tensor containing visible unit activations.</param>
    /// <returns>The output tensor containing hidden unit activations.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the RBM layer. It takes a tensor of visible unit
    /// activations, computes the probability of each hidden unit being active, and returns these
    /// probabilities as the output tensor. It also stores the input and output for use in training.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates what patterns the RBM detects in the input data.
    /// 
    /// During the forward pass:
    /// - The RBM receives input data (like an image or set of ratings)
    /// - It calculates how strongly each hidden unit (pattern detector) should activate
    /// - The result shows which patterns were detected in the input
    /// - The original input and pattern detections are saved for training
    /// 
    /// This is like the RBM saying "based on this input, here are the patterns I can see in it."
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException("Input must have at least one dimension.", nameof(input));

        int visibleSize = input.Shape[^1];
        if (visibleSize != _visibleUnits)
            throw new ArgumentException($"Expected input feature size {_visibleUnits} but got {visibleSize}.", nameof(input));

        int flatBatch = 1;
        for (int d = 0; d < rank - 1; d++)
            flatBatch *= input.Shape[d];

        var visible2D = rank == 1
            ? input.Reshape([1, _visibleUnits])
            : input.Reshape([flatBatch, _visibleUnits]);

        _lastVisibleInput = visible2D;

        Tensor<T> hiddenProbs = SampleHiddenGivenVisibleTensor(visible2D);
        _lastHiddenOutput = hiddenProbs;

        if (rank == 1)
            return hiddenProbs.Reshape([_hiddenUnits]);

        var outputShape = new int[rank];
        for (int d = 0; d < rank - 1; d++)
            outputShape[d] = input.Shape[d];
        outputShape[rank - 1] = _hiddenUnits;

        return hiddenProbs.Reshape(outputShape);
    }

    /// <summary>
    /// Performs the forward pass on GPU using FusedLinearGpu with Sigmoid activation.
    /// </summary>
    /// <param name="inputs">The GPU input tensors.</param>
    /// <returns>The GPU output tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException("Input must have at least one dimension.", nameof(input));

        int visibleSize = input.Shape[^1];
        if (visibleSize != _visibleUnits)
            throw new ArgumentException($"Expected input feature size {_visibleUnits} but got {visibleSize}.", nameof(input));

        // Calculate batch size from all dimensions except the last
        int flatBatch = 1;
        for (int d = 0; d < rank - 1; d++)
            flatBatch *= input.Shape[d];

        // Reshape input to 2D [batch, visibleUnits] for matrix multiply
        IGpuTensor<T> input2D = input;
        if (rank == 1)
        {
            input2D = input.CreateView(0, [1, _visibleUnits]);
            flatBatch = 1;
        }
        else if (rank > 2)
        {
            input2D = input.CreateView(0, [flatBatch, _visibleUnits]);
        }

        // Transpose weights for FusedLinearGpu (expects [inputFeatures, outputFeatures])
        // RBM weights are [hiddenUnits, visibleUnits], need [visibleUnits, hiddenUnits]
        var weightsT = Engine.TensorTranspose(_weights);

        // Use FusedLinearGpu with Sigmoid activation - fully GPU-resident
        var result = gpuEngine.FusedLinearGpu(input2D, weightsT, _hiddenBiases, FusedActivationType.Sigmoid);

        // Cache state for backward pass only during training
        if (IsTrainingMode)
        {
            _lastVisibleInput = input.ToTensor();
            if (_lastVisibleInput.Shape.Length != 2)
                _lastVisibleInput = _lastVisibleInput.Reshape([flatBatch, _visibleUnits]);
            _lastHiddenOutput = result.ToTensor();
        }

        // Reshape output to match expected shape
        if (rank == 1)
        {
            return result.CreateView(0, [_hiddenUnits]);
        }
        else if (rank > 2)
        {
            var outputShape = new int[rank];
            for (int d = 0; d < rank - 1; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[rank - 1] = _hiddenUnits;
            return result.CreateView(0, outputShape);
        }

        return result;
    }

    /// <summary>
    /// Trains the RBM using contrastive divergence with the given data.
    /// </summary>
    /// <param name="input">The input data vector.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <param name="kSteps">The number of Gibbs sampling steps (typically 1).</param>
    /// <remarks>
    /// <para>
    /// This method implements the contrastive divergence (CD) algorithm for training the RBM.
    /// It performs 'kSteps' of Gibbs sampling to approximate the model's distribution and
    /// updates weights and biases to make the model distribution closer to the data distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the RBM to recognize patterns in data.
    /// 
    /// Contrastive divergence works by comparing:
    /// 1. How units activate with real data ("reality")
    /// 2. How units activate with the RBM's own generated data ("imagination")
    /// 
    /// The training process:
    /// - First computes probabilities and samples from the visible to hidden layer
    /// - Then reconstructs the visible layer from the hidden layer
    /// - Repeats this Gibbs sampling chain for kSteps iterations
    /// - Finally updates the weights and biases based on the difference between 
    ///   the original data correlations and the generated data correlations
    /// 
    /// The standard approach uses CD-1 (k=1), which works surprisingly well in practice
    /// despite being a rough approximation.
    /// </para>
    /// </remarks>
    public void TrainWithContrastiveDivergence(Vector<T> input, T learningRate, int kSteps = 1)
    {
        // Delegate to tensor-based implementation
        TrainWithContrastiveDivergenceTensor(new Tensor<T>([input.Length], input), learningRate, kSteps);
    }

    /// <summary>
    /// Tensor-based contrastive divergence training - no type conversions in hot path.
    /// </summary>
    private void TrainWithContrastiveDivergenceTensor(Tensor<T> input, T learningRate, int kSteps = 1)
    {
        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException("Input must have at least one dimension.", nameof(input));

        int visibleSize = input.Shape[^1];
        if (visibleSize != _visibleUnits)
            throw new ArgumentException($"Expected input feature size {_visibleUnits} but got {visibleSize}.", nameof(input));

        int flatBatch = 1;
        for (int d = 0; d < rank - 1; d++)
            flatBatch *= input.Shape[d];

        var v0 = rank == 1
            ? input.Reshape([1, _visibleUnits])
            : input.Reshape([flatBatch, _visibleUnits]);

        var h0Probs = SampleHiddenGivenVisibleTensor(v0);
        var h0Samples = SampleBinaryStatesTensor(h0Probs);

        var vk = v0;
        var hkSamples = h0Samples;
        Tensor<T> vkProbs = new Tensor<T>([flatBatch, _visibleUnits]);
        Tensor<T> hkProbs = new Tensor<T>([flatBatch, _hiddenUnits]);

        for (int step = 0; step < kSteps; step++)
        {
            vkProbs = SampleVisibleGivenHiddenTensor(hkSamples);
            vk = SampleBinaryStatesTensor(vkProbs);

            hkProbs = SampleHiddenGivenVisibleTensor(vk);

            if (step < kSteps - 1)
            {
                hkSamples = SampleBinaryStatesTensor(hkProbs);
            }
            else
            {
                hkSamples = hkProbs;
            }
        }

        T batchScale = NumOps.Divide(learningRate, NumOps.FromDouble(flatBatch));

        var hiddenBiasDiff = Engine.TensorSubtract(h0Probs, hkProbs);
        var hiddenBiasGrad = Engine.ReduceSum(hiddenBiasDiff, new[] { 0 }, keepDims: false);
        var hiddenBiasDelta = Engine.TensorMultiplyScalar(hiddenBiasGrad, batchScale);
        _hiddenBiases = Engine.TensorAdd(_hiddenBiases, hiddenBiasDelta);

        var visibleBiasDiff = Engine.TensorSubtract(v0, vkProbs);
        var visibleBiasGrad = Engine.ReduceSum(visibleBiasDiff, new[] { 0 }, keepDims: false);
        var visibleBiasDelta = Engine.TensorMultiplyScalar(visibleBiasGrad, batchScale);
        _visibleBiases = Engine.TensorAdd(_visibleBiases, visibleBiasDelta);

        var positiveOuter = Engine.TensorMatMul(Engine.TensorTranspose(h0Probs), v0);
        var negativeOuter = Engine.TensorMatMul(Engine.TensorTranspose(hkProbs), vk);
        var weightGradient = Engine.TensorSubtract(positiveOuter, negativeOuter);
        var weightDelta = Engine.TensorMultiplyScalar(weightGradient, batchScale);
        _weights = Engine.TensorAdd(_weights, weightDelta);
    }


    /// <summary>
    /// Computes the outer product of two vectors as a 2D tensor.
    /// </summary>
    private Tensor<T> ComputeOuterProduct(Vector<T> a, Vector<T> b)
    {
        // Delegate to tensor version
        return ComputeOuterProductTensor(new Tensor<T>([a.Length], a), new Tensor<T>([b.Length], b));
    }

    /// <summary>
    /// Samples binary states (0 or 1) from probability values.
    /// </summary>
    /// <param name="probabilities">Vector of probability values between 0 and 1.</param>
    /// <returns>Vector of binary samples (0 or 1) based on the probabilities.</returns>
    /// <remarks>
    /// <para>
    /// This method converts probability values to binary states by comparing each probability
    /// with a random number. If the probability is greater than the random number, the state
    /// becomes 1; otherwise, it becomes 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds randomness to make the RBM's behavior probabilistic.
    ///
    /// For each probability value (e.g., 0.7):
    /// - Generate a random number between 0 and 1 (e.g., 0.4)
    /// - If the probability is higher than the random number (0.7 > 0.4), output 1
    /// - Otherwise, output 0
    ///
    /// This stochastic (random) behavior is essential for RBMs because:
    /// - It prevents the RBM from getting stuck in fixed patterns
    /// - It allows exploration of different possible states
    /// - It models the inherent uncertainty in pattern recognition
    ///
    /// For example, a unit with a 70% probability will be active roughly 70% of the time,
    /// but not always, creating a range of possible network states.
    /// </para>
    /// </remarks>
    private Vector<T> SampleBinaryStates(Vector<T> probabilities)
    {
        // Delegate to tensor version and convert back
        var probTensor = new Tensor<T>([probabilities.Length], probabilities);
        var samplesTensor = SampleBinaryStatesTensor(probTensor);
        return new Vector<T>(samplesTensor.ToArray());
    }

    /// <summary>
    /// Computes the backward pass of the RBM layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the RBM layer. In the context of RBM training,
    /// it is used to compute a reconstruction of the visible units given the hidden unit activations.
    /// This is part of the contrastive divergence training algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs the input based on the detected patterns.
    /// 
    /// During the backward pass:
    /// - The RBM takes the pattern detections (hidden units)
    /// - It tries to recreate the original input that would produce these patterns
    /// - The result is the RBM's "imagination" of what the input should look like
    /// - Both the reconstruction and its corresponding pattern detections are saved for training
    /// 
    /// This is like the RBM saying "if these patterns exist, this is what the input should look like."
    /// The difference between this reconstruction and the original input drives the learning process.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        int rank = outputGradient.Shape.Length;
        if (rank < 1)
            throw new ArgumentException("Output gradient must have at least one dimension.", nameof(outputGradient));

        int hiddenSize = outputGradient.Shape[^1];
        if (hiddenSize != _hiddenUnits)
            throw new ArgumentException($"Expected hidden gradient size {_hiddenUnits} but got {hiddenSize}.", nameof(outputGradient));

        int flatBatch = 1;
        for (int d = 0; d < rank - 1; d++)
            flatBatch *= outputGradient.Shape[d];

        var hidden2D = rank == 1
            ? outputGradient.Reshape([1, _hiddenUnits])
            : outputGradient.Reshape([flatBatch, _hiddenUnits]);

        _reconstructedHidden = hidden2D;

        Tensor<T> visibleProbs = SampleVisibleGivenHiddenTensor(hidden2D);
        _reconstructedVisible = visibleProbs;

        if (_originalInputShape == null)
            return visibleProbs;

        if (_originalInputShape.Length == 1)
            return visibleProbs.Reshape([_visibleUnits]);

        var outputShape = (int[])_originalInputShape.Clone();
        outputShape[^1] = _visibleUnits;
        return visibleProbs.Reshape(outputShape);
    }


    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients for the mean-field forward pass.
    /// This enables discriminative fine-tuning of the RBM using standard backpropagation, distinct from
    /// the Contrastive Divergence training used for generative learning.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastVisibleInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastVisibleInput, "input", requiresGradient: true);
        var weightsNode = Autodiff.TensorOperations<T>.Variable(_weights, "weights", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_hiddenBiases, "hBias", requiresGradient: true);

        var weightsT = Autodiff.TensorOperations<T>.Transpose(weightsNode);
        var weighted = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, weightsT);
        var preActivation = Autodiff.TensorOperations<T>.Add(weighted, biasNode);
        var output = Autodiff.TensorOperations<T>.Sigmoid(preActivation);

        Tensor<T> gradient = outputGradient;
        if (outputGradient.Shape.Length != 2)
        {
            int rank = outputGradient.Shape.Length;
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= outputGradient.Shape[d];
            gradient = outputGradient.Reshape([flatBatch, _hiddenUnits]);
        }

        output.Gradient = gradient;

        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((output, false));

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

        _weightsGradient = weightsNode.Gradient;
        _hiddenBiasesGradient = biasNode.Gradient;
        _visibleBiasesGradient = new Tensor<T>(_visibleBiases.Shape);
        _visibleBiasesGradient.Fill(NumOps.Zero);

        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
        if (_originalInputShape != null)
            return inputGradient.Reshape(_originalInputShape);

        return inputGradient;
    }



    /// <summary>
    /// Computes the probability of each hidden unit being active given the visible units (tensor-based).
    /// </summary>
    /// <param name="visible">The visible unit tensor (1D).</param>
    /// <returns>A tensor of probabilities for each hidden unit.</returns>
    private Tensor<T> SampleHiddenGivenVisibleTensor(Tensor<T> visible)
    {
        Tensor<T> visible2D = visible;
        bool squeeze = false;

        if (visible.Shape.Length == 1)
        {
            visible2D = visible.Reshape([1, _visibleUnits]);
            squeeze = true;
        }
        else if (visible.Shape.Length != 2)
        {
            throw new ArgumentException("Visible tensor must be 1D or 2D.", nameof(visible));
        }

        var weightsT = Engine.TensorTranspose(_weights);
        var weighted = Engine.TensorMatMul(visible2D, weightsT);
        var biasExpanded = _hiddenBiases.Reshape([1, _hiddenUnits]);
        var activations = Engine.TensorBroadcastAdd(weighted, biasExpanded);
        var probs = Engine.Sigmoid(activations);

        return squeeze ? probs.Reshape([_hiddenUnits]) : probs;
    }


    /// <summary>
    /// Computes the probability of each visible unit being active given the hidden units (tensor-based).
    /// </summary>
    /// <param name="hidden">The hidden unit tensor (1D).</param>
    /// <returns>A tensor of probabilities for each visible unit.</returns>
    private Tensor<T> SampleVisibleGivenHiddenTensor(Tensor<T> hidden)
    {
        Tensor<T> hidden2D = hidden;
        bool squeeze = false;

        if (hidden.Shape.Length == 1)
        {
            hidden2D = hidden.Reshape([1, _hiddenUnits]);
            squeeze = true;
        }
        else if (hidden.Shape.Length != 2)
        {
            throw new ArgumentException("Hidden tensor must be 1D or 2D.", nameof(hidden));
        }

        var weighted = Engine.TensorMatMul(hidden2D, _weights);
        var biasExpanded = _visibleBiases.Reshape([1, _visibleUnits]);
        var activations = Engine.TensorBroadcastAdd(weighted, biasExpanded);
        var probs = Engine.Sigmoid(activations);

        return squeeze ? probs.Reshape([_visibleUnits]) : probs;
    }


    /// <summary>
    /// Samples binary states from probability tensor using stochastic sampling.
    /// </summary>
    private Tensor<T> SampleBinaryStatesTensor(Tensor<T> probabilities)
    {
        var randomTensor = Tensor<T>.CreateRandom(probabilities.Shape);
        return Engine.TensorGreaterThan(probabilities, randomTensor);
    }


    /// <summary>
    /// Computes the probability of each hidden unit being active given the visible units.
    /// </summary>
    /// <param name="visible">The visible unit values.</param>
    /// <returns>A vector of probabilities for each hidden unit.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the conditional probability P(h|v) for each hidden unit in the RBM.
    /// It calculates the activation of each hidden unit as the weighted sum of visible unit values
    /// plus the hidden bias, then applies the activation function to convert to a probability.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how strongly each pattern detector responds to the input.
    ///
    /// For each hidden unit (pattern detector):
    /// - The RBM calculates a weighted sum of all visible units that connect to it
    /// - It adds the hidden unit's bias (default sensitivity)
    /// - It applies the activation function to convert to a probability between 0 and 1
    /// - The result tells how strongly this pattern is present in the input
    ///
    /// For example, if a hidden unit detects "horizontal lines" in images, this method
    /// would calculate how confident the RBM is that horizontal lines are present in the input image.
    /// </para>
    /// </remarks>
    private Vector<T> SampleHiddenGivenVisible(Vector<T> visible)
    {
        // Use tensor-based implementation and convert back
        var visibleTensor = new Tensor<T>([visible.Length], visible);
        var hiddenProbsTensor = SampleHiddenGivenVisibleTensor(visibleTensor);
        return new Vector<T>(hiddenProbsTensor.ToArray());
    }

    /// <summary>
    /// Computes the probability of each visible unit being active given the hidden units.
    /// </summary>
    /// <param name="hidden">The hidden unit values.</param>
    /// <returns>A vector of probabilities for each visible unit.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the conditional probability P(v|h) for each visible unit in the RBM.
    /// It calculates the activation of each visible unit as the weighted sum of hidden unit values
    /// plus the visible bias, then applies the activation function to convert to a probability.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs the input based on detected patterns.
    ///
    /// For each visible unit (input element):
    /// - The RBM calculates a weighted sum of all hidden units that connect to it
    /// - It adds the visible unit's bias (default preference)
    /// - It applies the activation function to convert to a probability between 0 and 1
    /// - The result is the RBM's "guess" at what this input value should be
    ///
    /// For example, in an image recognition task, this would reconstruct the pixel values
    /// based on the patterns (features) that the RBM detected in the forward pass.
    /// </para>
    /// </remarks>
    private Vector<T> SampleVisibleGivenHidden(Vector<T> hidden)
    {
        // Use tensor-based implementation and convert back
        var hiddenTensor = new Tensor<T>([hidden.Length], hidden);
        var visibleProbsTensor = SampleVisibleGivenHiddenTensor(hiddenTensor);
        return new Vector<T>(visibleProbsTensor.ToArray());
    }

    /// <summary>
    /// Updates the layer's parameters using either standard backpropagation or contrastive divergence.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// This method handles two training modes:
    /// 1. Discriminative training (backprop): Uses gradients computed by BackwardViaAutodiff.
    /// 2. Generative training (CD-k): Uses statistics from the Gibbs sampling chain.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // 1. Standard Backpropagation (Discriminative Fine-tuning)
        if (_weightsGradient != null && _visibleBiasesGradient != null && _hiddenBiasesGradient != null)
        {
            _weights = Engine.TensorSubtract(_weights, Engine.TensorMultiplyScalar(_weightsGradient, learningRate));
            _visibleBiases = Engine.TensorSubtract(_visibleBiases, Engine.TensorMultiplyScalar(_visibleBiasesGradient, learningRate));
            _hiddenBiases = Engine.TensorSubtract(_hiddenBiases, Engine.TensorMultiplyScalar(_hiddenBiasesGradient, learningRate));
            return;
        }

        // 2. Contrastive Divergence (Generative Training)
        if (_lastVisibleInput != null && _lastHiddenOutput != null &&
            _reconstructedVisible != null && _reconstructedHidden != null)
        {
            int batchSize = _lastVisibleInput.Shape[0];
            T batchScale = NumOps.Divide(learningRate, NumOps.FromDouble(batchSize));

            var positiveOuter = Engine.TensorMatMul(Engine.TensorTranspose(_lastHiddenOutput), _lastVisibleInput);
            var negativeOuter = Engine.TensorMatMul(Engine.TensorTranspose(_reconstructedHidden), _reconstructedVisible);
            var weightGradient = Engine.TensorSubtract(positiveOuter, negativeOuter);
            var weightDelta = Engine.TensorMultiplyScalar(weightGradient, batchScale);
            _weights = Engine.TensorAdd(_weights, weightDelta);

            var hiddenBiasDiff = Engine.TensorSubtract(_lastHiddenOutput, _reconstructedHidden);
            var hiddenBiasGrad = Engine.ReduceSum(hiddenBiasDiff, new[] { 0 }, keepDims: false);
            var hiddenBiasDelta = Engine.TensorMultiplyScalar(hiddenBiasGrad, batchScale);
            _hiddenBiases = Engine.TensorAdd(_hiddenBiases, hiddenBiasDelta);

            var visibleBiasDiff = Engine.TensorSubtract(_lastVisibleInput, _reconstructedVisible);
            var visibleBiasGrad = Engine.ReduceSum(visibleBiasDiff, new[] { 0 }, keepDims: false);
            var visibleBiasDelta = Engine.TensorMultiplyScalar(visibleBiasGrad, batchScale);
            _visibleBiases = Engine.TensorAdd(_visibleBiases, visibleBiasDelta);
        }
    }


    /// <summary>
    /// Computes the outer product of two tensors as a 2D tensor.
    /// </summary>
    private Tensor<T> ComputeOuterProductTensor(Tensor<T> a, Tensor<T> b)
    {
        // Use tensor matrix multiplication for outer product: outer(a, b) = a.reshape([n,1]) @ b.reshape([1,m])
        var aReshaped = a.Reshape([a.Length, 1]);
        var bReshaped = b.Reshape([1, b.Length]);
        return Engine.TensorMatMul(aReshaped, bReshaped);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method collects all trainable parameters of the RBM (weights, visible biases, and
    /// hidden biases) into a single vector. The parameters are arranged in the order: weights
    /// (row-major), visible biases, hidden biases.
    /// </para>
    /// <para><b>For Beginners:</b> This method packs all the RBM's learnable values into one list.
    /// 
    /// The returned vector contains:
    /// - All weight values (connections between visible and hidden units)
    /// - All visible bias values (default preferences for visible units)
    /// - All hidden bias values (default sensitivities for hidden units)
    /// 
    /// This is useful for:
    /// - Saving the RBM's state to a file
    /// - Loading a previously trained RBM
    /// - Using optimization algorithms that work on all parameters at once
    /// 
    /// Think of it as taking a snapshot of everything the RBM has learned.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate for production-grade parameter extraction
        return Vector<T>.Concatenate(
            new Vector<T>(_weights.ToArray()),
            new Vector<T>(_visibleBiases.ToArray()),
            new Vector<T>(_hiddenBiases.ToArray())
        );
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the cached data used during training (both CD and backprop).
    /// While RBMs don't maintain state between passes in the same way as recurrent networks,
    /// this implementation does cache intermediate values for training purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the RBM's "memory" of previous inputs.
    /// 
    /// When you call this method:
    /// - The RBM forgets the last data it saw
    /// - It clears internal storage used during training
    /// - This ensures each batch of data is processed independently
    /// 
    /// This is useful when you want to start fresh, such as when:
    /// - Beginning training on a new dataset
    /// - Switching from training mode to evaluation mode
    /// - Processing unrelated sequences of data
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values used during contrastive divergence training
        _lastVisibleInput = null;
        _lastHiddenOutput = null;
        _reconstructedVisible = null;
        _reconstructedHidden = null;

        // Clear gradients from backpropagation
        _weightsGradient = null;
        _visibleBiasesGradient = null;
        _hiddenBiasesGradient = null;
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    public override int ParameterCount => _visibleUnits * _hiddenUnits + _visibleUnits + _hiddenUnits;

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // RBMLayer JIT uses mean-field inference (deterministic approximation):
        // Instead of stochastic sampling, we use hidden probabilities directly
        // hidden_probs = sigmoid(W @ visible + hidden_bias)
        // This provides a differentiable deterministic forward pass

        var input = inputNodes[0];

        // Storage is already Tensor<T>, use directly
        var weightsNode = TensorOperations<T>.Constant(_weights, "rbm_weights");
        var biasNode = TensorOperations<T>.Constant(_hiddenBiases, "rbm_hidden_bias");

        // Reshape input to column vector for matrix multiplication
        var inputReshaped = TensorOperations<T>.Reshape(input, _visibleUnits, 1);

        // W @ visible
        var weighted = TensorOperations<T>.MatrixMultiply(weightsNode, inputReshaped);

        // Reshape weighted to match bias
        var weightedFlat = TensorOperations<T>.Reshape(weighted, _hiddenUnits);

        // W @ visible + bias
        var preActivation = TensorOperations<T>.Add(weightedFlat, biasNode);

        // Apply sigmoid for mean-field inference (probability of hidden unit being active)
        var hiddenProbs = TensorOperations<T>.Sigmoid(preActivation);

        // Apply layer activation if different from sigmoid
        var output = ApplyActivationToGraph(hiddenProbs);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. RBM uses mean-field inference for JIT compilation.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for RBM uses mean-field inference instead of stochastic sampling.
    /// This provides a deterministic forward pass where hidden probabilities are computed
    /// directly using sigmoid(W*v + b) without sampling. Training still uses Contrastive
    /// Divergence with sampling, but inference/forward pass can be JIT compiled.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}
