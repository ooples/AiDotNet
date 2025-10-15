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
    /// The weights matrix represents the strength of connections between visible and hidden units.
    /// Each element W[j,i] represents the connection strength between visible unit i and hidden unit j.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the weights as "connection strengths" between units.
    /// 
    /// The weight matrix:
    /// - Has dimensions [hiddenUnits × visibleUnits]
    /// - Each weight shows how strongly a visible unit influences a hidden unit
    /// - Positive weights mean "these units tend to be active together"
    /// - Negative weights mean "when one unit is active, the other tends to be inactive"
    /// 
    /// During training, these weights are adjusted to capture patterns in your data.
    /// </para>
    /// </remarks>
    private Matrix<T> _weights = default!;

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
    private Vector<T> _visibleBiases = default!;

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
    private Vector<T> _hiddenBiases = default!;

    /// <summary>
    /// Stores the last input from the visible layer during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the original visible input vector from the most recent forward pass.
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
    private Vector<T>? _lastVisibleInput;

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
    private Vector<T>? _lastHiddenOutput;

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
    private Vector<T>? _reconstructedVisible;

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
    private Vector<T>? _reconstructedHidden;

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
        _weights = new Matrix<T>(_hiddenUnits, _visibleUnits);
        _visibleBiases = new Vector<T>(_visibleUnits);
        _hiddenBiases = new Vector<T>(_hiddenUnits);

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
    /// var rbmLayer = new RBMLayer<float>(784, 200, new Vector<double>izedSigmoidActivation<float>());
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
        _weights = new Matrix<T>(_hiddenUnits, _visibleUnits);
        _visibleBiases = new Vector<T>(_visibleUnits);
        _hiddenBiases = new Vector<T>(_hiddenUnits);

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
        for (int i = 0; i < _visibleUnits; i++)
        {
            _visibleBiases[i] = NumOps.Zero;
            for (int j = 0; j < _hiddenUnits; j++)
            {
                _weights[j, i] = NumOps.FromDouble(Random.NextDouble() * 0.1 - 0.05);
            }
        }

        for (int j = 0; j < _hiddenUnits; j++)
        {
            _hiddenBiases[j] = NumOps.Zero;
        }
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
        Vector<T> visibleLayer = input.ToVector();
    
        // Store the visible input for training
        _lastVisibleInput = new Vector<T>(visibleLayer);
    
        // Compute hidden probabilities given visible
        Vector<T> hiddenProbs = SampleHiddenGivenVisible(visibleLayer);
    
        // Store the hidden output for training
        _lastHiddenOutput = new Vector<T>(hiddenProbs);
    
        return Tensor<T>.FromVector(hiddenProbs);
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
        // --- Positive phase (data-driven) ---
        // Compute hidden probabilities and samples given the input data
        Vector<T> v0 = input;
        Vector<T> h0Probs = SampleHiddenGivenVisible(v0);
        Vector<T> h0Samples = SampleBinaryStates(h0Probs);
    
        // --- Negative phase (model-driven) ---
        // Reconstruct visible layer and resample hidden layer
        Vector<T> vk = v0;
        Vector<T> hkSamples = h0Samples;
        Vector<T> vkProbs = Vector<T>.Empty();
        Vector<T> hkProbs = Vector<T>.Empty();
    
        // Perform k steps of Gibbs sampling
        for (int step = 0; step < kSteps; step++)
        {
            // Sample v given h
            vkProbs = SampleVisibleGivenHidden(hkSamples);
            vk = SampleBinaryStates(vkProbs);
        
            // Sample h given v
            hkProbs = SampleHiddenGivenVisible(vk);
        
            // In the last step, we keep the probabilities instead of samples
            // for a more stable gradient estimate
            if (step < kSteps - 1)
            {
                hkSamples = SampleBinaryStates(hkProbs);
            }
            else
            {
                hkSamples = hkProbs; // On the last step, use probabilities
            }
        }
    
        // --- Update weights and biases ---
        // Update weights: W += learningRate * ((v0 * h0) - (vk * hk))
        for (int j = 0; j < _hiddenUnits; j++)
        {
            // Update hidden biases: hBias += learningRate * (h0 - hk)
            T hiddenBiasDelta = NumOps.Multiply(learningRate, 
                NumOps.Subtract(h0Probs[j], hkProbs[j]));
            _hiddenBiases[j] = NumOps.Add(_hiddenBiases[j], hiddenBiasDelta);
        
            for (int i = 0; i < _visibleUnits; i++)
            {
                // Positive phase correlation
                T positiveGradient = NumOps.Multiply(v0[i], h0Probs[j]);
            
                // Negative phase correlation
                T negativeGradient = NumOps.Multiply(vk[i], hkProbs[j]);
            
                // Weight update
                T weightDelta = NumOps.Multiply(learningRate, 
                    NumOps.Subtract(positiveGradient, negativeGradient));
                _weights[j, i] = NumOps.Add(_weights[j, i], weightDelta);
            
                // Update visible biases (only once per training example)
                if (j == 0)
                {
                    // vBias += learningRate * (v0 - vk)
                    T visibleBiasDelta = NumOps.Multiply(learningRate, 
                        NumOps.Subtract(v0[i], vkProbs[i]));
                    _visibleBiases[i] = NumOps.Add(_visibleBiases[i], visibleBiasDelta);
                }
            }
        }
    }

    /// <summary>
    /// Samples binary states (0 or 1) from probability values.
    /// </summary>
    /// <param name="probabilities">Vector<double> of probability values between 0 and 1.</param>
    /// <returns>Vector<double> of binary samples (0 or 1) based on the probabilities.</returns>
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
        Vector<T> samples = new Vector<T>(probabilities.Length);
        for (int i = 0; i < probabilities.Length; i++)
        {
            double prob = Convert.ToDouble(probabilities[i]);
            samples[i] = NumOps.FromDouble(Random.NextDouble() < prob ? 1.0 : 0.0);
        }

        return samples;
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
        // In RBM training, this is used for the reconstruction phase
        Vector<T> hiddenLayer = outputGradient.ToVector();
    
        // Store the reconstructed hidden for training (usually this would be based on the reconstructed visible)
        _reconstructedHidden = new Vector<T>(hiddenLayer);
    
        // Compute visible probabilities given hidden
        Vector<T> visibleProbs = SampleVisibleGivenHidden(hiddenLayer);
    
        // Store the reconstructed visible for training
        _reconstructedVisible = new Vector<T>(visibleProbs);
    
        return Tensor<T>.FromVector(visibleProbs);
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
        Vector<T> hiddenProbs = new Vector<T>(_hiddenUnits);
        for (int j = 0; j < _hiddenUnits; j++)
        {
            T activation = _hiddenBiases[j];
            for (int i = 0; i < _visibleUnits; i++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(_weights[j, i], visible[i]));
            }

            hiddenProbs[j] = ScalarActivation != null
                ? ScalarActivation.Activate(activation)
                : VectorActivation!.Activate(new Vector<T>([activation]))[0];
        }

        return hiddenProbs;
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
        Vector<T> visibleProbs = new Vector<T>(_visibleUnits);
        for (int i = 0; i < _visibleUnits; i++)
        {
            T activation = _visibleBiases[i];
            for (int j = 0; j < _hiddenUnits; j++)
            {
                activation = NumOps.Add(activation, NumOps.Multiply(_weights[j, i], hidden[j]));
            }

            visibleProbs[i] = ScalarActivation != null
                ? ScalarActivation.Activate(activation)
                : VectorActivation!.Activate(new Vector<T>([activation]))[0];
        }

        return visibleProbs;
    }

    /// <summary>
    /// Updates the layer's parameters using contrastive divergence.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// This method implements contrastive divergence with k=1 (CD-1) for training the RBM.
    /// The steps are:
    /// 1. Start with a visible vector v0 (input data)
    /// 2. Compute hidden probabilities h0 given v0 and sample a hidden state
    /// 3. Compute visible probabilities v1 given h0 and sample a visible state
    /// 4. Compute hidden probabilities h1 given v1
    /// 5. Update weights and biases using the difference between positive and negative phases
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // This would typically be done with stored input and output from the forward pass
        // For a complete implementation, we would need to store:
        // - v0: The original visible input from the forward pass
        // - h0: The hidden activations from the forward pass
        // - v1: The reconstructed visible activations from sampling given h0
        // - h1: The hidden activations from the backward pass
    
        // Since these aren't readily available in the current implementation,
        // we'll update the method signature to include what we need:
        if (_lastVisibleInput != null && _lastHiddenOutput != null && 
            _reconstructedVisible != null && _reconstructedHidden != null)
        {
            // Compute the weight updates
            for (int j = 0; j < _hiddenUnits; j++)
            {
                for (int i = 0; i < _visibleUnits; i++)
                {
                    // Positive phase: v0 * h0
                    T positivePhase = NumOps.Multiply(_lastVisibleInput[i], _lastHiddenOutput[j]);
                
                    // Negative phase: v1 * h1
                    T negativePhase = NumOps.Multiply(_reconstructedVisible[i], _reconstructedHidden[j]);
                
                    // Update: W += learningRate * (positivePhase - negativePhase)
                    T delta = NumOps.Multiply(learningRate, NumOps.Subtract(positivePhase, negativePhase));
                    _weights[j, i] = NumOps.Add(_weights[j, i], delta);
                }
            
                // Update hidden biases: h_bias += learningRate * (h0 - h1)
                T hiddenBiasDelta = NumOps.Multiply(learningRate, 
                    NumOps.Subtract(_lastHiddenOutput[j], _reconstructedHidden[j]));
                _hiddenBiases[j] = NumOps.Add(_hiddenBiases[j], hiddenBiasDelta);
            }
        
            // Update visible biases: v_bias += learningRate * (v0 - v1)
            for (int i = 0; i < _visibleUnits; i++)
            {
                T visibleBiasDelta = NumOps.Multiply(learningRate, 
                    NumOps.Subtract(_lastVisibleInput[i], _reconstructedVisible[i]));
                _visibleBiases[i] = NumOps.Add(_visibleBiases[i], visibleBiasDelta);
            }
        }
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
        // Create a vector to hold all parameters
        Vector<T> parameters = new Vector<T>(ParameterCount);
    
        // Copy weights
        int index = 0;
        for (int j = 0; j < _hiddenUnits; j++)
        {
            for (int i = 0; i < _visibleUnits; i++)
            {
                parameters[index++] = _weights[j, i];
            }
        }
    
        // Copy visible biases
        for (int i = 0; i < _visibleUnits; i++)
        {
            parameters[index++] = _visibleBiases[i];
        }
    
        // Copy hidden biases
        for (int j = 0; j < _hiddenUnits; j++)
        {
            parameters[index++] = _hiddenBiases[j];
        }
    
        return parameters;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the cached data used during contrastive divergence training.
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
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    public override int ParameterCount => _visibleUnits * _hiddenUnits + _visibleUnits + _hiddenUnits;

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;
}