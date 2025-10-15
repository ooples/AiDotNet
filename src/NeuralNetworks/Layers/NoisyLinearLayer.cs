namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A linear layer with factorized Gaussian noise for exploration in reinforcement learning.
/// This layer is based on the paper "Noisy Networks for Exploration" (Fortunato et al., 2018).
/// </summary>
/// <remarks>
/// <para>
/// The Noisy Linear layer extends a standard linear layer by adding parametric noise to the weights and biases.
/// This enables the network to perform directed exploration by tracking the uncertainty in its value estimates.
/// The layer implements factorized Gaussian noise, which reduces the number of noise variables required.
/// </para>
/// <para><b>For Beginners:</b> This layer helps reinforcement learning agents explore more effectively.
/// 
/// Standard reinforcement learning agents often struggle with exploration:
/// - They rely on random actions to explore (like epsilon-greedy)
/// - This randomness isn't guided by what the agent knows or doesn't know
/// 
/// The Noisy Linear layer adds noise to the weights and biases:
/// - The noise is learned and adjusts during training
/// - It creates more exploration in areas where the agent is uncertain
/// - It allows different parts of the network to have different levels of noise
/// - As the agent becomes more confident, the noise naturally reduces
/// 
/// This approach has proven effective in many deep reinforcement learning applications,
/// particularly for complex environments with sparse rewards.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
public class NoisyLinearLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The mean weight parameters (without noise).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the learnable weights that determine the central tendency of the linear transformation.
    /// The actual weights used in the forward pass are these mean weights plus noise terms.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "average" or "central" weights.
    /// 
    /// Think of these as:
    /// - The base weights that the model learns
    /// - The stable part of the weights that captures learned patterns
    /// - What the weights would be in a standard layer without noise
    /// 
    /// During training, both these mean weights and the noise scale are learned.
    /// </para>
    /// </remarks>
    private Matrix<T> _weightsMu = default!;
    
    /// <summary>
    /// The mean bias parameters (without noise).
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the learnable biases that determine the central tendency of the linear transformation.
    /// The actual biases used in the forward pass are these mean biases plus noise terms.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "average" or "central" biases.
    /// 
    /// Like the mean weights, these biases:
    /// - Form the stable, learned part of the layer
    /// - Represent what biases would be in a standard layer
    /// - Get adjusted during training based on experience
    /// </para>
    /// </remarks>
    private Vector<T> _biasesMu = default!;
    
    /// <summary>
    /// The weight noise parameters that control the scale of noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These parameters control the scale of the noise applied to the weights. Larger values lead to more
    /// exploration, while smaller values lead to more exploitation. These parameters are learned during training.
    /// </para>
    /// <para><b>For Beginners:</b> These control how much uncertainty to add to each weight.
    /// 
    /// These parameters:
    /// - Determine how much noise to add to each weight
    /// - Are learned during training (not fixed)
    /// - Will decrease as the agent becomes more confident
    /// - Allow different weights to have different amounts of noise
    /// 
    /// This is what makes exploration "directed" rather than random - the model learns
    /// where it needs more exploration and where it needs less.
    /// </para>
    /// </remarks>
    private Matrix<T> _weightsSigma = default!;
    
    /// <summary>
    /// The bias noise parameters that control the scale of noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These parameters control the scale of the noise applied to the biases. Like the weight noise parameters,
    /// they are learned during training and allow for directed exploration.
    /// </para>
    /// <para><b>For Beginners:</b> These control how much uncertainty to add to each bias.
    /// 
    /// Similar to weights sigma, these parameters:
    /// - Control noise levels for the bias terms
    /// - Are adjusted during training
    /// - Help the agent express uncertainty in different parts of its prediction
    /// 
    /// Together with weight sigmas, they give the layer a "confidence level" for its output.
    /// </para>
    /// </remarks>
    private Vector<T> _biasesSigma = default!;
    
    /// <summary>
    /// The noise samples for the input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the random noise samples for the input features, sampled from a standard normal distribution.
    /// They are used to generate factorized noise for the weights matrix.
    /// </para>
    /// <para><b>For Beginners:</b> These are random values used to create noise for each input neuron.
    /// 
    /// In factorized noise:
    /// - We generate independent noise for inputs and outputs
    /// - Then combine them to create the full weight noise
    /// - This is more efficient than generating separate noise for each weight
    /// 
    /// These random values are resampled before each forward pass during training.
    /// </para>
    /// </remarks>
    private Vector<T> _epsilonInput = default!;
    
    /// <summary>
    /// The noise samples for the output features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the random noise samples for the output features, sampled from a standard normal distribution.
    /// They are used to generate factorized noise for the weights matrix and for the bias vector.
    /// </para>
    /// <para><b>For Beginners:</b> These are random values used to create noise for each output neuron.
    /// 
    /// These work with the input noise to create the pattern of noise across all weights.
    /// They're also used to add noise to the bias terms.
    /// </para>
    /// </remarks>
    private Vector<T> _epsilonOutput = default!;
    
    /// <summary>
    /// The noisy weights used in the current forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the actual weights used during the forward pass, computed as the mean weights plus the noise terms.
    /// The noise terms are a product of the noise parameters and the factorized noise samples.
    /// </para>
    /// <para><b>For Beginners:</b> These are the complete weights that include both the mean and noise.
    /// 
    /// When the layer processes data:
    /// - It uses these noisy weights, not just the mean weights
    /// - These combine the learned pattern (mean) with exploration (noise)
    /// - They're recalculated before each forward pass during training
    /// 
    /// The level of noise decreases over time as the agent learns.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights = default!;
    
    /// <summary>
    /// The noisy biases used in the current forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the actual biases used during the forward pass, computed as the mean biases plus the noise terms.
    /// The noise terms are a product of the noise parameters and the factorized noise samples.
    /// </para>
    /// <para><b>For Beginners:</b> These are the complete biases that include both the mean and noise.
    /// 
    /// Like the noisy weights, these:
    /// - Combine the learned bias values with exploration noise
    /// - Are used in the actual calculations
    /// - Change slightly each time during training
    /// </para>
    /// </remarks>
    private Tensor<T> _biases = default!;
    
    /// <summary>
    /// The gradients for the mean weights calculated during the backward pass.
    /// </summary>
    private Matrix<T>? _weightsMuGradient;
    
    /// <summary>
    /// The gradients for the mean biases calculated during the backward pass.
    /// </summary>
    private Vector<T>? _biasesMuGradient;
    
    /// <summary>
    /// The gradients for the weight noise parameters calculated during the backward pass.
    /// </summary>
    private Matrix<T>? _weightsSigmaGradient;
    
    /// <summary>
    /// The gradients for the bias noise parameters calculated during the backward pass.
    /// </summary>
    private Vector<T>? _biasesSigmaGradient;
    
    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    
    /// <summary>
    /// The initial scale of the noise.
    /// </summary>
    private readonly T _initialSigma = default!;
    
    /// <summary>
    /// The number of output features.
    /// </summary>
    private readonly int _outputFeatures;
    
    /// <summary>
    /// The random number generator used to sample noise.
    /// </summary>
    private readonly Random _random = default!;
    
    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The NoisyLinearLayer always returns true because it contains trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// This layer always supports training because it has weights and biases that can be updated.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="NoisyLinearLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">The number of input features.</param>
    /// <param name="outputFeatures">The number of output features.</param>
    /// <param name="activationFunction">The activation function to apply. Defaults to identity if not specified.</param>
    /// <param name="initialSigma">The initial scale of the noise. Defaults to 0.5 if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Noisy Linear Layer with randomly initialized parameters.
    /// The initial sigma controls the initial scale of the noise, which affects the amount of exploration.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with specific input and output sizes.
    /// 
    /// When creating this layer, you specify:
    /// - How many features your input has (inputFeatures)
    /// - How many features you want in the output (outputFeatures)
    /// - An optional activation function to apply after the linear transformation
    /// - How much initial noise to use for exploration (initialSigma)
    /// 
    /// A higher initialSigma value means more exploration at the beginning of training.
    /// </para>
    /// </remarks>
    public NoisyLinearLayer(int inputFeatures, int outputFeatures, IActivationFunction<T>? activationFunction = null, double initialSigma = 0.5)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _outputFeatures = outputFeatures;
        _initialSigma = NumOps.FromDouble(initialSigma);
        _random = new Random();
        
        _weightsMu = new Matrix<T>(inputFeatures, outputFeatures);
        _biasesMu = new Vector<T>(outputFeatures);
        _weightsSigma = new Matrix<T>(inputFeatures, outputFeatures);
        _biasesSigma = new Vector<T>(outputFeatures);
        
        _epsilonInput = new Vector<T>(inputFeatures);
        _epsilonOutput = new Vector<T>(outputFeatures);
        
        _weights = new Tensor<T>([inputFeatures, outputFeatures]);
        _biases = new Tensor<T>([1, outputFeatures]);
        
        InitializeParameters();
    }
    
    /// <summary>
    /// Initializes a new instance of the <see cref="NoisyLinearLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputFeatures">The number of input features.</param>
    /// <param name="outputFeatures">The number of output features.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply. Defaults to identity if not specified.</param>
    /// <param name="initialSigma">The initial scale of the noise. Defaults to 0.5 if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Noisy Linear Layer with a vector activation function.
    /// Vector<double> activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different outputs.
    /// </para>
    /// </remarks>
    public NoisyLinearLayer(int inputFeatures, int outputFeatures, IVectorActivationFunction<T>? vectorActivationFunction = null, double initialSigma = 0.5)
        : base([inputFeatures], [outputFeatures], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _outputFeatures = outputFeatures;
        _initialSigma = NumOps.FromDouble(initialSigma);
        _random = new Random();
        
        _weightsMu = new Matrix<T>(inputFeatures, outputFeatures);
        _biasesMu = new Vector<T>(outputFeatures);
        _weightsSigma = new Matrix<T>(inputFeatures, outputFeatures);
        _biasesSigma = new Vector<T>(outputFeatures);
        
        _epsilonInput = new Vector<T>(inputFeatures);
        _epsilonOutput = new Vector<T>(outputFeatures);
        
        _weights = new Tensor<T>([inputFeatures, outputFeatures]);
        _biases = new Tensor<T>([1, outputFeatures]);
        
        InitializeParameters();
    }
    
    /// <summary>
    /// Initializes the layer's parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the mean weights with scaled random values, the mean biases to zero,
    /// and the noise parameters to a constant value scaled by the input dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the layer's parameters.
    /// 
    /// The initialization:
    /// - Sets mean weights to small random values (for neural network training stability)
    /// - Sets mean biases to zero
    /// - Sets sigma parameters for weights based on input size
    /// - Sets sigma parameters for biases to a fixed value
    /// 
    /// Good initialization helps the network train more effectively from the start.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int inputFeatures = _weightsMu.Rows;
        
        // Initialize mean parameters with standard weight initialization
        var stdv = NumOps.FromDouble(1.0 / Math.Sqrt(inputFeatures));
        InitializeMatrix(_weightsMu, stdv);
        
        for (int i = 0; i < _biasesMu.Length; i++)
        {
            _biasesMu[i] = NumOps.Zero;
        }
        
        // Initialize sigma parameters
        for (int i = 0; i < _weightsSigma.Rows; i++)
        {
            for (int j = 0; j < _weightsSigma.Columns; j++)
            {
                _weightsSigma[i, j] = NumOps.Divide(_initialSigma, NumOps.FromDouble(Math.Sqrt(inputFeatures)));
            }
        }
        
        for (int i = 0; i < _biasesSigma.Length; i++)
        {
            _biasesSigma[i] = NumOps.Divide(_initialSigma, NumOps.FromDouble(Math.Sqrt(1.0)));
        }
        
        // Initialize noise and compute noisy weights
        ResetNoise();
    }
    
    /// <summary>
    /// Initializes a matrix with scaled random values.
    /// </summary>
    /// <param name="matrix">The matrix to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the provided matrix with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps with training stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a matrix with random values for starting weights.
    /// 
    /// The method:
    /// - Generates random numbers between -0.5 and 0.5
    /// - Multiplies them by a scale factor to control their size
    /// - Fills each position in the matrix with these scaled random values
    /// 
    /// Good initialization is important because it affects how quickly and how well the network learns.
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
    /// Generates a sample from a standard normal distribution.
    /// </summary>
    /// <returns>A random sample from N(0, 1).</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Box-Muller transform to generate samples from a standard normal distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This creates random numbers that follow a bell curve shape.
    /// 
    /// Standard normal distribution:
    /// - Has a mean (average) of 0
    /// - Has a standard deviation of 1
    /// - Most values fall between -3 and 3
    /// - Values closer to 0 are more likely
    /// 
    /// This is important for creating noise that has useful statistical properties.
    /// </para>
    /// </remarks>
    private T SampleGaussian()
    {
        // Box-Muller transform to generate standard normal distribution
        double u1 = 1.0 - _random.NextDouble(); // Uniform(0,1] random doubles
        double u2 = 1.0 - _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        
        return NumOps.FromDouble(z);
    }
    
    /// <summary>
    /// Applies a special transformation to convert standard normal samples to factorized noise.
    /// </summary>
    /// <param name="epsilon">The standard normal samples to transform.</param>
    /// <returns>A vector of transformed noise values.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the transformation f(x) = sign(x) * sqrt(|x|) to convert standard normal samples
    /// to factorized noise as described in the "Noisy Networks for Exploration" paper. This transformation
    /// preserves the expected value of the products while using fewer random variables.
    /// </para>
    /// <para><b>For Beginners:</b> This transforms standard random numbers into a special noise pattern.
    /// 
    /// The transformation:
    /// - Preserves the sign (positive or negative) of the original number
    /// - Uses the square root of the absolute value to adjust the magnitude
    /// - Creates a specific statistical pattern that works well for factorized noise
    /// 
    /// This allows the layer to create effective noise for exploration while using
    /// fewer random numbers (which is more efficient).
    /// </para>
    /// </remarks>
    private Vector<T> FactorizeNoise(Vector<T> epsilon)
    {
        var result = new Vector<T>(epsilon.Length);
        
        for (int i = 0; i < epsilon.Length; i++)
        {
            var sign = NumOps.GreaterThanOrEquals(epsilon[i], NumOps.Zero) 
                ? NumOps.One 
                : NumOps.FromDouble(-1);
            
            result[i] = NumOps.Multiply(sign, 
                NumOps.Sqrt(NumOps.Abs(epsilon[i])));
        }
        
        return result;
    }
    
    /// <summary>
    /// Resets the noise samples and recomputes the noisy weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method samples new noise, applies the factorization transformation, and recomputes the noisy
    /// weights and biases. It should be called before each forward pass during training to ensure
    /// exploration, and can be called less frequently or not at all during inference.
    /// </para>
    /// <para><b>For Beginners:</b> This method refreshes the noise in the layer for more exploration.
    /// 
    /// Each time this is called:
    /// - New random noise values are generated
    /// - The noise is transformed to have useful statistical properties
    /// - The noisy weights and biases are recalculated
    /// 
    /// During training, this should be called before every forward pass to
    /// enable exploration. During evaluation, you might not reset the noise
    /// to get more consistent predictions.
    /// </para>
    /// </remarks>
    public void ResetNoise()
    {
        // Sample new noise
        for (int i = 0; i < _epsilonInput.Length; i++)
        {
            _epsilonInput[i] = SampleGaussian();
        }
        
        for (int j = 0; j < _epsilonOutput.Length; j++)
        {
            _epsilonOutput[j] = SampleGaussian();
        }
        
        // Apply factorized noise transformation
        var epsilonInputFactorized = FactorizeNoise(_epsilonInput);
        var epsilonOutputFactorized = FactorizeNoise(_epsilonOutput);
        
        // Create tensors to hold the noisy weights and biases
        var weightsTensor = new Tensor<T>([_weightsMu.Rows, _weightsMu.Columns]);
        var biasesTensor = new Tensor<T>([1, _biasesMu.Length]);
        
        // Compute noisy weights
        for (int i = 0; i < _weightsMu.Rows; i++)
        {
            for (int j = 0; j < _weightsMu.Columns; j++)
            {
                var noise = NumOps.Multiply(epsilonInputFactorized[i], epsilonOutputFactorized[j]);
                weightsTensor[i, j] = NumOps.Add(_weightsMu[i, j], 
                    NumOps.Multiply(_weightsSigma[i, j], noise));
            }
        }
        
        // Compute noisy biases
        for (int j = 0; j < _biasesMu.Length; j++)
        {
            biasesTensor[0, j] = NumOps.Add(_biasesMu[j], 
                NumOps.Multiply(_biasesSigma[j], epsilonOutputFactorized[j]));
        }
        
        _weights = weightsTensor;
        _biases = biasesTensor;
    }
    
    /// <summary>
    /// Performs the forward pass of the layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after the noisy linear transformation and activation.</returns>
    /// <exception cref="InvalidOperationException">Thrown when called before initialization.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the noisy linear layer, which applies a noisy linear
    /// transformation followed by an activation function. The noise introduces randomness that enables
    /// exploration in reinforcement learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes data through the noisy linear layer.
    /// 
    /// During the forward pass:
    /// 1. The input data is stored for later use in the backward pass
    /// 2. The input is multiplied by the noisy weights matrix
    /// 3. The noisy biases are added to the result
    /// 4. The activation function is applied
    /// 
    /// The result includes randomness from the noise, which helps with exploration
    /// in reinforcement learning by making the agent try different actions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        
        // Linear transformation: y = x * W + b
        var output = input.MatrixMultiply(_weights);
        output = output.Add(_biases);
        
        // Apply activation function
        output = ApplyActivation(output);
        
        return output;
    }
    
    /// <summary>
    /// Performs the backward pass of the layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the noisy linear layer, which computes the gradients
    /// of the loss with respect to the layer's parameters and inputs. The gradients are used during the
    /// update step to improve the layer's performance.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to adjust the layer's parameters during training.
    /// 
    /// During the backward pass:
    /// 1. The gradient is passed through the activation function
    /// 2. Gradients for the mean weights and biases are calculated
    /// 3. Gradients for the noise parameters (sigmas) are calculated
    /// 4. The gradient with respect to the input is calculated for further backpropagation
    /// 
    /// This allows the layer to learn both the mean values (the task solution)
    /// and the noise levels (the exploration strategy) simultaneously.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        
        // Apply activation function gradient - null means identity activation derivative
        var activationGradient = ScalarActivation != null 
            ? ApplyActivationDerivative(outputGradient.Clone(), outputGradient) 
            : outputGradient;
        
        // Calculate factorized noise
        var epsilonInputFactorized = FactorizeNoise(_epsilonInput);
        var epsilonOutputFactorized = FactorizeNoise(_epsilonOutput);
        
        // Initialize gradients
        int inputFeatures = _lastInput.Shape[1];
        int batchSize = _lastInput.Shape[0];
        
        _weightsMuGradient = new Matrix<T>(inputFeatures, _outputFeatures);
        _biasesMuGradient = new Vector<T>(_outputFeatures);
        _weightsSigmaGradient = new Matrix<T>(inputFeatures, _outputFeatures);
        _biasesSigmaGradient = new Vector<T>(_outputFeatures);
        
        // Calculate gradients for mean parameters
        for (int i = 0; i < inputFeatures; i++)
        {
            for (int j = 0; j < _outputFeatures; j++)
            {
                T gradientSum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    gradientSum = NumOps.Add(gradientSum, 
                        NumOps.Multiply(_lastInput[b, i], activationGradient[b, j]));
                }
                _weightsMuGradient[i, j] = gradientSum;
            }
        }
        
        for (int j = 0; j < _outputFeatures; j++)
        {
            T gradientSum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                gradientSum = NumOps.Add(gradientSum, activationGradient[b, j]);
            }
            _biasesMuGradient[j] = gradientSum;
        }
        
        // Calculate gradients for sigma parameters
        for (int i = 0; i < inputFeatures; i++)
        {
            for (int j = 0; j < _outputFeatures; j++)
            {
                T noise = NumOps.Multiply(epsilonInputFactorized[i], epsilonOutputFactorized[j]);
                T gradientSum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    gradientSum = NumOps.Add(gradientSum, 
                        NumOps.Multiply(_lastInput[b, i], NumOps.Multiply(activationGradient[b, j], noise)));
                }
                _weightsSigmaGradient[i, j] = gradientSum;
            }
        }
        
        for (int j = 0; j < _outputFeatures; j++)
        {
            T gradientSum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                gradientSum = NumOps.Add(gradientSum, 
                    NumOps.Multiply(activationGradient[b, j], epsilonOutputFactorized[j]));
            }
            _biasesSigmaGradient[j] = gradientSum;
        }
        
        // Calculate input gradient
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    inputGradient[b, i] = NumOps.Add(inputGradient[b, i], 
                        NumOps.Multiply(activationGradient[b, j], _weights[i, j]));
                }
            }
        }
        
        return inputGradient;
    }
    
    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer based on the gradients calculated during the
    /// backward pass. The learning rate controls the size of the parameter updates. After updating the parameters,
    /// the noisy weights and biases are recomputed for the next forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's parameters during training.
    /// 
    /// During the update:
    /// - The mean weights and biases are adjusted to improve predictions
    /// - The noise parameters (sigmas) are adjusted to optimize exploration
    /// - The learning rate controls how big each adjustment is
    /// - The noisy weights and biases are recalculated with the updated parameters
    /// 
    /// Over time, the layer learns both how to solve the task (mean parameters)
    /// and how much uncertainty it should have (sigma parameters).
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsMuGradient == null || _biasesMuGradient == null || 
            _weightsSigmaGradient == null || _biasesSigmaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        
        // Update mean parameters
        for (int i = 0; i < _weightsMu.Rows; i++)
        {
            for (int j = 0; j < _weightsMu.Columns; j++)
            {
                _weightsMu[i, j] = NumOps.Subtract(_weightsMu[i, j], 
                    NumOps.Multiply(learningRate, _weightsMuGradient[i, j]));
            }
        }
        
        for (int j = 0; j < _biasesMu.Length; j++)
        {
            _biasesMu[j] = NumOps.Subtract(_biasesMu[j], 
                NumOps.Multiply(learningRate, _biasesMuGradient[j]));
        }
        
        // Update sigma parameters
        for (int i = 0; i < _weightsSigma.Rows; i++)
        {
            for (int j = 0; j < _weightsSigma.Columns; j++)
            {
                _weightsSigma[i, j] = NumOps.Subtract(_weightsSigma[i, j], 
                    NumOps.Multiply(learningRate, _weightsSigmaGradient[i, j]));
            }
        }
        
        for (int j = 0; j < _biasesSigma.Length; j++)
        {
            _biasesSigma[j] = NumOps.Subtract(_biasesSigma[j], 
                NumOps.Multiply(learningRate, _biasesSigmaGradient[j]));
        }
        
        // Recompute noisy weights and biases
        ResetNoise();
    }
    
    
    
    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method concatenates all trainable parameters of the layer into a single vector.
    /// The parameters are ordered as: mean weights, mean biases, weight sigmas, bias sigmas.
    /// </para>
    /// <para><b>For Beginners:</b> This method combines all the layer's parameters into one long list.
    /// 
    /// This flattened vector:
    /// - Contains all the parameter values in a specific order
    /// - Makes it easy to save all parameters at once
    /// - Is useful for optimization algorithms that work on all parameters together
    /// 
    /// You can reconstruct the original parameters from this vector using SetParameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        int totalParams = 
            _weightsMu.Rows * _weightsMu.Columns + 
            _biasesMu.Length + 
            _weightsSigma.Rows * _weightsSigma.Columns + 
            _biasesSigma.Length;
        
        var parameters = new Vector<T>(totalParams);
        int index = 0;
        
        // Copy mean weights
        for (int i = 0; i < _weightsMu.Rows; i++)
        {
            for (int j = 0; j < _weightsMu.Columns; j++)
            {
                parameters[index++] = _weightsMu[i, j];
            }
        }
        
        // Copy mean biases
        for (int i = 0; i < _biasesMu.Length; i++)
        {
            parameters[index++] = _biasesMu[i];
        }
        
        // Copy weight sigmas
        for (int i = 0; i < _weightsSigma.Rows; i++)
        {
            for (int j = 0; j < _weightsSigma.Columns; j++)
            {
                parameters[index++] = _weightsSigma[i, j];
            }
        }
        
        // Copy bias sigmas
        for (int i = 0; i < _biasesSigma.Length; i++)
        {
            parameters[index++] = _biasesSigma[i];
        }
        
        return parameters;
    }
    
    /// <summary>
    /// Sets the trainable parameters of the layer from a vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown if the vector has the wrong length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the layer from a single vector.
    /// The parameters should be in the same order as returned by GetParameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's parameters from a single list.
    /// 
    /// When setting parameters:
    /// - The vector must have the correct length to match all parameters
    /// - Parameters are retrieved in the same order they were stored
    /// - After setting, noisy weights and biases are recalculated
    /// 
    /// This is useful for loading a saved model or transferring parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedLength = 
            _weightsMu.Rows * _weightsMu.Columns + 
            _biasesMu.Length + 
            _weightsSigma.Rows * _weightsSigma.Columns + 
            _biasesSigma.Length;
        
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected parameters vector of length {expectedLength}, but got {parameters.Length}");
        }
        
        int index = 0;
        
        // Set mean weights
        for (int i = 0; i < _weightsMu.Rows; i++)
        {
            for (int j = 0; j < _weightsMu.Columns; j++)
            {
                _weightsMu[i, j] = parameters[index++];
            }
        }
        
        // Set mean biases
        for (int i = 0; i < _biasesMu.Length; i++)
        {
            _biasesMu[i] = parameters[index++];
        }
        
        // Set weight sigmas
        for (int i = 0; i < _weightsSigma.Rows; i++)
        {
            for (int j = 0; j < _weightsSigma.Columns; j++)
            {
                _weightsSigma[i, j] = parameters[index++];
            }
        }
        
        // Set bias sigmas
        for (int i = 0; i < _biasesSigma.Length; i++)
        {
            _biasesSigma[i] = parameters[index++];
        }
        
        // Recompute noisy weights and biases
        ResetNoise();
    }
    
    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// It also regenerates the noise for the next forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears temporary data and refreshes the noise.
    /// 
    /// When resetting the state:
    /// - Saved inputs and gradients are cleared
    /// - New noise is generated for weights and biases
    /// 
    /// This is useful between episodes in reinforcement learning or when
    /// starting to process new, unrelated data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _weightsMuGradient = null;
        _biasesMuGradient = null;
        _weightsSigmaGradient = null;
        _biasesSigmaGradient = null;
        
        // Reset noise to encourage exploration
        ResetNoise();
    }
}