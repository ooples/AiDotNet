namespace AiDotNet.NeuralNetworks;

public class GenerativeAdversarialNetwork<T> : NeuralNetworkBase<T>
{
    private readonly IFitnessCalculator<T> _fitnessCalculator;
    private Vector<T> _momentum;
    private Vector<T> _secondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate = 0.001;
    private double _learningRateDecay = 0.9999;
    private List<T> _generatorLosses = [];

    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    public GenerativeAdversarialNetwork(NeuralNetworkArchitecture<T> generatorArchitecture, 
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        IFitnessCalculator<T> fitnessCalculator,
        InputType inputType,
        double initialLearningRate = 0.001)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative, 
            NetworkComplexity.Medium, 
            generatorArchitecture.InputSize, 
            discriminatorArchitecture.OutputSize, 
            0, 0, 0, 
            null, null))
    {
        _fitnessCalculator = fitnessCalculator;
        _currentLearningRate = initialLearningRate;
    
        // Initialize optimizer parameters
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;
    
        // Initialize tracking collections
        _generatorLosses = new List<T>();
        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        _fitnessCalculator = fitnessCalculator;
        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();

        InitializeLayers();
    }

    public Vector<T> GenerateImage(Vector<T> noise)
    {
        return Generator.Predict(noise);
    }

    public T DiscriminateImage(Vector<T> image)
    {
        var result = Discriminator.Predict(image);
        return result[0];
    }

    public void TrainStep(Vector<T> realImages, Vector<T> noise)
    {
        // Train the discriminator
        var fakeImages = GenerateImage(noise);
        var realLabels = CreateLabelVector(realImages.Length, NumOps.One);
        var fakeLabels = CreateLabelVector(fakeImages.Length, NumOps.Zero);

        // Train on real images
        var realLoss = TrainDiscriminator(realImages, realLabels);

        // Train on fake images
        var fakeLoss = TrainDiscriminator(fakeImages, fakeLabels);

        var discriminatorLoss = NumOps.Add(realLoss, fakeLoss);

        // Train the generator
        var generatedImages = GenerateImage(noise);
        var allRealLabels = CreateLabelVector(generatedImages.Length, NumOps.One);

        // Train the generator to fool the discriminator
        var generatorLoss = TrainGenerator(noise, allRealLabels);
    }

    private static Vector<T> CreateLabelVector(int length, T value)
    {
        return new Vector<T>(Enumerable.Repeat(value, length).ToArray());
    }

    private T TrainDiscriminator(Vector<T> images, Vector<T> labels)
    {
        var predictions = Discriminator.Predict(images);
        var dataSetStats = new DataSetStats<T>
        {
            Predicted = predictions,
            Actual = labels
        };
        var loss = _fitnessCalculator.CalculateFitnessScore(dataSetStats);

        // Perform backpropagation and update weights
        var gradients = CalculateGradients(predictions, labels);
        ApplyGradients(Discriminator, gradients);
    
        return loss;
    }

    private T TrainGenerator(Vector<T> noise, Vector<T> targetLabels)
    {
        // Step 1: Generate fake images using the generator
        var generatedImages = Generator.Predict(noise);
    
        // Step 2: Forward pass through discriminator
        // Note: We need to keep track of activations for backpropagation
        Discriminator.SetTrainingMode(false); // Freeze discriminator weights during generator training
        var discriminatorOutput = Discriminator.Predict(generatedImages);
    
        // Step 3: Calculate loss - we want the discriminator to classify fake images as real
        var dataSetStats = new DataSetStats<T>
        {
            Predicted = discriminatorOutput,
            Actual = targetLabels
        };
        var loss = _fitnessCalculator.CalculateFitnessScore(dataSetStats);
    
        // Step 4: Backpropagation through the combined network (Generator + Discriminator)
        // First, get gradients from discriminator output with respect to its inputs
        var outputGradients = new Vector<T>(discriminatorOutput.Length);
        for (int i = 0; i < discriminatorOutput.Length; i++)
        {
            // For binary cross-entropy loss when target is 1 (real), gradient is -1/(output)
            outputGradients[i] = NumOps.Divide(
                NumOps.Negate(NumOps.One),
                NumOps.Add(discriminatorOutput[i], NumOps.FromDouble(1e-10)) // Add small epsilon to avoid division by zero
            );
        }
    
        // Step 5: Backpropagate through discriminator to get gradients at its input (which is the generator's output)
        var discriminatorInputGradients = Discriminator.Backpropagate(outputGradients);
    
        // Step 6: Backpropagate through generator using the gradients from discriminator
        var generatorGradients = Generator.Backpropagate(discriminatorInputGradients, noise);
    
        // Step 7: Extract the actual parameter gradients from the generator
        var parameterGradients = Generator.GetParameterGradients();
    
        // Step 8: Apply gradients to update generator parameters
        ApplyGradients(Generator, parameterGradients);
    
        // Step 9: Re-enable training mode for discriminator for future training steps
        Discriminator.SetTrainingMode(true);
    
        // Step 10: Track metrics for monitoring training progress
        _generatorLosses.Add(loss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0); // Keep only recent losses for moving average
        }
    
        // Optional: Implement early stopping or adaptive learning rate based on loss trends
        if (_generatorLosses.Count >= 10)
        {
            var recentAverage = _generatorLosses.Skip(_generatorLosses.Count - 5).Average(l => Convert.ToDouble(l));
            var previousAverage = _generatorLosses.Skip(_generatorLosses.Count - 10).Take(5).Average(l => Convert.ToDouble(l));
        
            // If loss is not improving, reduce learning rate
            if (recentAverage > previousAverage * 0.99)
            {
                _currentLearningRate *= 0.95; // Reduce learning rate by 5%
            }
        }
    
        return loss;
    }

    private Vector<T> CalculateGradients(Vector<T> predictions, Vector<T> targets)
    {
        // Simple gradient calculation - in a real implementation, this would be more complex
        // and would depend on your loss function
        var gradients = new Vector<T>(predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            gradients[i] = NumOps.Subtract(predictions[i], targets[i]);
        }
        return gradients;
    }

    private void ApplyGradients(ConvolutionalNeuralNetwork<T> network, Vector<T> gradients)
    {
        // Get current parameters
        var currentParams = network.GetParameters();
        var updatedParams = new Vector<T>(currentParams.Length);
    
        // Initialize momentum if it doesn't exist
        if (_momentum == null || _momentum.Length != currentParams.Length)
        {
            _momentum = new Vector<T>(currentParams.Length);
            _momentum.Fill(NumOps.Zero);
        }
    
        // Gradient clipping to prevent exploding gradients
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0); // Typical threshold value
    
        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
            }
        }
    
        // Apply Adam optimizer parameters
        var learningRate = NumOps.FromDouble(_currentLearningRate);
        var beta1 = NumOps.FromDouble(0.9);  // Momentum coefficient
        var beta2 = NumOps.FromDouble(0.999); // RMS coefficient
        var epsilon = NumOps.FromDouble(1e-8); // Small value to prevent division by zero
    
        // Update parameters with momentum and adaptive learning rate
        for (int i = 0; i < currentParams.Length && i < gradients.Length; i++)
        {
            // Update momentum
            _momentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _momentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );
        
            // Update second moment estimate if using Adam
            if (_secondMoment != null)
            {
                _secondMoment[i] = NumOps.Add(
                    NumOps.Multiply(beta2, _secondMoment[i]),
                    NumOps.Multiply(
                        NumOps.Subtract(NumOps.One, beta2),
                        NumOps.Multiply(gradients[i], gradients[i])
                    )
                );
            
                // Bias correction
                var momentumCorrected = NumOps.Divide(_momentum[i], NumOps.Subtract(NumOps.One, _beta1Power));
                var secondMomentCorrected = NumOps.Divide(_secondMoment[i], NumOps.Subtract(NumOps.One, _beta2Power));
            
                // Adam update
                var adaptiveLR = NumOps.Divide(
                    learningRate,
                    NumOps.Add(NumOps.Sqrt(secondMomentCorrected), epsilon)
                );
            
                updatedParams[i] = NumOps.Subtract(
                    currentParams[i],
                    NumOps.Multiply(adaptiveLR, momentumCorrected)
                );
            }
            else
            {
                // Simple SGD with momentum
                updatedParams[i] = NumOps.Subtract(
                    currentParams[i],
                    NumOps.Multiply(learningRate, _momentum[i])
                );
            }
        }
    
        // Update beta powers for next iteration
        _beta1Power = NumOps.Multiply(_beta1Power, beta1);
        _beta2Power = NumOps.Multiply(_beta2Power, beta2);
    
        // Apply learning rate decay
        _currentLearningRate *= _learningRateDecay;
    
        // Update network parameters
        network.UpdateParameters(updatedParams);
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        // In a GAN, "predict" typically means generating an image
        return GenerateImage(input);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        // Split parameters between generator and discriminator
        int generatorParamCount = Generator.GetParameterCount();
        var generatorParams = parameters.SubVector(0, generatorParamCount);
        var discriminatorParams = parameters.SubVector(generatorParamCount, parameters.Length - generatorParamCount);

        Generator.UpdateParameters(generatorParams);
        Discriminator.UpdateParameters(discriminatorParams);
    }

    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        // Serialize Generator
        writer.Write("Generator");
        Generator.Serialize(writer);

        // Serialize Discriminator
        writer.Write("Discriminator");
        Discriminator.Serialize(writer);
    }

    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        // Deserialize Generator
        string generatorLabel = reader.ReadString();
        if (generatorLabel != "Generator")
            throw new InvalidOperationException("Expected Generator data, but found: " + generatorLabel);
        Generator.Deserialize(reader);

        // Deserialize Discriminator
        string discriminatorLabel = reader.ReadString();
        if (discriminatorLabel != "Discriminator")
            throw new InvalidOperationException("Expected Discriminator data, but found: " + discriminatorLabel);
        Discriminator.Deserialize(reader);
    }

    protected override void InitializeLayers()
    {
        // GAN doesn't use layers directly, so this method is empty
    }
}