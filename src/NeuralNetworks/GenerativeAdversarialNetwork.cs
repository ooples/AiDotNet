namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Generative Adversarial Network (GAN), a deep learning architecture that consists of two neural networks
/// (a generator and a discriminator) competing against each other in a zero-sum game.
/// </summary>
/// <remarks>
/// <para>
/// A Generative Adversarial Network (GAN) is a powerful machine learning architecture that uses two neural networks - 
/// a generator and a discriminator - that are trained simultaneously through adversarial training. The generator 
/// network learns to create realistic synthetic data samples (like images), while the discriminator network learns 
/// to distinguish between real data and the generator's synthetic outputs. As training progresses, the generator 
/// becomes better at creating realistic data, and the discriminator becomes better at distinguishing real from fake, 
/// pushing each other to improve in a competitive process.
/// </para>
/// <para><b>For Beginners:</b> A GAN is like an art forger and an art detective competing against each other.
/// 
/// Think of it this way:
/// - The generator is like an art forger trying to create fake paintings that look real
/// - The discriminator is like an art detective trying to tell which paintings are real and which are fake
/// - As the forger gets better, the detective has to improve too
/// - As the detective gets better, the forger is forced to create more convincing fakes
/// - Eventually, the forger becomes so good that their fake paintings are nearly indistinguishable from real ones
/// 
/// This continuous competition drives both networks to improve, resulting in a generator that
/// can create remarkably realistic synthetic data like images, music, or text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GenerativeAdversarialNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the fitness calculator used to evaluate how well the networks are performing.
    /// </summary>
    /// <value>The fitness calculator instance used for evaluating network performance.</value>
    /// <remarks>
    /// <para>
    /// The fitness calculator is used to compute loss values that guide the training of both the generator
    /// and discriminator networks. In GANs, the choice of loss function is critical for stable training
    /// and good performance. Common choices include binary cross-entropy loss or Wasserstein loss.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the networks know how well they're doing.
    /// 
    /// Think of the fitness calculator as:
    /// - A scoring system that evaluates how well each network is performing
    /// - For the discriminator, it measures how accurately it can distinguish real from fake
    /// - For the generator, it measures how well it can fool the discriminator
    /// - These scores guide how the networks should improve in the next training step
    /// 
    /// Just like a coach provides feedback to athletes, the fitness calculator provides
    /// feedback to help both networks improve their performance.
    /// </para>
    /// </remarks>
    private readonly IFitnessCalculator<T> _fitnessCalculator;

    /// <summary>
    /// Gets or sets the momentum values for the optimizer.
    /// </summary>
    /// <value>A vector of momentum values for each parameter.</value>
    /// <remarks>
    /// <para>
    /// Momentum is an optimization technique that helps accelerate gradient descent in the relevant direction
    /// and dampens oscillations. It does this by adding a fraction of the previous parameter update to the
    /// current update. This helps the optimizer converge faster and avoid getting stuck in local minima.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the networks learn more smoothly.
    /// 
    /// Think of momentum as:
    /// - A ball rolling down a hill that builds up speed
    /// - It helps the network keep moving in a consistent direction
    /// - It smooths out the learning process, preventing wild changes
    /// - This makes training more stable and often faster
    /// 
    /// For example, if the network is consistently trying to adjust a parameter in
    /// the same direction, momentum helps it make bigger adjustments over time.
    /// </para>
    /// </remarks>
    private Vector<T> _momentum;

    /// <summary>
    /// Gets or sets the second moment estimates for the Adam optimizer.
    /// </summary>
    /// <value>A vector of second moment values for each parameter.</value>
    /// <remarks>
    /// <para>
    /// The second moment estimates are used by the Adam optimizer to adapt the learning rate for each parameter.
    /// They track the squared gradients, providing a measure of how quickly each parameter is changing.
    /// Parameters that change more rapidly get smaller learning rates, and vice versa, which helps stabilize training.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the networks adjust their learning speed for different parts.
    /// 
    /// Think of second moments as:
    /// - A record of how wildly each part of the network has been changing
    /// - Parts that change a lot get smaller updates (to avoid instability)
    /// - Parts that change little get larger updates (to learn faster)
    /// - This adaptive approach helps GANs train more reliably
    /// 
    /// It's like automatically adjusting the sensitivity of different controls
    /// based on how jumpy they've been in the past.
    /// </para>
    /// </remarks>
    private Vector<T> _secondMoment;

    /// <summary>
    /// Gets or sets the current value of beta1 raised to the power of the iteration count for Adam optimizer.
    /// </summary>
    /// <value>The current beta1 power value.</value>
    /// <remarks>
    /// <para>
    /// This value is used for bias correction in the Adam optimizer. The beta1 parameter controls the
    /// exponential decay rate for the first moment estimates (momentum). The power value is updated at
    /// each training step and helps correct the bias in the early stages of training.
    /// </para>
    /// <para><b>For Beginners:</b> This is a technical value that helps the optimizer work correctly.
    /// 
    /// Think of beta1Power as:
    /// - A correction factor for the optimizer
    /// - It helps make the early stages of training more accurate
    /// - Without it, the network might learn too slowly at the beginning
    /// - It's automatically adjusted during training
    /// 
    /// This is part of what makes modern optimizers like Adam so effective for
    /// training complex models like GANs.
    /// </para>
    /// </remarks>
    private T _beta1Power;

    /// <summary>
    /// Gets or sets the current value of beta2 raised to the power of the iteration count for Adam optimizer.
    /// </summary>
    /// <value>The current beta2 power value.</value>
    /// <remarks>
    /// <para>
    /// This value is used for bias correction in the Adam optimizer. The beta2 parameter controls the
    /// exponential decay rate for the second moment estimates. The power value is updated at
    /// each training step and helps correct the bias in the early stages of training.
    /// </para>
    /// <para><b>For Beginners:</b> This is another technical value that helps the optimizer work correctly.
    /// 
    /// Think of beta2Power as:
    /// - A companion to beta1Power for the second moment estimates
    /// - It ensures the adaptive learning rates are accurate from the start
    /// - Without it, the adaptation might be too aggressive or too conservative initially
    /// - Like beta1Power, it's automatically adjusted during training
    /// 
    /// These correction factors are what make Adam one of the preferred optimizers
    /// for training GANs.
    /// </para>
    /// </remarks>
    private T _beta2Power;

    /// <summary>
    /// Gets or sets the current learning rate for the optimizer.
    /// </summary>
    /// <value>A double representing the current learning rate.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines the step size at each iteration while moving toward a minimum of the loss function.
    /// In this implementation, the learning rate can decay over time and can be adapted based on training progress.
    /// Finding the right learning rate is critical for effective GAN training.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how big the adjustments are during training.
    /// 
    /// Think of the learning rate as:
    /// - The size of the steps the networks take when learning
    /// - Too large, and they might overshoot and never find the best solution
    /// - Too small, and training will take forever
    /// - In this implementation, it gradually decreases over time
    /// 
    /// The learning rate is one of the most important hyperparameters to tune
    /// when training GANs, as they can be notoriously unstable.
    /// </para>
    /// </remarks>
    private double _currentLearningRate = 0.001;

    /// <summary>
    /// Gets or sets the rate at which the learning rate decays during training.
    /// </summary>
    /// <value>A double representing the learning rate decay factor.</value>
    /// <remarks>
    /// <para>
    /// The learning rate decay factor determines how quickly the learning rate decreases over time.
    /// A value close to 1 means the learning rate decreases very slowly, while a smaller value
    /// causes it to decrease more rapidly. Decreasing the learning rate over time can help the
    /// model converge to a more optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the step size shrinks during training.
    /// 
    /// Think of learning rate decay as:
    /// - A factor that gradually reduces the learning rate
    /// - At the beginning, large steps help explore the solution space quickly
    /// - As training progresses, smaller steps help fine-tune the solution
    /// - A value of 0.9999 means the learning rate decreases very slowly
    /// 
    /// This is like starting with bold brush strokes when painting, then gradually
    /// switching to finer brushes for the details.
    /// </para>
    /// </remarks>
    private double _learningRateDecay = 0.9999;

    /// <summary>
    /// Gets or sets the list of recent generator loss values for monitoring training progress.
    /// </summary>
    /// <value>A list of loss values from recent training iterations.</value>
    /// <remarks>
    /// <para>
    /// This list stores the loss values from recent generator training steps. It's used to monitor
    /// training progress, detect instabilities, and potentially adapt hyperparameters like the
    /// learning rate. The list is limited to a fixed size to focus on recent performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how well the generator is improving over time.
    /// 
    /// Think of generatorLosses as:
    /// - A record of how well the generator has been doing recently
    /// - If the losses are decreasing, the generator is improving
    /// - If they're increasing or unstable, something might be wrong
    /// - This history can be used to automatically adjust the training process
    /// 
    /// It's like keeping a score history for a sports team to track
    /// improvement and make coaching adjustments.
    /// </para>
    /// </remarks>
    private List<T> _generatorLosses = [];

    /// <summary>
    /// Gets the generator network that creates synthetic data.
    /// </summary>
    /// <value>A convolutional neural network that generates synthetic data.</value>
    /// <remarks>
    /// <para>
    /// The Generator is a neural network that takes random noise as input and produces synthetic data
    /// (such as images) as output. During training, it learns to create increasingly realistic data
    /// that can fool the Discriminator. In this implementation, it's specifically a convolutional
    /// neural network, which is well-suited for image generation tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "forger" network that creates fake data.
    /// 
    /// Think of the Generator as:
    /// - An artist creating paintings from random starting points
    /// - It takes random noise (like static) and shapes it into structured data
    /// - Its goal is to create outputs so realistic they fool the Discriminator
    /// - It improves by learning from the feedback of the Discriminator
    /// 
    /// For example, in an image generation task, the Generator might start by creating
    /// blurry, unrealistic images, but gradually learn to create sharp, detailed,
    /// and realistic images as training progresses.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network that distinguishes between real and synthetic data.
    /// </summary>
    /// <value>A convolutional neural network that classifies data as real or synthetic.</value>
    /// <remarks>
    /// <para>
    /// The Discriminator is a neural network that takes data (either real or generated) as input
    /// and outputs a probability that the data is real. During training, it learns to better
    /// distinguish between real data and the Generator's synthetic data. In this implementation,
    /// it's specifically a convolutional neural network, which is well-suited for image classification tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "detective" network that tries to spot fakes.
    /// 
    /// Think of the Discriminator as:
    /// - An art expert examining paintings to determine if they're authentic
    /// - It analyzes data (like images) and gives a probability that it's real
    /// - Its goal is to correctly identify real data and detect generated fakes
    /// - It improves by learning from its mistakes
    /// 
    /// For example, in an image generation task, the Discriminator essentially
    /// answers the question "Is this a real photograph or a computer-generated image?"
    /// and becomes increasingly sophisticated in its ability to tell the difference.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="GenerativeAdversarialNetwork{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.</param>
    /// <param name="fitnessCalculator">The fitness calculator used to compute loss values during training.</param>
    /// <param name="inputType">The type of input the GAN will process.</param>
    /// <param name="initialLearningRate">The initial learning rate for the optimizer. Default is 0.001.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new Generative Adversarial Network with the specified generator and discriminator
    /// architectures. It also sets up the optimization parameters and initializes tracking collections for monitoring
    /// training progress. The GAN's architecture is a combination of the generator and discriminator architectures.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the complete GAN system with both networks.
    /// 
    /// When creating a new GAN:
    /// - You provide separate architectures for the generator and discriminator
    /// - The fitnessCalculator determines how performance is measured
    /// - The inputType specifies what kind of data the GAN will work with
    /// - The initialLearningRate controls how quickly the networks learn initially
    /// 
    /// Think of it like establishing the rules and roles for the forger and detective
    /// before their competition begins.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Generates a synthetic image from random noise input.
    /// </summary>
    /// <param name="noise">The random noise vector used as input to the generator.</param>
    /// <returns>A vector representing the generated image.</returns>
    /// <remarks>
    /// <para>
    /// This method passes a noise vector through the Generator network to produce a synthetic image.
    /// The noise vector serves as a seed for the generation process, with different noise vectors
    /// resulting in different generated images. The dimensionality and distribution of the noise
    /// vector are important factors in the diversity and quality of the generated images.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new synthetic image from random input.
    /// 
    /// Think of this method as:
    /// - Taking random inspiration (noise) and turning it into an artwork
    /// - Different random inputs will create different outputs
    /// - The quality and realism depend on how well-trained the Generator is
    /// - After training, this is the main function you'll use to create new content
    /// 
    /// For example, you might generate 100 different images by feeding 
    /// 100 different random noise vectors to this method.
    /// </para>
    /// </remarks>
    public Vector<T> GenerateImage(Vector<T> noise)
    {
        return Generator.Predict(noise);
    }

    /// <summary>
    /// Evaluates how real an image appears to the discriminator.
    /// </summary>
    /// <param name="image">The image vector to evaluate.</param>
    /// <returns>A value between 0 and 1 representing the discriminator's confidence that the image is real.</returns>
    /// <remarks>
    /// <para>
    /// This method passes an image through the Discriminator network to determine how likely the
    /// image is to be real rather than generated. A value close to 1 indicates the Discriminator
    /// thinks the image is real, while a value close to 0 indicates it thinks the image is fake.
    /// This method can be used to evaluate both real images and images created by the Generator.
    /// </para>
    /// <para><b>For Beginners:</b> This checks how convincing an image is to the discriminator.
    /// 
    /// Think of this method as:
    /// - Asking the art detective, "Do you think this painting is real?"
    /// - The answer is a number between 0 and 1
    /// - 0 means "definitely fake"
    /// - 1 means "definitely real" 
    /// - 0.5 means "unsure - could be either"
    /// 
    /// This can be used to evaluate how well the generator is performing
    /// by seeing if its images can fool the discriminator.
    /// </para>
    /// </remarks>
    public T DiscriminateImage(Vector<T> image)
    {
        var result = Discriminator.Predict(image);
        return result[0];
    }

    /// <summary>
    /// Performs one step of training for both the generator and discriminator.
    /// </summary>
    /// <param name="realImages">A vector of real images for training the discriminator.</param>
    /// <param name="noise">A vector of random noise for training the generator.</param>
    /// <remarks>
    /// <para>
    /// This method performs one complete training iteration for the GAN. It first trains the Discriminator
    /// on both real images and fake images generated by the Generator. Then it trains the Generator to
    /// create images that can fool the Discriminator. This adversarial training process is the core of
    /// how GANs learn to generate realistic data.
    /// </para>
    /// <para><b>For Beginners:</b> This is one round of the competition between the forger and detective.
    /// 
    /// The training step works like this:
    /// - First, the detective (discriminator) learns to better distinguish real from fake:
    ///   1. It examines real images and learns to recognize them
    ///   2. It examines fake images from the generator and learns to spot them
    /// - Then, the forger (generator) gets its turn:
    ///   3. It creates new fake images from random noise
    ///   4. It learns from how well these fakes fooled the discriminator
    /// 
    /// This back-and-forth process is what drives both networks to improve.
    /// You'll typically run thousands of these training steps.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Creates a vector filled with a specified value, typically used for labels.
    /// </summary>
    /// <param name="length">The length of the vector to create.</param>
    /// <param name="value">The value to fill the vector with.</param>
    /// <returns>A vector of the specified length filled with the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This utility method creates a vector of a specified length and fills it with a single value.
    /// In the context of GANs, it's typically used to create label vectors where 1 represents "real"
    /// and 0 represents "fake". These label vectors are used as targets during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This creates training labels (what the network should aim for).
    /// 
    /// Think of this method as:
    /// - Creating a list of "correct answers" for training
    /// - For real images, the correct answer is usually 1 ("real")
    /// - For fake images, the correct answer is usually 0 ("fake")
    /// - When training the generator to fool the discriminator, we use 1 ("real") as the target
    /// 
    /// This is like creating a grading key that the networks try to match during training.
    /// </para>
    /// </remarks>
    private static Vector<T> CreateLabelVector(int length, T value)
    {
        return new Vector<T>(Enumerable.Repeat(value, length).ToArray());
    }

    /// <summary>
    /// Trains the discriminator on a batch of images with corresponding labels.
    /// </summary>
    /// <param name="images">The images to train on, either real or generated.</param>
    /// <param name="labels">The labels indicating whether each image is real (1) or fake (0).</param>
    /// <returns>The loss value for this training step.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the Discriminator network to better distinguish between real and fake images.
    /// It computes predictions for the given images, calculates the loss between these predictions and
    /// the expected labels, and updates the Discriminator's weights to reduce this loss. The goal is
    /// for the Discriminator to assign high scores to real images and low scores to fake ones.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the detective to better spot real and fake images.
    /// 
    /// The discriminator training process:
    /// - The discriminator examines a batch of images and makes its predictions
    /// - These predictions are compared to the correct answers (labels)
    /// - The discriminator calculates how wrong it was (the loss)
    /// - It adjusts its internal parameters to make better predictions next time
    /// 
    /// For example, if the discriminator mistakenly thinks a fake image is real,
    /// it will adjust its parameters to be more skeptical of similar images in the future.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Trains the generator to create images that can fool the discriminator.
    /// </summary>
    /// <param name="noise">The random noise vector used as input to the generator.</param>
    /// <param name="targetLabels">The target labels, typically all 1's to make the discriminator think the generated images are real.</param>
    /// <returns>The loss value for this training step.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the Generator network to create images that the Discriminator will classify as real.
    /// It first generates fake images from the noise input, then passes these images through the Discriminator.
    /// The loss is calculated based on how well these fake images fool the Discriminator. The Generator's weights
    /// are then updated to reduce this loss, making it better at creating realistic images. During this process,
    /// the Discriminator's weights are frozen, as we're only training the Generator.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the forger to create more convincing fake images.
    /// 
    /// The generator training process:
    /// - The generator creates fake images from random noise
    /// - These images are evaluated by the discriminator
    /// - The generator's goal is to make images that the discriminator thinks are real
    /// - It adjusts its parameters to make more convincing images next time
    /// - Only the generator learns in this step - the discriminator is temporarily frozen
    /// 
    /// This is where the generator learns to create more realistic content
    /// based on the discriminator's feedback.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Calculates the gradients for backpropagation based on predictions and target values.
    /// </summary>
    /// <param name="predictions">The predicted values from the network.</param>
    /// <param name="targets">The target values (labels).</param>
    /// <returns>A vector of gradients for each prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the gradients for the network based on the difference between the
    /// predicted values and the target values. In a more complex implementation, this calculation
    /// would depend on the specific loss function being used. These gradients are then used to
    /// update the network's weights during backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how much to adjust the network based on its errors.
    /// 
    /// Think of the gradient calculation as:
    /// - Measuring how wrong each prediction was
    /// - Determining the direction and amount to adjust each parameter
    /// - The bigger the error, the larger the gradient and adjustment
    /// - This guides the learning process toward better predictions
    /// 
    /// For example, if the discriminator predicted 0.3 for a real image (label 1),
    /// the gradient would indicate it needs to adjust to give higher scores to similar images.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Applies calculated gradients to update the parameters of a network.
    /// </summary>
    /// <param name="network">The neural network to update.</param>
    /// <param name="gradients">The gradients to apply.</param>
    /// <remarks>
    /// <para>
    /// This method applies the calculated gradients to update the parameters of the specified network.
    /// It implements the Adam optimizer, which is an adaptive optimization algorithm well-suited for
    /// GAN training. The method includes gradient clipping to prevent exploding gradients, momentum
    /// to smooth updates, and adaptive learning rates based on the history of gradients. It also
    /// includes learning rate decay and bias correction for more stable training.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the network's internal values based on its learning.
    /// 
    /// Think of applying gradients as:
    /// - Making adjustments to the network's internal knobs and dials
    /// - Using an advanced algorithm (Adam) that adapts to how training is progressing
    /// - Preventing wild adjustments that could destabilize training
    /// - Gradually reducing the size of adjustments as training progresses
    /// 
    /// This method contains several advanced techniques that help GANs train more
    /// stably, which is critical since they can be notoriously difficult to train.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Makes a prediction using the current state of the Generative Adversarial Network.
    /// </summary>
    /// <param name="input">The input vector (typically random noise) to make a prediction for.</param>
    /// <returns>The generated output vector (typically an image).</returns>
    /// <remarks>
    /// <para>
    /// In the context of a GAN, "predict" typically means generating synthetic data from an input noise vector.
    /// This method is essentially a wrapper around the GenerateImage method, emphasizing that in a GAN,
    /// prediction is the process of generating new synthetic data rather than classifying existing data.
    /// </para>
    /// <para><b>For Beginners:</b> This creates new data from random input, like the GenerateImage method.
    /// 
    /// In a GAN, "prediction" means:
    /// - Taking random noise and transforming it into structured data
    /// - Using the Generator to create something new
    /// - This is different from typical neural networks where prediction means classification
    /// 
    /// This method is provided to conform to the NeuralNetworkBase interface, but
    /// functionally it does the same thing as GenerateImage.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        // In a GAN, "predict" typically means generating an image
        return GenerateImage(input);
    }

    /// <summary>
    /// Updates the parameters of both the generator and discriminator networks.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update both networks with.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of both the Generator and Discriminator networks. It splits the input
    /// parameter vector into two parts - one for the Generator and one for the Discriminator - and applies these
    /// updates to the respective networks. This is typically used when loading a pre-trained GAN or when
    /// implementing advanced training strategies that involve directly manipulating the networks' parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This updates both networks with new parameter values.
    /// 
    /// When updating parameters:
    /// - The input is a long list of numbers representing all values in both networks
    /// - The method divides this list between the generator and discriminator
    /// - Each network gets its own chunk of values
    /// - The networks use these values to update their internal settings
    /// 
    /// This method is primarily used when loading a saved model or implementing
    /// advanced optimization techniques rather than during normal training.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // Split parameters between generator and discriminator
        int generatorParamCount = Generator.GetParameterCount();
        var generatorParams = parameters.SubVector(0, generatorParamCount);
        var discriminatorParams = parameters.SubVector(generatorParamCount, parameters.Length - generatorParamCount);

        Generator.UpdateParameters(generatorParams);
        Discriminator.UpdateParameters(discriminatorParams);
    }

    /// <summary>
    /// Serializes the Generative Adversarial Network to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Generative Adversarial Network to a binary stream. It separately
    /// serializes the Generator and Discriminator networks, including labels to identify each part.
    /// This allows the GAN to be saved to disk and later restored with its trained parameters intact.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the complete GAN to a file so you can use it later.
    /// 
    /// When saving the GAN:
    /// - It separately saves both the generator and discriminator
    /// - Each part is labeled so they can be correctly loaded later
    /// - All weights and settings for both networks are preserved
    /// 
    /// This is like taking a snapshot of the entire system - including both the
    /// forger and detective networks. You can later load this snapshot to use
    /// the trained GAN without having to train it again.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Deserializes the Generative Adversarial Network from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the data format is invalid or unexpected.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Generative Adversarial Network from a binary stream. It looks for
    /// the labeled Generator and Discriminator sections and deserializes each network accordingly. This allows
    /// a previously saved GAN to be restored from disk with all its trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a previously saved GAN from a file.
    /// 
    /// When loading the GAN:
    /// - It looks for the labeled sections for the generator and discriminator
    /// - It loads each network with its saved weights and settings
    /// - It verifies that the data is in the expected format
    /// 
    /// This lets you use a previously trained GAN without having to train it again.
    /// It's like restoring the complete snapshot of both networks.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the layers of the Generative Adversarial Network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is overridden from the base class but is empty because a GAN doesn't use layers directly.
    /// Instead, the GAN architecture consists of two separate neural networks (Generator and Discriminator)
    /// that each have their own layers. These networks are initialized separately in the constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This method is empty because GANs work differently from standard neural networks.
    /// 
    /// Unlike traditional neural networks:
    /// - GANs don't have a single sequence of layers
    /// - Instead, they consist of two separate networks (Generator and Discriminator)
    /// - Each of these networks has its own layers
    /// - These networks are initialized separately in the constructor
    /// 
    /// This method is only included because it's required by the base class,
    /// but it doesn't need to do anything in a GAN implementation.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // GAN doesn't use layers directly, so this method is empty
    }
}