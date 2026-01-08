namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Self-Organizing Map, which is an unsupervised neural network that produces a low-dimensional representation of input data.
/// </summary>
/// <remarks>
/// <para>
/// A Self-Organizing Map (SOM), also known as a Kohonen map, is a type of artificial neural network that
/// uses unsupervised learning to produce a low-dimensional (typically two-dimensional) representation
/// of higher-dimensional input data. SOMs preserve the topological properties of the input space, meaning
/// that similar inputs will be mapped to nearby neurons in the output map. This makes SOMs useful for
/// visualization, clustering, and dimensionality reduction of complex data.
/// </para>
/// <para><b>For Beginners:</b> A Self-Organizing Map is like a smart way to arrange data on a map based on similarities.
/// 
/// Think of it like organizing books on a bookshelf:
/// - You have many books (input data) with different characteristics
/// - You want to arrange them so similar books are placed near each other
/// - Over time, you develop a system where sci-fi books are in one section, romance in another, etc.
/// 
/// A SOM works in a similar way:
/// - It takes complex data with many attributes
/// - It creates a 2D "map" where each location represents certain characteristics
/// - Similar data points end up mapped to nearby locations
/// - Different regions of the map represent different types of data
/// 
/// This is useful for:
/// - Visualizing complex data with many dimensions
/// - Finding natural groupings (clusters) in data
/// - Reducing complex data to simpler patterns
/// - Discovering relationships that might not be obvious
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SelfOrganizingMap<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the weight matrix representing the connection strengths between input dimensions and map neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The weights matrix defines the position of each map neuron in the input space. Each row of the matrix
    /// corresponds to a neuron in the map, and each column corresponds to a dimension of the input data.
    /// During training, these weights are adjusted to better represent the distribution of the input data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the "memory" of each location on the map.
    /// 
    /// Each position on the map has a set of weights that:
    /// - Represent what kind of data that position "expects" to see
    /// - Are adjusted during training to better match input data
    /// - Eventually define what that region of the map represents
    /// 
    /// For example, if your data describes books with attributes like "number of pages", "publication year", and "reading level":
    /// - Each position on the map would have weights for all these attributes
    /// - Some positions might develop weights representing "short, recent, easy-to-read books"
    /// - Others might develop weights for "long, classic, advanced books"
    /// </para>
    /// </remarks>
    private Matrix<T> _weights { get; set; }

    /// <summary>
    /// Gets or sets the width of the map (number of neurons horizontally).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The map width defines the horizontal dimension of the two-dimensional grid of neurons that form the SOM.
    /// Together with the map height, it determines the total number of neurons in the map and affects the
    /// resolution and detail with which the input space can be represented.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many columns are in your map grid.
    /// 
    /// For example:
    /// - A map width of 10 means your map has 10 positions across
    /// - Larger widths allow for more detailed maps but require more computation
    /// - The width and height together determine how many different "positions" your map can represent
    /// 
    /// Think of it like the resolution of a screen - more pixels (neurons) allows for a more detailed picture.
    /// </para>
    /// </remarks>
    private int _mapWidth { get; set; }

    /// <summary>
    /// Gets or sets the height of the map (number of neurons vertically).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The map height defines the vertical dimension of the two-dimensional grid of neurons that form the SOM.
    /// Together with the map width, it determines the total number of neurons in the map and affects the
    /// resolution and detail with which the input space can be represented.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many rows are in your map grid.
    /// 
    /// For example:
    /// - A map height of 8 means your map has 8 positions down
    /// - With a width of 10 and height of 8, you would have 80 total positions on your map
    /// - More positions allow for more nuanced organization of your data
    /// 
    /// Most SOMs try to use a fairly square shape (similar width and height) for balanced representation.
    /// </para>
    /// </remarks>
    private int _mapHeight { get; set; }

    /// <summary>
    /// Gets or sets the dimensionality of the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input dimension represents the number of features or attributes in the input data. Each input
    /// vector will have this many elements, and each neuron in the map will have this many weights.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different attributes or measurements each data point has.
    /// 
    /// For example:
    /// - If your data is about people, you might have dimensions for height, weight, age, etc.
    /// - If your data is about music, you might have dimensions for tempo, volume, pitch, etc.
    /// - If your data is about images, each pixel might be a dimension (making for very high-dimensional data)
    /// 
    /// The SOM's job is to take this high-dimensional data and organize it on a simple 2D map,
    /// making it easier to visualize and understand.
    /// </para>
    /// </remarks>
    private int _inputDimension { get; set; }

    private int _totalEpochs;
    private int _currentEpoch;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfOrganizingMap{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the SOM.</param>
    /// <exception cref="ArgumentException">Thrown when input dimension or output size is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Self-Organizing Map based on the provided architecture. It extracts
    /// the input dimension from the architecture's input size and calculates appropriate map dimensions
    /// (width and height) based on the architecture's output size. It attempts to make the map as square
    /// as possible for better visualization and organization.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Self-Organizing Map with its basic structure.
    /// 
    /// When creating a new SOM:
    /// - The architecture tells us how many input dimensions we have (how many attributes each data point has)
    /// - The architecture also suggests how many total positions we want on our map
    /// - The constructor tries to make the map as square as possible (e.g., 10×10 rather than 5×20)
    /// - It may adjust the total map size slightly to make a perfect square if needed
    /// 
    /// Once the dimensions are set, it creates weight values for each position on the map.
    /// These weights are initially random and will be adjusted during training.
    /// </para>
    /// </remarks>
    public SelfOrganizingMap(NeuralNetworkArchitecture<T> architecture, int totalEpochs = 1000, ILossFunction<T>? lossFunction = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        // Get input dimension from the architecture
        _inputDimension = architecture.InputSize;

        if (_inputDimension <= 0)
        {
            throw new ArgumentException("Input dimension must be greater than zero for SOM.");
        }

        // Get map size from the output size
        int mapSize = architecture.OutputSize;

        if (mapSize <= 0)
        {
            throw new ArgumentException("Map size (output size) must be greater than zero for SOM.");
        }

        // Calculate map dimensions - allow for rectangular maps
        int totalNeurons = mapSize;
        double aspectRatio = 1.6; // Golden ratio, but this could be adjustable

        _mapWidth = (int)Math.Round(Math.Sqrt(totalNeurons * aspectRatio));
        _mapHeight = (int)Math.Round(totalNeurons / (double)_mapWidth);

        // Adjust to ensure we have exactly the right number of neurons
        while (_mapWidth * _mapHeight < totalNeurons)
        {
            if (_mapWidth / (double)_mapHeight < aspectRatio)
                _mapWidth++;
            else
                _mapHeight++;
        }

        while (_mapWidth * _mapHeight > totalNeurons)
        {
            if (_mapWidth / (double)_mapHeight > aspectRatio)
                _mapWidth--;
            else
                _mapHeight--;
        }

        if (_mapWidth * _mapHeight != totalNeurons)
        {
            // Use a logger instead of Console.WriteLine
            Console.WriteLine($"Adjusted map size from {totalNeurons} to {_mapWidth * _mapHeight} to maintain aspect ratio.");
        }

        _weights = new Matrix<T>(_mapWidth * _mapHeight, _inputDimension);
        _totalEpochs = totalEpochs;
        _currentEpoch = 0;

        InitializeWeights();
        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers. In a SOM, this method is typically empty as SOMs use direct weight and map parameters rather than standard neural network layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SOMs differ from feedforward neural networks in that they don't use a layer-based computation model.
    /// Instead, they directly manipulate weights and use a competitive learning approach where neurons
    /// compete to respond to input patterns. Therefore, this method is typically empty or performs
    /// specialized initialization for SOMs.
    /// </para>
    /// <para><b>For Beginners:</b> SOMs work differently from standard neural networks.
    /// 
    /// While standard neural networks process data through sequential layers:
    /// - SOMs use a competitive approach where neurons "compete" to respond to input
    /// - They don't use the same layer concept as feedforward networks
    /// - They operate directly on the weights connecting the input to the map neurons
    /// 
    /// That's why this method is empty - the SOM initializes its weights directly
    /// rather than creating a sequence of layers like a standard neural network.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // SOM doesn't use layers in the same way as feedforward networks
        // Instead, we'll initialize the weights directly
    }

    /// <summary>
    /// Initializes the weights of the SOM with random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights of each neuron in the map with random values between 0 and 1.
    /// These initial weights represent random positions in the input space, providing a starting point
    /// for the learning process. During training, these weights will be adjusted to better represent
    /// the distribution of the input data.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives random starting values to all map positions.
    /// 
    /// When initializing weights:
    /// - Each position on the map gets random values for each input dimension
    /// - These random values give the SOM a starting point for learning
    /// - Without this randomness, all positions would start identical and couldn't differentiate
    /// 
    /// Think of it like randomly arranging books on a shelf before you start organizing them.
    /// This initial randomness is important because it allows different areas of the map to
    /// specialize in different types of input patterns during training.
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        for (int i = 0; i < _mapWidth * _mapHeight; i++)
        {
            for (int j = 0; j < _inputDimension; j++)
            {
                _weights[i, j] = NumOps.FromDouble(Random.NextDouble());
            }
        }
    }

    /// <summary>
    /// Finds the index of the neuron that best matches the input vector.
    /// </summary>
    /// <param name="input">The input vector to match.</param>
    /// <returns>The index of the best matching neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method finds the neuron in the map that is closest to the input vector in the input space.
    /// It computes the Euclidean distance between the input vector and each neuron's weight vector,
    /// and returns the index of the neuron with the minimum distance. This neuron is called the
    /// Best Matching Unit (BMU).
    /// </para>
    /// <para><b>For Beginners:</b> This method finds which position on the map is most similar to the input data.
    /// 
    /// The process:
    /// - Calculate how "different" the input is from each position on the map
    /// - Find the position with the smallest difference (closest match)
    /// - Return the index of this position (called the Best Matching Unit or BMU)
    /// 
    /// Think of it like finding which section of a bookstore would best fit a new book.
    /// You compare the book to each section and put it where it fits best.
    /// 
    /// This matching process is at the heart of how SOMs categorize and organize data.
    /// </para>
    /// </remarks>
    private int FindBestMatchingUnit(Vector<T> input)
    {
        int bmu = 0;
        T minDistance = NumOps.MaxValue;

        for (int i = 0; i < _mapWidth * _mapHeight; i++)
        {
            T distance = CalculateDistance(input, _weights.GetRow(i));
            if (NumOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                bmu = i;
            }
        }

        return bmu;
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="v1">The first vector.</param>
    /// <param name="v2">The second vector.</param>
    /// <returns>The Euclidean distance between the vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Euclidean distance between two vectors, which is the square root of the
    /// sum of squared differences between corresponding elements. This distance measure is used to quantify
    /// how similar two data points are in the input space.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how different two data points are from each other.
    /// 
    /// The Euclidean distance:
    /// - Compares each attribute of the two data points
    /// - Squares the differences to make all values positive
    /// - Adds up all the squared differences
    /// - Takes the square root to get the final distance
    /// 
    /// For example, if comparing two books with attributes for page count and publication year:
    /// - Book 1: 300 pages, published in 2010
    /// - Book 2: 400 pages, published in 2020
    /// - The calculation would be: sqrt[(300-400)² + (2010-2020)²] = sqrt(10,100 + 100) = sqrt(10,200) ≈ 101
    /// 
    /// A smaller distance means the data points are more similar.
    /// </para>
    /// </remarks>
    private T CalculateDistance(Vector<T> v1, Vector<T> v2)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            T diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Updates the weights of neurons in the neighborhood of the best matching unit.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="bmu">The index of the best matching unit.</param>
    /// <param name="learningRate">The current learning rate.</param>
    /// <param name="radius">The current neighborhood radius.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights of neurons within the specified radius of the best matching unit.
    /// The update strength depends on both the learning rate and the distance from the BMU, with neurons
    /// closer to the BMU being updated more strongly. This approach helps preserve the topological
    /// properties of the input space in the resulting map.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts map positions to better represent the input data.
    /// 
    /// When updating weights:
    /// - The best matching position (BMU) is updated most strongly
    /// - Nearby positions are also updated, but less strongly
    /// - Positions outside the radius are not updated at all
    /// 
    /// This neighborhood approach is key to the SOM's ability to organize data:
    /// - It creates regions of similar neurons on the map
    /// - It preserves relationships between different data types
    /// - It forms a smooth transition between different categories
    /// 
    /// For example, if a sci-fi book is matched to a position, nearby positions will also
    /// become slightly more "sci-fi-like," creating a region for similar books.
    /// </para>
    /// </remarks>
    private void UpdateWeights(Vector<T> input, int bmu, T learningRate, T radius)
    {
        int bmuX = bmu % _mapWidth;
        int bmuY = bmu / _mapWidth;

        for (int x = 0; x < _mapWidth; x++)
        {
            for (int y = 0; y < _mapHeight; y++)
            {
                int index = y * _mapWidth + x;
                T distance = NumOps.Sqrt(NumOps.FromDouble((x - bmuX) * (x - bmuX) + (y - bmuY) * (y - bmuY)));

                if (NumOps.LessThan(distance, radius))
                {
                    T influence = CalculateInfluence(distance, radius);
                    Vector<T> weightDelta = CalculateWeightDelta(input, _weights.GetRow(index), learningRate, influence);
                    Vector<T> updatedWeights = new Vector<T>(_inputDimension);
                    for (int i = 0; i < _inputDimension; i++)
                    {
                        updatedWeights[i] = NumOps.Add(_weights[index, i], weightDelta[i]);
                    }

                    _weights.SetRow(index, updatedWeights);
                }
            }
        }
    }

    /// <summary>
    /// Calculates the current learning rate based on the initial rate and the current epoch.
    /// </summary>
    /// <param name="initialLearningRate">The initial learning rate.</param>
    /// <param name="currentEpoch">The current epoch number.</param>
    /// <param name="totalEpochs">The total number of epochs.</param>
    /// <returns>The current learning rate.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the learning rate for the current training epoch. The learning rate
    /// decreases exponentially over time, starting from the initial learning rate. This decreasing
    /// schedule allows the SOM to make large adjustments early in training and fine-tune the 
    /// representation in later epochs.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines how quickly the map adapts at each stage of training.
    /// 
    /// The learning rate:
    /// - Starts high, allowing big changes early in training
    /// - Gradually decreases over time
    /// - Ends much lower, allowing only fine adjustments in the final stages
    /// 
    /// This approach is like learning a new skill:
    /// - At first, you make big improvements quickly
    /// - As you get better, progress becomes more gradual and refined
    /// 
    /// This decreasing learning rate helps the SOM converge to a stable representation.
    /// </para>
    /// </remarks>
    private T CalculateLearningRate(T initialLearningRate, int currentEpoch, int totalEpochs)
    {
        return NumOps.Multiply(initialLearningRate, NumOps.Exp(NumOps.Negate(NumOps.FromDouble(currentEpoch / totalEpochs))));
    }

    /// <summary>
    /// Calculates the current neighborhood radius based on the initial radius and the current epoch.
    /// </summary>
    /// <param name="initialRadius">The initial neighborhood radius.</param>
    /// <param name="currentEpoch">The current epoch number.</param>
    /// <param name="timeConstant">The time constant for radius calculation.</param>
    /// <returns>The current neighborhood radius.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the neighborhood radius for the current training epoch. The radius
    /// decreases exponentially over time, starting from the initial radius. This decreasing schedule
    /// allows the SOM to organize broadly at the beginning and then focus on local refinements in
    /// later epochs, promoting global ordering followed by fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines how far the influence spreads at each stage of training.
    /// 
    /// The neighborhood radius:
    /// - Starts large, affecting many positions during early training
    /// - Gradually shrinks over time
    /// - Ends much smaller, affecting only very close positions in the final stages
    /// 
    /// This approach:
    /// - First creates a broad organization across the entire map
    /// - Then fine-tunes specific regions with greater precision
    /// - Results in a well-organized map that preserves relationships between data
    /// 
    /// It's like first arranging books by general category (fiction, non-fiction, etc.),
    /// then organizing within each category more precisely.
    /// </para>
    /// </remarks>
    private T CalculateRadius(T initialRadius, int currentEpoch, T timeConstant)
    {
        return NumOps.Multiply(initialRadius, NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.FromDouble(currentEpoch), timeConstant))));
    }

    /// <summary>
    /// Calculates the influence of the best matching unit on a neuron based on distance.
    /// </summary>
    /// <param name="distance">The distance from the neuron to the best matching unit.</param>
    /// <param name="radius">The current neighborhood radius.</param>
    /// <returns>The influence factor.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates how strongly a neuron should be influenced by the current input based on
    /// its distance from the best matching unit (BMU). The influence decreases exponentially with distance,
    /// following a Gaussian neighborhood function. Neurons closer to the BMU are influenced more strongly
    /// than those farther away.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how much a position is influenced based on its distance from the BMU.
    /// 
    /// The influence:
    /// - Is strongest at the BMU itself (distance = 0)
    /// - Decreases rapidly as you move away from the BMU
    /// - Uses a bell curve shape (Gaussian function) for smooth transition
    /// 
    /// For example:
    /// - The BMU might be updated with 100% strength
    /// - A position 1 step away might be updated with 60% strength
    /// - A position 2 steps away might be updated with 14% strength
    /// - A position far away might receive practically no update
    /// 
    /// This gradual decrease in influence creates smooth regions on the map rather than
    /// sharp boundaries between different types of data.
    /// </para>
    /// </remarks>
    private T CalculateInfluence(T distance, T radius)
    {
        T distanceSquared = NumOps.Multiply(distance, distance);
        T radiusSquared = NumOps.Multiply(radius, radius);

        return NumOps.Exp(NumOps.Negate(NumOps.Divide(distanceSquared, NumOps.Multiply(NumOps.FromDouble(2), radiusSquared))));
    }

    /// <summary>
    /// Calculates the weight delta for updating a neuron's weights.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="weight">The neuron's current weight vector.</param>
    /// <param name="learningRate">The current learning rate.</param>
    /// <param name="influence">The influence factor based on distance from the BMU.</param>
    /// <returns>The weight delta vector.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates how much each weight of a neuron should be updated based on the current input,
    /// learning rate, and influence factor. The update moves the neuron's weights closer to the input vector,
    /// with the magnitude of the change determined by both the learning rate and the influence factor.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how much each position's weights should change.
    /// 
    /// The weight delta:
    /// - Is the difference between the input data and the current weights
    /// - Is scaled by both learning rate and influence
    /// - Determines how much the position moves toward matching the input
    /// 
    /// This calculation ensures that:
    /// - Positions move toward matching their inputs
    /// - Changes are proportional to learning rate and influence
    /// - The BMU moves more than distant positions
    /// 
    /// The formula (learningRate * influence * (input - weight)) moves each weight
    /// some fraction of the way toward matching the corresponding input value.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateWeightDelta(Vector<T> input, Vector<T> weight, T learningRate, T influence)
    {
        Vector<T> delta = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            delta[i] = NumOps.Multiply(NumOps.Multiply(learningRate, influence), NumOps.Subtract(input[i], weight[i]));
        }

        return delta;
    }

    /// <summary>
    /// Updates the parameters of the SOM from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <exception cref="ArgumentException">Thrown when the parameter vector length doesn't match the expected number of weights.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the SOM's weight matrix from a flat parameter vector. The parameter vector must have
    /// a length equal to (mapWidth × mapHeight) × inputDimension. While SOMs typically use specialized training
    /// algorithms (see the Train method), this method allows for direct parameter updates, which can be useful
    /// for optimization algorithms or parameter transfer scenarios.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows direct parameter updates when needed.
    ///
    /// While SOMs typically use competitive learning:
    /// - SOMs use a competitive learning approach
    /// - They update based on neighborhood and distance
    /// - They directly adjust weights based on similarity to input
    ///
    /// However, this method allows direct parameter updates for certain optimization
    /// algorithms or parameter transfer scenarios. For typical SOM training, use the Train method instead.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int expectedLength = (_mapWidth * _mapHeight) * _inputDimension;

        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Parameter vector length mismatch. Expected {expectedLength} parameters but got {parameters.Length}.", nameof(parameters));
        }

        int paramIndex = 0;

        for (int i = 0; i < _mapWidth * _mapHeight; i++)
        {
            for (int j = 0; j < _inputDimension; j++)
            {
                _weights[i, j] = parameters[paramIndex++];
            }
        }
    }

    /// <summary>
    /// Predicts the output for a given input using the trained Self-Organizing Map.
    /// </summary>
    /// <param name="input">The input tensor to predict.</param>
    /// <returns>A tensor representing the prediction result.</returns>
    /// <exception cref="ArgumentException">Thrown when the input shape doesn't match the expected input dimension.</exception>
    /// <remarks>
    /// <para>
    /// This method finds the Best Matching Unit (BMU) for the given input and returns a one-hot encoded
    /// tensor representing the BMU's position in the map. The output tensor has a size equal to the
    /// total number of neurons in the map (_mapWidth * _mapHeight).
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best matching position on the map for new input data.
    /// 
    /// When predicting:
    /// - It checks if the input data has the correct number of attributes
    /// - It finds the position on the map that best matches the input (the BMU)
    /// - It creates an output where only the BMU position is marked as active (1), and all others are inactive (0)
    /// 
    /// This output tells you which part of the map best represents the input data,
    /// which can be used for classification, clustering, or visualization.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Handle any rank input - flatten to 1D if needed
        var flatInput = input.Rank == 1 ? input : input.Reshape([input.Length]);

        if (flatInput.Length != _inputDimension)
        {
            throw new ArgumentException($"Input must have {_inputDimension} elements, but got {flatInput.Length}");
        }

        input = flatInput;

        int bmu = FindBestMatchingUnit(input.ToVector());

        // Create a one-hot encoded output tensor
        var output = new Tensor<T>(new[] { _mapWidth * _mapHeight });
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = i == bmu ? NumOps.One : NumOps.Zero;
        }

        return output;
    }

    /// <summary>
    /// Trains the Self-Organizing Map using the provided input.
    /// </summary>
    /// <param name="input">The input tensor to train on.</param>
    /// <param name="expectedOutput">The expected output tensor (not used in SOM training).</param>
    /// <exception cref="ArgumentException">Thrown when the input shape doesn't match the expected input dimension.</exception>
    /// <remarks>
    /// <para>
    /// This method performs one training step for the SOM. It finds the Best Matching Unit (BMU) for the input,
    /// calculates the current learning rate and neighborhood radius, and updates the weights of the neurons
    /// in the BMU's neighborhood. The learning rate and radius decrease over time to refine the map's organization.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the map based on new input data.
    /// 
    /// During training:
    /// - It checks if the input data has the correct number of attributes
    /// - It finds the best matching position (BMU) for the input
    /// - It calculates how much the map should change (learning rate) and how far the change should spread (radius)
    /// - It updates the map, with the BMU and nearby positions becoming more like the input data
    /// - It keeps track of how many times the map has been trained (epochs)
    /// 
    /// Over time, this process organizes the map so that similar inputs activate nearby positions.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Handle any rank input - flatten to 1D if needed
        var flatInput = input.Rank == 1 ? input : input.Reshape([input.Length]);

        if (flatInput.Length != _inputDimension)
        {
            throw new ArgumentException($"Input must have {_inputDimension} elements, but got {flatInput.Length}");
        }

        input = flatInput;

        // SOM training doesn't use expectedOutput, so we ignore it

        int bmu = FindBestMatchingUnit(input.ToVector());

        // Calculate current learning rate and radius
        T learningRate = CalculateLearningRate(NumOps.FromDouble(0.1), _currentEpoch, _totalEpochs);
        T radius = CalculateRadius(NumOps.FromDouble(Math.Max(_mapWidth, _mapHeight) / 2.0), _currentEpoch, NumOps.FromDouble(_totalEpochs / Math.Log(_mapWidth * _mapHeight)));

        // Update weights
        UpdateWeights(input.ToVector(), bmu, learningRate, radius);

        // Calculate and set the quantization error as the loss
        // This is the distance between the input and the BMU's weights
        Vector<T> bmuWeights = _weights.GetRow(bmu);
        LastLoss = CalculateDistance(input.ToVector(), bmuWeights);

        _currentEpoch++;
    }

    /// <summary>
    /// Gets the metadata of the Self-Organizing Map model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the SOM.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the SOM, including its type, dimensions, and training progress.
    /// It also includes serialized model data, which can be used to save or transfer the model state.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides a summary of the SOM's current state.
    /// 
    /// The metadata includes:
    /// - The type of model (Self-Organizing Map)
    /// - The number of input dimensions
    /// - The width and height of the map
    /// - The total number of training epochs planned
    /// - The current number of training epochs completed
    /// - A serialized version of the entire model (useful for saving or sharing the model)
    /// 
    /// This information is useful for keeping track of the model's configuration and training progress.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SelfOrganizingMap,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", _inputDimension },
                { "MapWidth", _mapWidth },
                { "MapHeight", _mapHeight },
                { "TotalEpochs", _totalEpochs },
                { "CurrentEpoch", _currentEpoch }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes the specific data of the Self-Organizing Map.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the SOM-specific data to a binary stream. It includes the map dimensions,
    /// training parameters, and the weights of all neurons in the map.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important information about the SOM.
    /// 
    /// It saves:
    /// - The number of input dimensions
    /// - The width and height of the map
    /// - The total number of training epochs
    /// - The current training epoch
    /// - All the weights for every position on the map
    /// 
    /// This allows the entire state of the SOM to be saved and later restored exactly as it was.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_inputDimension);
        writer.Write(_mapWidth);
        writer.Write(_mapHeight);
        writer.Write(_totalEpochs);
        writer.Write(_currentEpoch);

        // Serialize weights
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes the specific data of the Self-Organizing Map.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads SOM-specific data from a binary stream and reconstructs the map's state.
    /// It restores the map dimensions, training parameters, and the weights of all neurons in the map.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads all the important information about a previously saved SOM.
    /// 
    /// It loads:
    /// - The number of input dimensions
    /// - The width and height of the map
    /// - The total number of training epochs
    /// - The current training epoch
    /// - All the weights for every position on the map
    /// 
    /// This allows a previously saved SOM to be fully restored to continue training or make predictions.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _inputDimension = reader.ReadInt32();
        _mapWidth = reader.ReadInt32();
        _mapHeight = reader.ReadInt32();
        _totalEpochs = reader.ReadInt32();
        _currentEpoch = reader.ReadInt32();

        // Deserialize weights
        _weights = new Matrix<T>(_mapWidth * _mapHeight, _inputDimension);
        for (int i = 0; i < _weights.Rows; i++)
        {
            for (int j = 0; j < _weights.Columns; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Creates a new instance of the Self-Organizing Map with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Self-Organizing Map.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current Self-Organizing Map, including its architecture,
    /// weights, and training progress. The new instance is completely independent of the original,
    /// allowing modifications without affecting the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact copy of the current SOM.
    /// 
    /// The copy includes:
    /// - The same map dimensions (width and height)
    /// - The same input dimension
    /// - The same weights for all positions on the map
    /// - The same training progress (current epoch)
    /// 
    /// This is useful when you want to:
    /// - Create a backup before continuing training
    /// - Create variations of the same map for different purposes
    /// - Share the map while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SelfOrganizingMap<T>(Architecture, _totalEpochs, LossFunction);
    }
}
