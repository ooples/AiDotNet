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
    public SelfOrganizingMap(NeuralNetworkArchitecture<T> architecture) : base(architecture)
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
    
        // Calculate map dimensions - try to make it as square as possible
        _mapWidth = (int)Math.Sqrt(mapSize);
        _mapHeight = (int)Math.Ceiling(mapSize / (double)_mapWidth);
    
        // Adjust if not a perfect square
        if (_mapWidth * _mapHeight != mapSize)
        {
            // We'll use the closest perfect square for simplicity
            int perfectSquare = _mapWidth * _mapWidth;
            if (Math.Abs(perfectSquare - mapSize) < Math.Abs((_mapWidth + 1) * (_mapWidth + 1) - mapSize))
            {
                _mapWidth = _mapWidth;
                _mapHeight = _mapWidth; // Make it a perfect square
            }
            else
            {
                _mapWidth = _mapWidth + 1;
                _mapHeight = _mapWidth; // Make it a perfect square
            }
        
            // Log a warning that we're adjusting the map size
            Console.WriteLine($"Warning: Adjusting map size from {mapSize} to {_mapWidth * _mapHeight} to make it a perfect square.");
        }
    
        _weights = new Matrix<T>(_mapWidth * _mapHeight, _inputDimension);
    
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
    /// Processes the input through the SOM to find the best matching neuron and returns its weights.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The weight vector of the best matching neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the prediction process of the SOM. It finds the neuron in the map that
    /// best matches the input vector (the one with the smallest Euclidean distance to the input), and
    /// returns the weight vector of that neuron. This represents the SOM's "response" to the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds which position on the map best matches your input data.
    /// 
    /// The prediction process has two steps:
    /// 1. Compare the input data to every position on the map
    /// 2. Return the weights of the position that most closely matches the input
    /// 
    /// For example, if you input data about a science fiction book:
    /// - The SOM would check which position on the map is most similar to sci-fi books
    /// - It would return the characteristics of that position
    /// 
    /// This is how you can use a trained SOM to categorize new data or find similarities
    /// between items. Similar inputs will be mapped to the same or nearby positions.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        int bmu = FindBestMatchingUnit(input);
        return _weights.GetRow(bmu);
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
    /// - The calculation would be: √[(300-400)² + (2010-2020)²] = √(10,100 + 100) = √10,200 ≈ 101
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
    /// Trains the SOM on the provided input data.
    /// </summary>
    /// <param name="input">The input vector to train on.</param>
    /// <param name="epochs">The number of training epochs.</param>
    /// <param name="initialLearningRate">The initial learning rate.</param>
    /// <param name="initialRadius">The initial radius of the neighborhood function.</param>
    /// <remarks>
    /// <para>
    /// This method trains the SOM using the provided input vector. For each epoch, it finds the Best Matching Unit (BMU),
    /// calculates the current learning rate and neighborhood radius, and updates the weights of neurons within the radius
    /// of the BMU. The learning rate and radius decrease over time, allowing the SOM to first organize broadly and then
    /// fine-tune the representation. This implementation uses batch training with a single input vector, but in practice,
    /// SOMs are often trained with multiple input vectors.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the SOM to organize data based on similarities.
    /// 
    /// The training process:
    /// 1. Find which position on the map (BMU) best matches the input
    /// 2. Update that position to better match the input
    /// 3. Also update nearby positions, but less strongly (based on radius)
    /// 4. Repeat for multiple epochs, gradually reducing learning rate and radius
    /// 
    /// Parameters:
    /// - initialLearningRate: How quickly positions adapt (starts high, decreases over time)
    /// - initialRadius: How far the influence spreads (starts large, shrinks over time)
    /// - epochs: How many times to repeat the process
    /// 
    /// Over time, this creates a map where:
    /// - Similar inputs activate the same or nearby positions
    /// - Different regions specialize in different types of input
    /// - The overall structure preserves relationships in the original data
    /// </para>
    /// </remarks>
    public void Train(Vector<T> input, int epochs, T initialLearningRate, T initialRadius)
    {
        T timeConstant = NumOps.FromDouble(epochs / Math.Log(Convert.ToDouble(initialRadius)));

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int bmu = FindBestMatchingUnit(input);
            T learningRate = CalculateLearningRate(initialLearningRate, epoch, epochs);
            T radius = CalculateRadius(initialRadius, epoch, timeConstant);

            UpdateWeights(input, bmu, learningRate, radius);
        }
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
    /// The formula (learningRate × influence × (input - weight)) moves each weight
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
    /// Updates the parameters of the SOM. This method is not typically used in SOMs and throws a NotImplementedException.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <exception cref="NotImplementedException">Always thrown as this method is not implemented for SOMs.</exception>
    /// <remarks>
    /// <para>
    /// SOMs typically use specialized training algorithms rather than the generic parameter update approach
    /// used by other neural networks. This method throws a NotImplementedException to indicate that SOMs
    /// should be trained using the Train method instead.
    /// </para>
    /// <para><b>For Beginners:</b> This method is not used in SOMs because they train differently.
    /// 
    /// While standard neural networks use backpropagation to update parameters:
    /// - SOMs use a competitive learning approach
    /// - They update based on neighborhood and distance
    /// - They directly adjust weights based on similarity to input
    /// 
    /// Instead of using this method, you should use the Train method to train a SOM.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // This method is not typically used in SOMs
        throw new NotImplementedException("UpdateParameters is not implemented for Self-Organizing Maps.");
    }

    /// <summary>
    /// Saves the state of the Self-Organizing Map to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save the state to.</param>
    /// <exception cref="ArgumentNullException">Thrown if the writer is null.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the entire state of the SOM, including the map dimensions, input dimension,
    /// and all neuron weights. It writes these values to the provided binary writer, allowing the SOM
    /// to be saved to a file or other storage medium and later restored.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the entire state of the SOM to a file.
    /// 
    /// When serializing:
    /// - The map dimensions (width and height) are saved
    /// - The input dimension is saved
    /// - All weight values for all positions on the map are saved
    /// 
    /// This is useful for:
    /// - Saving a trained SOM to use later
    /// - Sharing a trained SOM with others
    /// - Creating backups during long training processes
    /// 
    /// Think of it like taking a complete snapshot of the SOM that can be restored later.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(_mapWidth);
        writer.Write(_mapHeight);
        writer.Write(_inputDimension);

        for (int i = 0; i < _mapWidth * _mapHeight; i++)
        {
            for (int j = 0; j < _inputDimension; j++)
            {
                writer.Write(Convert.ToDouble(_weights[i, j]));
            }
        }
    }

    /// <summary>
    /// Loads the state of the Self-Organizing Map from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load the state from.</param>
    /// <exception cref="ArgumentNullException">Thrown if the reader is null.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the state of the SOM from a binary reader. It reads the map dimensions,
    /// input dimension, and all neuron weights, and reconstructs the SOM from these values. This allows
    /// a previously saved SOM to be restored and used for prediction or further training.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved SOM state from a file.
    /// 
    /// When deserializing:
    /// - The map dimensions (width and height) are read first
    /// - The input dimension is read
    /// - A new weights matrix is created with these dimensions
    /// - All weight values are read and restored to their saved values
    /// 
    /// This allows you to:
    /// - Load a previously trained SOM
    /// - Continue using or training a SOM from where you left off
    /// - Use SOMs created by others
    /// 
    /// Think of it like restoring a complete snapshot of a SOM that was saved earlier.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        _mapWidth = reader.ReadInt32();
        _mapHeight = reader.ReadInt32();
        _inputDimension = reader.ReadInt32();

        _weights = new Matrix<T>(_mapWidth * _mapHeight, _inputDimension);

        for (int i = 0; i < _mapWidth * _mapHeight; i++)
        {
            for (int j = 0; j < _inputDimension; j++)
            {
                _weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }
}