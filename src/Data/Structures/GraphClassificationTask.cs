using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Structures;

/// <summary>
/// Represents a graph classification task where the goal is to classify entire graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph classification assigns a label to an entire graph based on its structure and node/edge features.
/// Unlike node classification (classify individual nodes) or link prediction (predict edges),
/// graph classification treats the whole graph as a single data point.
/// </para>
/// <para><b>For Beginners:</b> Graph classification is like determining the category of a complex object.
///
/// **Real-world examples:**
///
/// **Molecular Property Prediction:**
/// - Input: Molecular graph (atoms as nodes, bonds as edges)
/// - Task: Predict molecular properties
/// - Examples:
///   * Is this molecule toxic?
///   * What is the solubility?
///   * Will this be a good drug candidate?
/// - Dataset: ZINC, QM9, BACE
///
/// **Protein Function Prediction:**
/// - Input: Protein structure graph
/// - Task: Predict protein function or family
/// - How: Analyze amino acid sequences and 3D structure
///
/// **Chemical Reaction Prediction:**
/// - Input: Reaction graph showing reactants and products
/// - Task: Predict reaction type or outcome
///
/// **Social Network Analysis:**
/// - Input: Community subgraphs
/// - Task: Classify community type or behavior
/// - Example: Identify bot networks vs organic communities
///
/// **Code Analysis:**
/// - Input: Abstract syntax tree (AST) or control flow graph
/// - Task: Detect bugs, classify code functionality
/// - Example: "Is this code snippet vulnerable to SQL injection?"
///
/// **Key Challenge:** Graph-level representation
/// - Must aggregate information from all nodes and edges
/// - Common approaches: Global pooling, hierarchical pooling, set2set
/// </para>
/// </remarks>
public class GraphClassificationTask<T>
{
    /// <summary>
    /// List of training graphs.
    /// </summary>
    /// <remarks>
    /// Each graph in the list is an independent sample with its own structure and features.
    /// </remarks>
    public List<GraphData<T>> TrainGraphs { get; set; } = new List<GraphData<T>>();

    /// <summary>
    /// List of validation graphs.
    /// </summary>
    public List<GraphData<T>> ValGraphs { get; set; } = new List<GraphData<T>>();

    /// <summary>
    /// List of test graphs.
    /// </summary>
    public List<GraphData<T>> TestGraphs { get; set; } = new List<GraphData<T>>();

    /// <summary>
    /// Labels for training graphs.
    /// Shape: [num_train_graphs] or [num_train_graphs, num_classes] for multi-label.
    /// </summary>
    public Tensor<T> TrainLabels { get; set; } = new Tensor<T>([0]);

    /// <summary>
    /// Labels for validation graphs.
    /// Shape: [num_val_graphs] or [num_val_graphs, num_classes].
    /// </summary>
    public Tensor<T> ValLabels { get; set; } = new Tensor<T>([0]);

    /// <summary>
    /// Labels for test graphs.
    /// Shape: [num_test_graphs] or [num_test_graphs, num_classes].
    /// </summary>
    public Tensor<T> TestLabels { get; set; } = new Tensor<T>([0]);

    /// <summary>
    /// Number of classes in the classification task.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Whether this is a multi-label classification task.
    /// </summary>
    /// <remarks>
    /// - False: Each graph has exactly one label (e.g., molecule is toxic or not)
    /// - True: Each graph can have multiple labels (e.g., molecule has multiple properties)
    /// </remarks>
    public bool IsMultiLabel { get; set; } = false;

    /// <summary>
    /// Whether this is a regression task instead of classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For regression tasks (e.g., predicting molecular energy), labels are continuous values
    /// rather than discrete classes.
    /// </para>
    /// <para><b>For Beginners:</b> The difference between classification and regression:
    /// - **Classification**: Predict categories (e.g., "toxic" vs "non-toxic")
    /// - **Regression**: Predict continuous values (e.g., "solubility = 2.3 mg/L")
    ///
    /// Examples:
    /// - Classification: Is this molecule a good drug? (Yes/No)
    /// - Regression: What is this molecule's binding affinity? (0.0 to 10.0)
    /// </para>
    /// </remarks>
    public bool IsRegression { get; set; } = false;

    /// <summary>
    /// Average number of nodes per graph (for informational purposes).
    /// </summary>
    public double AvgNumNodes { get; set; }

    /// <summary>
    /// Average number of edges per graph (for informational purposes).
    /// </summary>
    public double AvgNumEdges { get; set; }
}
