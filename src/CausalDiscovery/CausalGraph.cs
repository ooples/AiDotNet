using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Represents a causal Directed Acyclic Graph (DAG) discovered from observational data.
/// This is both a standalone queryable model and an analysis result.
/// </summary>
/// <remarks>
/// <para>
/// A causal graph encodes the causal relationships between variables as a weighted adjacency matrix.
/// An edge from variable i to variable j with weight w means "variable i directly causes variable j
/// with strength w." The graph is guaranteed to be a DAG (no cycles), meaning you cannot follow
/// directed edges and return to the starting variable.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a map of cause-and-effect relationships. Each variable
/// is a node, and arrows between nodes show which variables directly cause changes in others.
/// The weight of each arrow tells you how strong the causal effect is.
///
/// You can query this graph to answer questions like:
/// <list type="bullet">
/// <item>"What causes variable X?" — use <see cref="GetParents(int)"/></item>
/// <item>"What does variable X affect?" — use <see cref="GetChildren(int)"/></item>
/// <item>"What is the Markov blanket of X?" — use <see cref="GetMarkovBlanket(int)"/></item>
/// <item>"What order should I process variables?" — use <see cref="TopologicalSort"/></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for edge weights (e.g., float, double).</typeparam>
public class CausalGraph<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the weighted adjacency matrix where entry [i,j] represents the causal effect from variable i to variable j.
    /// </summary>
    /// <remarks>
    /// <para>A non-zero entry at [i,j] means there is a directed edge from variable i to variable j.</para>
    /// </remarks>
    public Matrix<T> AdjacencyMatrix { get; }

    /// <summary>
    /// Gets the number of variables in the graph.
    /// </summary>
    public int NumVariables { get; }

    /// <summary>
    /// Gets the feature/variable names.
    /// </summary>
    public string[] FeatureNames { get; }

    /// <summary>
    /// Initializes a new CausalGraph from a weighted adjacency matrix.
    /// </summary>
    /// <param name="adjacencyMatrix">Weighted adjacency matrix [d x d]. Entry [i,j] = weight of edge i → j.</param>
    /// <param name="featureNames">Variable names. Must have length equal to the matrix dimension.</param>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square or names don't match dimensions.</exception>
    public CausalGraph(Matrix<T> adjacencyMatrix, string[] featureNames)
    {
        Guard.NotNull(adjacencyMatrix);
        Guard.NotNull(featureNames);

        if (adjacencyMatrix.Rows != adjacencyMatrix.Columns)
        {
            throw new ArgumentException("Adjacency matrix must be square.", nameof(adjacencyMatrix));
        }

        if (featureNames.Length != adjacencyMatrix.Rows)
        {
            throw new ArgumentException(
                $"Feature names length ({featureNames.Length}) must match matrix dimension ({adjacencyMatrix.Rows}).",
                nameof(featureNames));
        }

        if (featureNames.Length != new HashSet<string>(featureNames).Count)
        {
            throw new ArgumentException("Feature names must be unique.", nameof(featureNames));
        }

        AdjacencyMatrix = adjacencyMatrix;
        NumVariables = adjacencyMatrix.Rows;
        FeatureNames = featureNames;
    }

    /// <summary>
    /// Gets the indices of parent variables (direct causes) of the specified variable.
    /// </summary>
    /// <param name="variableIndex">The index of the variable to query.</param>
    /// <returns>Array of parent variable indices.</returns>
    public int[] GetParents(int variableIndex)
    {
        ValidateIndex(variableIndex);
        return NonZeroIndices(Engine.GetColumn(AdjacencyMatrix, variableIndex))
            .Where(i => i != variableIndex).ToArray();
    }

    /// <summary>
    /// Gets the parent variables (direct causes) of a named variable.
    /// </summary>
    /// <param name="variableName">The name of the variable to query.</param>
    /// <returns>Array of parent variable names.</returns>
    public string[] GetParents(string variableName)
    {
        int idx = GetIndex(variableName);
        return GetParents(idx).Select(i => FeatureNames[i]).ToArray();
    }

    /// <summary>
    /// Gets the indices of child variables (direct effects) of the specified variable.
    /// </summary>
    /// <param name="variableIndex">The index of the variable to query.</param>
    /// <returns>Array of child variable indices.</returns>
    public int[] GetChildren(int variableIndex)
    {
        ValidateIndex(variableIndex);
        return NonZeroIndices(Engine.GetRow(AdjacencyMatrix, variableIndex))
            .Where(j => j != variableIndex).ToArray();
    }

    /// <summary>
    /// Gets the child variables (direct effects) of a named variable.
    /// </summary>
    public string[] GetChildren(string variableName)
    {
        int idx = GetIndex(variableName);
        return GetChildren(idx).Select(i => FeatureNames[i]).ToArray();
    }

    /// <summary>
    /// Gets all ancestor variables (direct and indirect causes) of the specified variable.
    /// </summary>
    /// <param name="variableIndex">The index of the variable to query.</param>
    /// <returns>Array of ancestor variable indices.</returns>
    public int[] GetAncestors(int variableIndex)
    {
        ValidateIndex(variableIndex);
        var ancestors = new HashSet<int>();
        var queue = new Queue<int>(GetParents(variableIndex));

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (ancestors.Add(current))
            {
                foreach (int parent in GetParents(current))
                {
                    queue.Enqueue(parent);
                }
            }
        }

        return [.. ancestors];
    }

    /// <summary>
    /// Gets all descendant variables (direct and indirect effects) of the specified variable.
    /// </summary>
    /// <param name="variableIndex">The index of the variable to query.</param>
    /// <returns>Array of descendant variable indices.</returns>
    public int[] GetDescendants(int variableIndex)
    {
        ValidateIndex(variableIndex);
        var descendants = new HashSet<int>();
        var queue = new Queue<int>(GetChildren(variableIndex));

        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (descendants.Add(current))
            {
                foreach (int child in GetChildren(current))
                {
                    queue.Enqueue(child);
                }
            }
        }

        return [.. descendants];
    }

    /// <summary>
    /// Gets the Markov blanket of a variable: its parents, children, and co-parents of its children.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Markov blanket is the minimal set of variables that makes a
    /// variable conditionally independent of all others. Knowing the Markov blanket values,
    /// you have all the information needed to predict the variable — no other variables help.
    /// </para>
    /// </remarks>
    /// <param name="variableIndex">The index of the variable to query.</param>
    /// <returns>Array of Markov blanket variable indices.</returns>
    public int[] GetMarkovBlanket(int variableIndex)
    {
        ValidateIndex(variableIndex);
        var blanket = new HashSet<int>();

        // Parents
        foreach (int parent in GetParents(variableIndex))
        {
            blanket.Add(parent);
        }

        // Children and co-parents of children
        foreach (int child in GetChildren(variableIndex))
        {
            blanket.Add(child);
            foreach (int coParent in GetParents(child))
            {
                if (coParent != variableIndex)
                {
                    blanket.Add(coParent);
                }
            }
        }

        return [.. blanket];
    }

    /// <summary>
    /// Gets the weight of the edge from one variable to another.
    /// </summary>
    /// <param name="fromIndex">Source variable index.</param>
    /// <param name="toIndex">Target variable index.</param>
    /// <returns>The edge weight, or zero if no edge exists.</returns>
    public T GetEdgeWeight(int fromIndex, int toIndex)
    {
        ValidateIndex(fromIndex);
        ValidateIndex(toIndex);
        return AdjacencyMatrix[fromIndex, toIndex];
    }

    /// <summary>
    /// Gets the weight of the edge between named variables.
    /// </summary>
    public T GetEdgeWeight(string fromName, string toName)
    {
        return GetEdgeWeight(GetIndex(fromName), GetIndex(toName));
    }

    /// <summary>
    /// Returns whether there is a directed edge from one variable to another.
    /// </summary>
    public bool HasEdge(int fromIndex, int toIndex)
    {
        ValidateIndex(fromIndex);
        ValidateIndex(toIndex);
        return !IsZero(AdjacencyMatrix[fromIndex, toIndex]);
    }

    /// <summary>
    /// Returns whether the graph is a valid DAG (no directed cycles).
    /// </summary>
    /// <remarks>
    /// <para>Uses Kahn's algorithm (topological sort) to check for cycles.</para>
    /// </remarks>
    public bool IsDAG()
    {
        int[] inDegree = ComputeInDegrees();

        var queue = new Queue<int>();
        for (int i = 0; i < NumVariables; i++)
        {
            if (inDegree[i] == 0)
            {
                queue.Enqueue(i);
            }
        }

        int visited = 0;
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            visited++;
            foreach (int j in NonZeroIndices(Engine.GetRow(AdjacencyMatrix, node)))
            {
                inDegree[j]--;
                if (inDegree[j] == 0)
                {
                    queue.Enqueue(j);
                }
            }
        }

        return visited == NumVariables;
    }

    /// <summary>
    /// Returns a topological ordering of the variables (causes before effects).
    /// </summary>
    /// <returns>Array of variable indices in topological order.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the graph contains a cycle.</exception>
    public int[] TopologicalSort()
    {
        int[] inDegree = ComputeInDegrees();

        var queue = new Queue<int>();
        for (int i = 0; i < NumVariables; i++)
        {
            if (inDegree[i] == 0)
            {
                queue.Enqueue(i);
            }
        }

        var order = new List<int>();
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            order.Add(node);
            foreach (int j in NonZeroIndices(Engine.GetRow(AdjacencyMatrix, node)))
            {
                inDegree[j]--;
                if (inDegree[j] == 0)
                {
                    queue.Enqueue(j);
                }
            }
        }

        if (order.Count != NumVariables)
        {
            throw new InvalidOperationException("Graph contains a cycle and cannot be topologically sorted.");
        }

        return [.. order];
    }

    /// <summary>
    /// Gets all edges in the graph as (from, to, weight) tuples, optionally filtered by minimum weight.
    /// </summary>
    /// <param name="minAbsWeight">Minimum absolute weight to include an edge. Default: 0 (all non-zero edges).</param>
    /// <returns>List of (from, to, weight) tuples.</returns>
    public List<(int From, int To, T Weight)> GetEdges(double minAbsWeight = 0.0)
    {
        var edges = new List<(int From, int To, T Weight)>();
        var flat = Tensor<T>.FromMatrix(AdjacencyMatrix).Reshape(new[] { NumVariables * NumVariables });
        var kept = Engine.TensorGreaterThan(Engine.TensorAbs(flat), _numOps.FromDouble(minAbsWeight));
        int[] positions = Engine.TensorNonzero(kept).ToArray();
        if (positions.Length == 0)
            return edges;

        T[] weights = Engine.TensorIndexSelect(flat, new Tensor<int>(positions, new[] { positions.Length }), 0).ToArray();
        for (int e = 0; e < positions.Length; e++)
        {
            edges.Add((positions[e] / NumVariables, positions[e] % NumVariables, weights[e]));
        }

        return edges;
    }

    /// <summary>
    /// Gets all edges as named tuples for human-readable output.
    /// </summary>
    public List<(string From, string To, T Weight)> GetNamedEdges(double minAbsWeight = 0.0)
    {
        return GetEdges(minAbsWeight)
            .Select(e => (FeatureNames[e.From], FeatureNames[e.To], e.Weight))
            .ToList();
    }

    /// <summary>
    /// Computes a simple node importance score based on out-degree weighted by edge strength.
    /// </summary>
    /// <returns>Dictionary mapping variable index to importance score.</returns>
    public Dictionary<int, double> GetNodeImportance()
    {
        var importance = new Dictionary<int, double>();
        var scores = Engine.ReduceSum(Engine.TensorAbs(Tensor<T>.FromMatrix(AdjacencyMatrix)), new[] { 1 }, false)
            .Cast<double>().ToArray();
        for (int i = 0; i < NumVariables; i++)
        {
            importance[i] = scores[i];
        }

        return importance;
    }

    /// <summary>
    /// Gets the total number of edges in the graph.
    /// </summary>
    public int EdgeCount
    {
        get
        {
            return Engine.TensorCountNonzero(Tensor<T>.FromMatrix(AdjacencyMatrix));
        }
    }

    /// <summary>
    /// Gets the density of the graph (fraction of possible edges that exist).
    /// </summary>
    public double Density => NumVariables > 1
        ? (double)EdgeCount / (NumVariables * (NumVariables - 1))
        : 0.0;

    /// <summary>
    /// Computes the interventional distribution P(target | do(intervention = value)) using the
    /// truncated factorization formula from Pearl's do-calculus.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The do-calculus intervention do(X = x) works by:
    /// <list type="number">
    /// <item>Removing all incoming edges to the intervention variable (breaking its causes)</item>
    /// <item>Setting the intervention variable to the specified value</item>
    /// <item>Propagating the effect forward through the causal graph using the learned relationships</item>
    /// </list>
    /// </para>
    /// <para>
    /// For a linear causal model X = W^T X + ε, the truncated factorization produces:
    /// For each sample, set X_intervention = value, then simulate downstream effects using the
    /// topological order of the graph, computing each variable from its parents' values.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This answers "what would happen to Y if we FORCE X to be some value?"
    /// It's like running a simulated experiment using the discovered causal relationships.
    /// </para>
    /// </remarks>
    /// <param name="interventionVariable">Index of the variable being intervened on.</param>
    /// <param name="interventionValue">The value to set the intervention variable to.</param>
    /// <param name="targetVariable">Index of the target variable whose distribution we want.</param>
    /// <param name="observationalData">The original observational data [n_samples x n_variables].</param>
    /// <returns>An <see cref="InterventionalDistribution{T}"/> representing P(target | do(intervention = value)).</returns>
    /// <exception cref="ArgumentException">If variable indices are out of range or the same variable is used for both.</exception>
    public InterventionalDistribution<T> ComputeInterventionalDistribution(
        int interventionVariable, T interventionValue,
        int targetVariable, Matrix<T> observationalData)
    {
        ValidateIndex(interventionVariable);
        ValidateIndex(targetVariable);

        if (observationalData.Columns != NumVariables)
        {
            throw new ArgumentException(
                $"Observational data columns ({observationalData.Columns}) must match graph variables ({NumVariables}).",
                nameof(observationalData));
        }

        int n = observationalData.Rows;
        int d = NumVariables;

        // Get topological order for forward propagation
        int[] topoOrder;
        try
        {
            topoOrder = TopologicalSort();
        }
        catch (InvalidOperationException)
        {
            // Graph has cycles — fall back to simple linear regression estimate
            return ComputeInterventionalFallback(
                interventionVariable, interventionValue, targetVariable, observationalData);
        }

        // Compute observational mean of target for ACE calculation
        var data = Tensor<T>.FromMatrix(observationalData);
        T obsTargetMean = Engine.TensorMean(Engine.TensorSliceAxis(data, 1, targetVariable));

        // Edge weights whose magnitude exceeds 1e-10; the diagonal is excluded when computing noise.
        var weights = Tensor<T>.FromMatrix(AdjacencyMatrix);
        var significant = Engine.TensorGreaterThan(Engine.TensorAbs(weights), _numOps.FromDouble(1e-10));
        var activeWeights = Engine.TensorWhere(significant, weights, Engine.TensorZerosLike(weights));
        var offDiagonal = Engine.TensorSubtract(Engine.TensorOnesLike(weights), Engine.TensorEye<T>(d));
        var noiseWeights = Engine.TensorMultiply(activeWeights, offDiagonal);

        // Pre-compute residuals (noise) for each variable using observational data:
        // ε_j = X_j - Σ_{i≠j} W[i,j] * X_i for every sample at once.
        var residuals = Engine.TensorSubtract(data, Engine.TensorMatMul(data, noiseWeights)).ToMatrix();

        // Simulate the intervention for all samples together: start from the observational values,
        // force the intervention variable, then propagate forward in topological order.
        var simulated = observationalData.Clone();
        var forced = Engine.TensorFullLike(Engine.TensorSliceAxis(data, 1, interventionVariable), interventionValue)
            .ToVector();
        Engine.SetColumn(simulated, interventionVariable, forced);

        // Variables downstream of the intervention are recomputed from the structural equations;
        // upstream or unaffected variables keep their observational values.
        var affectedByIntervention = new HashSet<int>(GetDescendants(interventionVariable));
        affectedByIntervention.Add(interventionVariable);
        var activeWeightMatrix = activeWeights.ToMatrix();

        foreach (int j in topoOrder)
        {
            if (j == interventionVariable)
            {
                // Intervention variable: forced to intervention value
                Engine.SetColumn(simulated, j, forced);
                continue;
            }

            if (!affectedByIntervention.Contains(j))
            {
                // Not affected by intervention: keep observational value
                continue;
            }

            // Affected downstream variable: noise + Σ_i W[i,j] * simulated_i
            var parentsEffect = Engine.MatrixVectorMultiply(simulated, Engine.GetColumn(activeWeightMatrix, j));
            Engine.SetColumn(simulated, j, Engine.Add(Engine.GetColumn(residuals, j), parentsEffect));
        }

        var interventionalSamples = Engine.GetColumn(simulated, targetVariable).AsTensor().Cast<double>().ToArray();

        return new InterventionalDistribution<T>(
            interventionVariableIndex: interventionVariable,
            interventionVariableName: FeatureNames[interventionVariable],
            interventionValue: interventionValue,
            targetVariableIndex: targetVariable,
            targetVariableName: FeatureNames[targetVariable],
            samples: interventionalSamples,
            observationalMean: _numOps.ToDouble(obsTargetMean));
    }

    /// <summary>
    /// Computes the interventional distribution using named variables.
    /// </summary>
    /// <param name="interventionVariableName">Name of the intervention variable.</param>
    /// <param name="interventionValue">The value to set via do-operator.</param>
    /// <param name="targetVariableName">Name of the target variable.</param>
    /// <param name="observationalData">The original observational data.</param>
    /// <returns>An <see cref="InterventionalDistribution{T}"/> representing P(target | do(intervention = value)).</returns>
    public InterventionalDistribution<T> ComputeInterventionalDistribution(
        string interventionVariableName, T interventionValue,
        string targetVariableName, Matrix<T> observationalData)
    {
        return ComputeInterventionalDistribution(
            GetIndex(interventionVariableName), interventionValue,
            GetIndex(targetVariableName), observationalData);
    }

    /// <summary>
    /// Fallback for computing interventional distribution when the graph is not a DAG.
    /// Uses simple linear regression of target on intervention variable.
    /// </summary>
    private InterventionalDistribution<T> ComputeInterventionalFallback(
        int interventionVariable, T interventionValue,
        int targetVariable, Matrix<T> observationalData)
    {
        var data = Tensor<T>.FromMatrix(observationalData);
        var x = Engine.TensorSliceAxis(data, 1, interventionVariable);
        var y = Engine.TensorSliceAxis(data, 1, targetVariable);

        // Simple OLS: target = a + b * intervention + noise, with the slope from the centered
        // moments sum(dx dy) / (sum(dx^2) + 1e-10). Centering first avoids the cancellation of
        // sum(x^2) - n mean(x)^2 when the intervention variable has a large mean.
        T mx = Engine.TensorMean(x), my = Engine.TensorMean(y);
        var dx = Engine.TensorSubtractScalar(x, mx);
        var dy = Engine.TensorSubtractScalar(y, my);
        T b = _numOps.Divide(
            Engine.TensorSum(Engine.TensorMultiply(dx, dy)),
            _numOps.Add(Engine.TensorSum(Engine.TensorSquare(dx)), _numOps.FromDouble(1e-10)));
        T a = _numOps.Subtract(my, _numOps.Multiply(b, mx));

        // Generate interventional samples using the regression residuals:
        // sample = a + b * value + (y - (a + b * x))
        var yPred = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(x, b), a);
        var noise = Engine.TensorSubtract(y, yPred);
        T intervened = _numOps.Add(a, _numOps.Multiply(b, interventionValue));
        var samples = Engine.TensorAddScalar(noise, intervened).Cast<double>().ToArray();

        return new InterventionalDistribution<T>(
            interventionVariableIndex: interventionVariable,
            interventionVariableName: FeatureNames[interventionVariable],
            interventionValue: interventionValue,
            targetVariableIndex: targetVariable,
            targetVariableName: FeatureNames[targetVariable],
            samples: samples,
            observationalMean: _numOps.ToDouble(my));
    }

    private bool IsZero(T value)
    {
        return _numOps.Equals(value, _numOps.Zero);
    }

    /// <summary>Indices of the non-zero entries of a vector, in ascending order.</summary>
    private static int[] NonZeroIndices(Vector<T> values)
        => Engine.TensorNonzero(values.AsTensor()).ToArray();

    /// <summary>Number of incoming edges (non-zero entries) in each column of the adjacency matrix.</summary>
    private int[] ComputeInDegrees()
    {
        var hasEdge = Engine.TensorNotEquals(Tensor<T>.FromMatrix(AdjacencyMatrix), _numOps.Zero);
        return Engine.ReduceSum(hasEdge, new[] { 0 }, false).Cast<int>().ToArray();
    }

    private int GetIndex(string variableName)
    {
        int idx = Array.IndexOf(FeatureNames, variableName);
        if (idx < 0)
        {
            throw new ArgumentException($"Variable '{variableName}' not found in feature names.", nameof(variableName));
        }

        return idx;
    }

    private void ValidateIndex(int index)
    {
        if (index < 0 || index >= NumVariables)
        {
            throw new ArgumentOutOfRangeException(nameof(index),
                $"Variable index {index} is out of range [0, {NumVariables - 1}].");
        }
    }
}
