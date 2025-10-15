using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ReinforcementLearning.ReplayBuffers
{
    /// <summary>
    /// A binary tree structure for efficient sampling of experiences based on priority.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// The sum tree is a binary tree where each leaf node contains a priority value,
    /// and each internal node contains the sum of its children's values. This allows
    /// for efficient O(log n) sampling of experiences based on their priorities.
    /// </remarks>
    public class SumTree<T>
    {
        private readonly T[] _tree;
        private readonly int _capacity;
        private readonly INumericOperations<T> _numOps = default!;

        /// <summary>
        /// Gets the total sum of all priorities in the tree.
        /// </summary>
        public T Total => _tree[0]; // Root node contains the sum of all priorities

        /// <summary>
        /// Initializes a new instance of the <see cref="SumTree{T}"/> class.
        /// </summary>
        /// <param name="capacity">The number of leaf nodes in the tree (max number of stored experiences).</param>
        /// <param name="numOps">The numeric operations for type T.</param>
        public SumTree(int capacity, INumericOperations<T> numOps)
        {
            _capacity = capacity;
            _numOps = numOps;
            
            // Size of the tree is 2 * capacity - 1
            // (capacity leaf nodes + capacity - 1 internal nodes)
            _tree = new T[2 * capacity - 1];
            Clear();
        }

        /// <summary>
        /// Updates the priority at a specific leaf node index.
        /// </summary>
        /// <param name="dataIndex">The index of the experience in the buffer.</param>
        /// <param name="priority">The new priority value.</param>
        public void Update(int dataIndex, T priority)
        {
            // Convert data index to tree index
            int treeIndex = dataIndex + _capacity - 1;
            
            // Update the leaf node
            T change = _numOps.Subtract(priority, _tree[treeIndex]);
            _tree[treeIndex] = priority;
            
            // Propagate changes upward through the tree
            PropagateChanges(treeIndex, change);
        }

        /// <summary>
        /// Propagates priority changes up the tree.
        /// </summary>
        /// <param name="treeIndex">The index of the node in the tree.</param>
        /// <param name="change">The change in priority value.</param>
        private void PropagateChanges(int treeIndex, T change)
        {
            int parent = (treeIndex - 1) / 2;

            while (parent >= 0)
            {
                _tree[parent] = _numOps.Add(_tree[parent], change);
                parent = (parent - 1) / 2;
            }
        }

        /// <summary>
        /// Retrieves the leaf node index and priority value for a given cumulative value.
        /// </summary>
        /// <param name="value">The cumulative value to search for.</param>
        /// <returns>A tuple containing the data index and priority value.</returns>
        public (int, T) Get(T value)
        {
            // Start at the root
            int treeIndex = 0;
            
            // Traverse the tree to find the leaf node
            while (treeIndex < _capacity - 1)
            {
                int leftChildIndex = 2 * treeIndex + 1;
                int rightChildIndex = leftChildIndex + 1;
                
                if (rightChildIndex >= _tree.Length)
                {
                    treeIndex = leftChildIndex;
                    continue;
                }
                
                // Go left or right depending on the value
                if (_numOps.LessThanOrEquals(value, _tree[leftChildIndex]))
                {
                    treeIndex = leftChildIndex;
                }
                else
                {
                    value = _numOps.Subtract(value, _tree[leftChildIndex]);
                    treeIndex = rightChildIndex;
                }
            }
            
            // Convert tree index to data index
            int dataIndex = treeIndex - (_capacity - 1);
            
            return (dataIndex, _tree[treeIndex]);
        }

        /// <summary>
        /// Clears the sum tree by setting all values to zero.
        /// </summary>
        public void Clear()
        {
            for (int i = 0; i < _tree.Length; i++)
            {
                _tree[i] = _numOps.Zero;
            }
        }
    }
}