// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

// Polyfill for PriorityQueue<TElement, TPriority> to support .NET Framework 4.6.2 and 4.7.1
// PriorityQueue was introduced in .NET 6.0

#if !NET6_0_OR_GREATER

using System.Collections.Generic;

namespace System.Collections.Generic
{
    /// <summary>
    /// Represents a min priority queue.
    /// </summary>
    /// <typeparam name="TElement">Specifies the type of elements in the queue.</typeparam>
    /// <typeparam name="TPriority">Specifies the type of priority associated with enqueued elements.</typeparam>
    /// <remarks>
    /// This is a polyfill implementation for .NET Framework compatibility.
    /// It uses a binary heap internally for O(log n) enqueue and dequeue operations.
    /// </remarks>
    public class PriorityQueue<TElement, TPriority>
    {
        private readonly List<(TElement Element, TPriority Priority)> _heap;
        private readonly IComparer<TPriority> _comparer;

        /// <summary>
        /// Initializes a new instance of the PriorityQueue class.
        /// </summary>
        public PriorityQueue()
            : this(Comparer<TPriority>.Default)
        {
        }

        /// <summary>
        /// Initializes a new instance of the PriorityQueue class with a specified comparer.
        /// </summary>
        /// <param name="comparer">The comparer to use for priority comparisons.</param>
        public PriorityQueue(IComparer<TPriority> comparer)
        {
            _heap = new List<(TElement, TPriority)>();
            _comparer = comparer ?? Comparer<TPriority>.Default;
        }

        /// <summary>
        /// Initializes a new instance of the PriorityQueue class with a specified initial capacity.
        /// </summary>
        /// <param name="initialCapacity">The initial capacity of the queue.</param>
        public PriorityQueue(int initialCapacity)
            : this(initialCapacity, Comparer<TPriority>.Default)
        {
        }

        /// <summary>
        /// Initializes a new instance of the PriorityQueue class with a specified initial capacity and comparer.
        /// </summary>
        /// <param name="initialCapacity">The initial capacity of the queue.</param>
        /// <param name="comparer">The comparer to use for priority comparisons.</param>
        public PriorityQueue(int initialCapacity, IComparer<TPriority> comparer)
        {
            _heap = new List<(TElement, TPriority)>(initialCapacity);
            _comparer = comparer ?? Comparer<TPriority>.Default;
        }

        /// <summary>
        /// Gets the number of elements in the priority queue.
        /// </summary>
        public int Count => _heap.Count;

        /// <summary>
        /// Adds the specified element with associated priority to the queue.
        /// </summary>
        /// <param name="element">The element to add.</param>
        /// <param name="priority">The priority associated with the element.</param>
        public void Enqueue(TElement element, TPriority priority)
        {
            _heap.Add((element, priority));
            HeapifyUp(_heap.Count - 1);
        }

        /// <summary>
        /// Removes and returns the element with the minimum priority.
        /// </summary>
        /// <returns>The element with the minimum priority.</returns>
        /// <exception cref="InvalidOperationException">The queue is empty.</exception>
        public TElement Dequeue()
        {
            if (_heap.Count == 0)
                throw new InvalidOperationException("The priority queue is empty.");

            var result = _heap[0].Element;
            int lastIndex = _heap.Count - 1;
            _heap[0] = _heap[lastIndex];
            _heap.RemoveAt(lastIndex);

            if (_heap.Count > 0)
                HeapifyDown(0);

            return result;
        }

        /// <summary>
        /// Returns the element with the minimum priority without removing it.
        /// </summary>
        /// <returns>The element with the minimum priority.</returns>
        /// <exception cref="InvalidOperationException">The queue is empty.</exception>
        public TElement Peek()
        {
            if (_heap.Count == 0)
                throw new InvalidOperationException("The priority queue is empty.");

            return _heap[0].Element;
        }

        /// <summary>
        /// Attempts to remove and return the element with the minimum priority.
        /// </summary>
        /// <param name="element">When this method returns, contains the element, if the operation succeeded.</param>
        /// <param name="priority">When this method returns, contains the priority, if the operation succeeded.</param>
        /// <returns>true if an element was removed; otherwise, false.</returns>
        public bool TryDequeue(out TElement element, out TPriority priority)
        {
            if (_heap.Count == 0)
            {
                element = default!;
                priority = default!;
                return false;
            }

            element = _heap[0].Element;
            priority = _heap[0].Priority;

            int lastIndex = _heap.Count - 1;
            _heap[0] = _heap[lastIndex];
            _heap.RemoveAt(lastIndex);

            if (_heap.Count > 0)
                HeapifyDown(0);

            return true;
        }

        /// <summary>
        /// Attempts to return the element with the minimum priority without removing it.
        /// </summary>
        /// <param name="element">When this method returns, contains the element, if the operation succeeded.</param>
        /// <param name="priority">When this method returns, contains the priority, if the operation succeeded.</param>
        /// <returns>true if an element exists; otherwise, false.</returns>
        public bool TryPeek(out TElement element, out TPriority priority)
        {
            if (_heap.Count == 0)
            {
                element = default!;
                priority = default!;
                return false;
            }

            element = _heap[0].Element;
            priority = _heap[0].Priority;
            return true;
        }

        /// <summary>
        /// Removes all elements from the priority queue.
        /// </summary>
        public void Clear()
        {
            _heap.Clear();
        }

        /// <summary>
        /// Ensures that the priority queue can hold at least the specified capacity.
        /// </summary>
        /// <param name="capacity">The minimum capacity to ensure.</param>
        public void EnsureCapacity(int capacity)
        {
            if (_heap.Capacity < capacity)
            {
                _heap.Capacity = capacity;
            }
        }

        private void HeapifyUp(int index)
        {
            while (index > 0)
            {
                int parentIndex = (index - 1) / 2;
                if (_comparer.Compare(_heap[index].Priority, _heap[parentIndex].Priority) >= 0)
                    break;

                Swap(index, parentIndex);
                index = parentIndex;
            }
        }

        private void HeapifyDown(int index)
        {
            while (true)
            {
                int smallest = index;
                int leftChild = 2 * index + 1;
                int rightChild = 2 * index + 2;

                if (leftChild < _heap.Count &&
                    _comparer.Compare(_heap[leftChild].Priority, _heap[smallest].Priority) < 0)
                {
                    smallest = leftChild;
                }

                if (rightChild < _heap.Count &&
                    _comparer.Compare(_heap[rightChild].Priority, _heap[smallest].Priority) < 0)
                {
                    smallest = rightChild;
                }

                if (smallest == index)
                    break;

                Swap(index, smallest);
                index = smallest;
            }
        }

        private void Swap(int i, int j)
        {
            var temp = _heap[i];
            _heap[i] = _heap[j];
            _heap[j] = temp;
        }
    }
}

#endif
