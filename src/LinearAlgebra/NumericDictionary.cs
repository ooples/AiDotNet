using System.Collections;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinearAlgebra;

/// <summary>
/// A thread-safe dictionary implementation that uses INumericOperations for key comparison,
/// allowing generic numeric types T to be used as keys without requiring the notnull constraint.
/// </summary>
/// <typeparam name="TKey">The numeric type used for keys.</typeparam>
/// <typeparam name="TValue">The type of values stored in the dictionary.</typeparam>
/// <remarks>
/// <para>
/// This collection solves the problem of using generic numeric types as dictionary keys in a generic context.
/// Standard Dictionary requires TKey to have a notnull constraint, but this library intentionally avoids
/// such constraints to maintain flexibility. NumericDictionary uses INumericOperations.Equals() for
/// key comparison instead of the default equality comparer.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a special dictionary designed for numeric keys.
///
/// Regular dictionaries in C# have trouble with generic number types because they need special
/// guarantees about how keys are compared. This dictionary uses the same math operations
/// (INumericOperations) that the rest of the library uses, so it works seamlessly with any
/// numeric type (float, double, decimal, etc.).
///
/// It's also thread-safe, meaning multiple parts of your code can read from and write to it
/// at the same time without causing errors.
/// </para>
/// </remarks>
public class NumericDictionary<TKey, TValue> : IEnumerable<KeyValuePair<TKey, TValue>>
{
    private readonly List<TKey> _keys;
    private readonly List<TValue> _values;
    private readonly INumericOperations<TKey> _numOps;
    private readonly object _lock = new object();

    /// <summary>
    /// Initializes a new instance of the NumericDictionary class.
    /// </summary>
    public NumericDictionary()
    {
        _keys = new List<TKey>();
        _values = new List<TValue>();
        _numOps = MathHelper.GetNumericOperations<TKey>();
    }

    /// <summary>
    /// Initializes a new instance of the NumericDictionary class with specified initial capacity.
    /// </summary>
    /// <param name="capacity">The initial capacity of the dictionary.</param>
    public NumericDictionary(int capacity)
    {
        _keys = new List<TKey>(capacity);
        _values = new List<TValue>(capacity);
        _numOps = MathHelper.GetNumericOperations<TKey>();
    }

    /// <summary>
    /// Gets the number of key/value pairs in the dictionary.
    /// </summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _keys.Count;
            }
        }
    }

    /// <summary>
    /// Gets a collection containing the keys in the dictionary.
    /// </summary>
    public IReadOnlyList<TKey> Keys
    {
        get
        {
            lock (_lock)
            {
                return _keys.ToList().AsReadOnly();
            }
        }
    }

    /// <summary>
    /// Gets a collection containing the values in the dictionary.
    /// </summary>
    public IReadOnlyList<TValue> Values
    {
        get
        {
            lock (_lock)
            {
                return _values.ToList().AsReadOnly();
            }
        }
    }

    /// <summary>
    /// Gets or sets the value associated with the specified key.
    /// </summary>
    /// <param name="key">The key of the value to get or set.</param>
    /// <returns>The value associated with the specified key.</returns>
    /// <exception cref="KeyNotFoundException">The key was not found in the dictionary.</exception>
    public TValue this[TKey key]
    {
        get
        {
            lock (_lock)
            {
                int index = FindKeyIndex(key);
                if (index < 0)
                {
                    throw new KeyNotFoundException($"The key was not found in the dictionary.");
                }
                return _values[index];
            }
        }
        set
        {
            lock (_lock)
            {
                int index = FindKeyIndex(key);
                if (index >= 0)
                {
                    _values[index] = value;
                }
                else
                {
                    _keys.Add(key);
                    _values.Add(value);
                }
            }
        }
    }

    /// <summary>
    /// Adds the specified key and value to the dictionary.
    /// </summary>
    /// <param name="key">The key of the element to add.</param>
    /// <param name="value">The value of the element to add.</param>
    /// <exception cref="ArgumentException">An element with the same key already exists.</exception>
    public void Add(TKey key, TValue value)
    {
        lock (_lock)
        {
            if (FindKeyIndex(key) >= 0)
            {
                throw new ArgumentException("An element with the same key already exists in the dictionary.");
            }
            _keys.Add(key);
            _values.Add(value);
        }
    }

    /// <summary>
    /// Attempts to add the specified key and value to the dictionary.
    /// </summary>
    /// <param name="key">The key of the element to add.</param>
    /// <param name="value">The value of the element to add.</param>
    /// <returns>true if the key/value pair was added successfully; false if the key already exists.</returns>
    public bool TryAdd(TKey key, TValue value)
    {
        lock (_lock)
        {
            if (FindKeyIndex(key) >= 0)
            {
                return false;
            }
            _keys.Add(key);
            _values.Add(value);
            return true;
        }
    }

    /// <summary>
    /// Determines whether the dictionary contains the specified key.
    /// </summary>
    /// <param name="key">The key to locate.</param>
    /// <returns>true if the dictionary contains an element with the specified key; otherwise, false.</returns>
    public bool ContainsKey(TKey key)
    {
        lock (_lock)
        {
            return FindKeyIndex(key) >= 0;
        }
    }

    /// <summary>
    /// Attempts to get the value associated with the specified key.
    /// </summary>
    /// <param name="key">The key of the value to get.</param>
    /// <param name="value">When this method returns, contains the value associated with the specified key,
    /// if the key is found; otherwise, the default value for the type of the value parameter.</param>
    /// <returns>true if the dictionary contains an element with the specified key; otherwise, false.</returns>
    public bool TryGetValue(TKey key, out TValue value)
    {
        lock (_lock)
        {
            int index = FindKeyIndex(key);
            if (index >= 0)
            {
                value = _values[index];
                return true;
            }
            value = default!;
            return false;
        }
    }

    /// <summary>
    /// Removes the value with the specified key from the dictionary.
    /// </summary>
    /// <param name="key">The key of the element to remove.</param>
    /// <returns>true if the element was successfully found and removed; otherwise, false.</returns>
    public bool Remove(TKey key)
    {
        lock (_lock)
        {
            int index = FindKeyIndex(key);
            if (index >= 0)
            {
                _keys.RemoveAt(index);
                _values.RemoveAt(index);
                return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Removes all keys and values from the dictionary.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _keys.Clear();
            _values.Clear();
        }
    }

    /// <summary>
    /// Finds the index of the specified key using INumericOperations.Equals for comparison.
    /// </summary>
    /// <param name="key">The key to find.</param>
    /// <returns>The index of the key, or -1 if not found.</returns>
    private int FindKeyIndex(TKey key)
    {
        for (int i = 0; i < _keys.Count; i++)
        {
            if (_numOps.Equals(_keys[i], key))
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Returns an enumerator that iterates through the dictionary.
    /// </summary>
    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        List<KeyValuePair<TKey, TValue>> pairs;
        lock (_lock)
        {
            pairs = new List<KeyValuePair<TKey, TValue>>(_keys.Count);
            for (int i = 0; i < _keys.Count; i++)
            {
                pairs.Add(new KeyValuePair<TKey, TValue>(_keys[i], _values[i]));
            }
        }
        return pairs.GetEnumerator();
    }

    /// <summary>
    /// Returns an enumerator that iterates through the dictionary.
    /// </summary>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
