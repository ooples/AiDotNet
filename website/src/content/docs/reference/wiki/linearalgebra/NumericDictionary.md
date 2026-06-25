---
title: "NumericDictionary<TKey, TValue>"
description: "A thread-safe dictionary implementation that uses INumericOperations for key comparison, allowing generic numeric types T to be used as keys without requiring the notnull constraint."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinearAlgebra`

A thread-safe dictionary implementation that uses INumericOperations for key comparison,
allowing generic numeric types T to be used as keys without requiring the notnull constraint.

## For Beginners

Think of this as a special dictionary designed for numeric keys.

Regular dictionaries in C# have trouble with generic number types because they need special
guarantees about how keys are compared. This dictionary uses the same math operations
(INumericOperations) that the rest of the library uses, so it works seamlessly with any
numeric type (float, double, decimal, etc.).

It's also thread-safe, meaning multiple parts of your code can read from and write to it
at the same time without causing errors.

## How It Works

This collection solves the problem of using generic numeric types as dictionary keys in a generic context.
Standard Dictionary requires TKey to have a notnull constraint, but this library intentionally avoids
such constraints to maintain flexibility. NumericDictionary uses INumericOperations.Equals() for
key comparison instead of the default equality comparer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NumericDictionary` | Initializes a new instance of the NumericDictionary class. |
| `NumericDictionary(Int32)` | Initializes a new instance of the NumericDictionary class with specified initial capacity. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of key/value pairs in the dictionary. |
| `Item()` | Gets or sets the value associated with the specified key. |
| `Keys` | Gets a collection containing the keys in the dictionary. |
| `Values` | Gets a collection containing the values in the dictionary. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(,)` | Adds the specified key and value to the dictionary. |
| `Clear` | Removes all keys and values from the dictionary. |
| `ContainsKey()` | Determines whether the dictionary contains the specified key. |
| `FindKeyIndex()` | Finds the index of the specified key using INumericOperations.Equals for comparison. |
| `GetEnumerator` | Returns an enumerator that iterates through the dictionary. |
| `Remove()` | Removes the value with the specified key from the dictionary. |
| `System#Collections#IEnumerable#GetEnumerator` | Returns an enumerator that iterates through the dictionary. |
| `TryAdd(,)` | Attempts to add the specified key and value to the dictionary. |
| `TryGetValue(,)` | Attempts to get the value associated with the specified key. |

