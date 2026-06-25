---
title: "Swarm<T>"
description: "A peer-to-peer multi-agent team where control transfers between members over one shared conversation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Agents`

A peer-to-peer multi-agent team where control transfers between members over one shared conversation.
Unlike a `SupervisorAgent` (a hub that runs workers as subroutines), a swarm has no
central coordinator: whichever member is active answers directly, and may hand the whole conversation
to a peer, who then continues from the same history.

## For Beginners

Imagine specialists passing a customer between them. The customer (the
conversation) stays the same; whoever is currently helping can either answer or say "let me transfer you
to my colleague". This class runs that back-and-forth and gives you the final answer plus which member
gave it.

## How It Works

Handoffs are expressed as native tool calls (`transfer_to_<peer>`). The swarm intercepts these
at the loop level: rather than executing them, it switches the active member and re-runs the turn with
the new member's instructions and tools against the unchanged conversation. A member's own (non-handoff)
tools are executed normally. The whole team shares one `MaxIterations` budget,
which guarantees termination even if two members would otherwise ping-pong.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Swarm(IReadOnlyList<SwarmMember<>>,String,SwarmOptions)` | Initializes a new swarm. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `RunAsync(IReadOnlyList<ChatMessage>,CancellationToken)` |  |
| `RunAsync(String,CancellationToken)` | Runs the swarm against a single user request, starting with the entry member. |

