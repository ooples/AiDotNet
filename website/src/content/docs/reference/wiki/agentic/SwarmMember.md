---
title: "SwarmMember<T>"
description: "One participant in a `Swarm`: a named persona with its own chat model, instructions, tools, and the set of peers it is allowed to hand control to."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Agents`

One participant in a `Swarm`: a named persona with its own chat model, instructions,
tools, and the set of peers it is allowed to hand control to.

## For Beginners

Picture a support desk where staff pass a customer between them. Each
staff member (a member here) has their own expertise, their own tools, and a list of colleagues they
can transfer the customer to. The conversation history travels with the customer.

## How It Works

Unlike a worker under a `SupervisorAgent` (which runs as a self-contained subroutine),
a swarm member shares one running conversation with its peers. When a member hands off, the next member
continues the *same* conversation, so context flows directly between peers rather than being
summarized through a coordinator.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwarmMember(String,IChatClient<>,String,String,ToolCollection,IReadOnlyList<String>)` | Initializes a new swarm member. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Client` | Gets the chat model the member uses while it is the active responder. |
| `Description` | Gets a short description of the member's specialty. |
| `Handoffs` | Gets the peers this member may hand off to: `null` means "any other member", an empty list means "no handoffs". |
| `Name` | Gets the member's unique name. |
| `SystemPrompt` | Gets the member's instructions/persona, applied while it is active, or `null`. |
| `Tools` | Gets the member's own (non-handoff) tools. |

