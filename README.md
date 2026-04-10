# Argus

**A causal trajectory engine for AI agents.**

Most memory systems ask "what did the agent SAY?" Argus asks "what did the agent DO next, and did it WORK?"

Every conversation turn gets compressed into a single 3-character glyph encoding what was touched, what was done, and how it landed. A 10-turn conversation becomes a string like `FL CE mr- !D- CL+`. That string is a **causal fingerprint** — a compressed shape of a decision path.

When the conversation later emits a reversal signal ("wait", "rollback", "that broke it"), Argus reaches back in time and paints the prior glyphs **POISON**. When it sees a success signal ("works", "shipped", "merged"), it paints them **GOLD** and saves what came next as a continuation hint.

On every new turn, before the model runs, Argus fuzzy-matches the session's current glyph trail against its labeled patterns. If the trail resembles a poison path, it injects:

> WARNING: this trajectory has preceded failure 3 times in this workspace.

If it resembles a gold path, it injects:

> HINT: last time this pattern succeeded, the next move was: "restart the model router and re-embed."

No ML. No embeddings. No training data. Just pattern mining with reversal/success signals as the supervision.

## Why It Works

**The compression is the trick.** A glyph throws away the content (file names, model names, words) but preserves the structure (entity type, verb type, valence). Two completely different conversations about completely different topics generate the SAME glyph string if they followed the same decision shape.

This means Argus catches patterns it has never literally seen before. A new failure mode that follows the same shape as a known failure mode gets flagged.

**The retrocausal labeling is the second trick.** You don't need labeled training data. Argus listens for natural language signals — words humans say when something works or breaks — and labels the prior trajectory accordingly. The labels are earned through actual outcomes, not predicted.

## How It Compares

| Approach | What It Compares | Generalizes? |
|----------|-----------------|--------------|
| **Vector search** | Word similarity | Yes, but for content |
| **Knowledge graph** | Explicit relationships | Yes, but only what's connected |
| **Drift detector** | Behavior over time | No, just notices change |
| **Reflection engine** | Decision quality | Yes, but expensive (LLM-based) |
| **Argus** | Shape of causality | Yes, with zero training |

Argus is the only one that compares the **structure** of decision paths.

## The Glyph Alphabet

Every turn becomes 3 characters: `[entity-class][verb-class][valence]`.

```
ENTITY (what was touched):
  F = file        T = tool         U = url
  M = model       K = credential   E = email
  S = service     C = concept      D = database
  X = command     ! = error        . = unknown

VERB (what was done):
  R = read        W = write        E = edit
  X = execute     A = ask          C = create
  L = lookup      F = fix          ? = unknown

VALENCE (how it landed):
  + = success     - = failure      blank = neutral
```

UPPERCASE = inbound (user/event), lowercase = outbound (assistant). You can read a trajectory like `FL mr- !D- CL+` as:

> "User looked at a file → assistant ran a model and it failed → an error was thrown → user looked at a config → success."

Human-readable at a glance. Machine-comparable across analogous situations.

## API

```javascript
import { createArgus } from './argus.mjs';

const argus = createArgus({
  db,        // Your database adapter (see Adapter Interface below)
  router,    // Your model router (only used by reflection — optional)
  config: { minPatternSupport: 2, similarityThreshold: 0.75 },
});

await argus.init();

// Observe a turn
await argus.observe({
  profile: 'agent-1',
  inboundText: 'fix the deployment script',
  outboundText: 'I edited deploy.sh and ran it',
  toolCalls: ['Edit', 'Bash'],
});

// Check if current trajectory predicts trouble or success
const prediction = await argus.check({ profile: 'agent-1' });
// Returns: { poisonMatches: [...], goldMatches: [...], shouldWarn: bool }

// Manually label a trajectory (training)
await argus.label({
  glyphs: 'FE mr- !D-',
  label: 'poison',
  continuation: 'always backup the file first',
});
```

## Endpoints (when run as a server)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/argus/check` | POST | Match current trajectory against patterns |
| `/argus/trajectory` | GET | Show current session glyphs |
| `/argus/patterns` | GET | List labeled patterns by support |
| `/argus/stats` | GET | Turns observed, predictions fired, hit rate |
| `/argus/label` | POST | Manually label a trajectory |
| `/argus/reset` | POST | Clear in-memory trajectory |

## Database Adapter Interface

Argus is database-agnostic. You provide an adapter with these methods:

```javascript
const db = {
  async query(sql, params) { /* execute query, return rows */ },
  async health() { /* return true/false */ },
};
```

The default schema (auto-created on `init()`):

```sql
DEFINE TABLE argus_patterns SCHEMALESS;
-- Each row: { glyphs, label, support, continuation, last_seen, source_session }
```

Works with SurrealDB out of the box. Easy to adapt to Postgres, SQLite, or any JSON store.

## Use Cases

- **Catch repeated failure modes** — your agent keeps making the same kind of mistake
- **Surface winning patterns** — when things work, capture what came next
- **Prevent expensive rollbacks** — warn before the model commits to a path that's failed before
- **Cross-session learning** — patterns learned in one conversation help future conversations
- **Workspace-aware coaching** — what worked in your codebase, not what worked in general

## Status

v0.2.0. Built as part of [Nautilus](https://github.com/Krillian8/nautilus). Extracted into its own repo for focused discussion.

**v0.2 changes (after first audit):**
- Glyphs expanded from 3 to 5 chars (32,768 states vs 288). Added target-modifier and meta dimensions.
- Replaced Jaro-Winkler with positional Dice — recent glyphs weighted higher than old ones.
- Signal hardening: negation handling, conditional/interrogative filtering, code block stripping, 2-signal accumulator, user-only labeling.
- Pattern credibility: minSupport=4, minSessionCount=2 (must appear in 2+ sessions before firing).
- Cooldown: max 1 hint per 5 turns per session.
- Shadow mode: log predictions without injecting, for tuning.
- Better session keying: conversationId/channelId/sessionId before falling back to profile name.

Apache 2.0. Zero dependencies. Single file (~850 lines).

## Discussion

This is a research prototype. The core idea is novel but unproven at scale. Open questions:

- What's the right glyph granularity? Too coarse = everything matches. Too fine = nothing matches.
- How many labeled examples before predictions become reliable?
- Does the technique survive adversarial inputs (prompts designed to look like a known good pattern)?
- Could it work without retrocausal labeling — pure unsupervised pattern mining?

PRs and discussion welcome.
