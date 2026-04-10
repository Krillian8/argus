/**
 * Nautilus Argus — Causal Trajectory Engine  v0.2
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Most memory systems ask: "what did the agent SAY?"
 * Argus asks:             "what did the agent DO next, and did it WORK?"
 *
 * ─── The idea ──────────────────────────────────────────────────────────────
 *
 * Every turn is compressed to a 5-char glyph: [entity][verb][target][valence][meta]
 * Ten turns becomes a string like "F1r0s+ Fr0.n Mc0?r" — a causal fingerprint.
 * That string is a TRAJECTORY. Argus stores every trajectory it sees.
 * When a reversal signal fires, it retroactively POISONS the prior N glyphs.
 * When a success signal fires, it GILDS them.
 * On every new turn, Argus checks the live trajectory tail against labeled
 * history. If it resembles a POISON path, it injects a warning into context.
 *
 * Nobody does this. Argus compares the *shape of causality itself.*
 *
 * ─── Glyph Alphabet (5 chars, 32,768 total states) ─────────────────────────
 *
 *  Pos 1 — Entity class (hex 0-f):
 *    0=file  1=tool  2=url   3=model  4=credential  5=email  6=service
 *    7=concept  8=table  9=command  a=error  b=test  c=log  d=network
 *    e=package  f=git
 *
 *  Pos 2 — Verb class (hex 0-f):
 *    0=fix  1=create  2=run  3=read  4=delete  5=write  6=verify  7=plan
 *    8=restart  9=install  a=deploy  b=debug  c=commit  d=revert  e=ask  f=abort
 *
 *  Pos 3 — Target modifier (0-7):
 *    0=self  1=system  2=user  3=remote  4=local  5=ephemeral  6=persistent  7=unknown
 *
 *  Pos 4 — Valence (4 states):
 *    + positive  - negative  ? uncertain  . neutral
 *
 *  Pos 5 — Meta (4 states):
 *    n=normal  r=retry  c=cascade  f=first-time
 *
 *  Example glyph: "02r+n" = file/run/remote/positive/normal
 *                 "a0s-r" = error/fix/self/negative/retry
 *
 * ─── Endpoints ─────────────────────────────────────────────────────────────
 *
 *   POST /argus/check        — What does the current trajectory tail predict?
 *   GET  /argus/trajectory   — Current session's live trajectory glyphs
 *   GET  /argus/patterns     — Top labeled patterns (gold + poison)
 *   GET  /argus/stats        — Totals, hit rate, effectiveness
 *   POST /argus/label        — Manual label override (gold|poison|neutral)
 *   POST /argus/reset        — Clear the in-memory session trajectory
 *
 * ─── Usage ─────────────────────────────────────────────────────────────────
 *
 *   import { createArgus } from './argus.mjs';
 *   argus = createArgus({ db, router, memory, graph, config: config.argus });
 *   argus.attachTo(context);   // self-wraps beforeProxy / afterProxy
 *   await argus.init();
 *
 * ─── CHANGELOG ─────────────────────────────────────────────────────────────
 *
 *   v0.2 (2026-04-08):
 *   - 5-char glyphs (was 3): entity/verb/target/valence/meta — 32,768 states
 *   - Positional Dice similarity replaces Jaro-Winkler (correct for categorical seqs)
 *   - Weighted average scoring: positional*0.7 + overallDice*0.3 (was max)
 *   - Signal hardening: negation filter, conditional filter, code-block stripping,
 *     source separation (user vs assistant), 2-hit-in-3-turns requirement
 *   - labelWindow: 3 for poison (was 6), 6 for gold (unchanged)
 *   - Fixed labelWindow slice math (no silent shrink on young sessions)
 *   - minPatternSupport raised from 2 to 4; added session_count >= 2 gate
 *   - Cooldown: max 1 Argus hint per 5 turns per session
 *   - Entity collision: multi-entity tracking via meta 'c' (cascade flag)
 *   - Shadow mode: shadowMode:true records predictions without injecting hints
 *   - Better session keying: conversationId > profile name (no "default" conflation)
 */

// Minimal logger
const log = (level, category, message, meta) => {
  const ts = new Date().toISOString();
  const metaStr = meta ? ' ' + JSON.stringify(meta) : '';
  const line = `${ts} | ${level.toUpperCase().padEnd(5)} | ${category} | ${message}${metaStr}`;
  if (level === 'error') console.error(line);
  else console.log(line);
};

// ═══════════════════════════════════════════════════════════════════════════
// Factory
// ═══════════════════════════════════════════════════════════════════════════

export function createArgus({ db, router, memory, graph, config = {} }) {
  return new Argus({ db, router, memory, graph, config });
}

// ═══════════════════════════════════════════════════════════════════════════
// Glyph Alphabet
// ═══════════════════════════════════════════════════════════════════════════

// Pos 1: Entity class — hex char
const ENTITY_MAP = {
  file: '0', tool: '1', url: '2', model: '3', credential: '4',
  email: '5', service: '6', concept: '7', table: '8', command: '9',
  error: 'a', test: 'b', log: 'c', network: 'd', package: 'e', git: 'f',
};

// Entity detection priority (highest priority = set pos 1, mark cascade if multiple)
const ENTITY_PRIORITY = [
  'model', 'service', 'error', 'git', 'file', 'tool', 'table',
  'package', 'url', 'credential', 'email', 'network', 'concept', 'test', 'log', 'command',
];

// Entity detection patterns (keyed by entity type)
const ENTITY_PATTERNS = {
  error:      /\b(error|exception|traceback|failed|crash|fatal|panic)\b/i,
  model:      /\b(model|llm|gpt|claude|gemma|mistral|deepseek|qwen|ollama)\b/i,
  git:        /\b(git|commit|push|pull|branch|merge|rebase|pr|diff|HEAD)\b/i,
  service:    /\b(server|service|daemon|worker|api|endpoint|port|gateway)\b/i,
  file:       /\.[a-z]{1,5}(\b|$)/i,
  tool:       /\b(tool|function|plugin|extension|hook|middleware)\b/i,
  table:      /\b(table|database|db|sql|query|record|schema|surreal)\b/i,
  package:    /\b(npm|pip|cargo|yarn|install|package|dependency|module)\b/i,
  url:        /https?:\/\//i,
  credential: /\b(token|key|secret|auth|password|credential|api[_-]?key)\b/i,
  email:      /\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b/i,
  network:    /\b(dns|ip|tcp|http|ssl|tls|proxy|firewall|route|socket)\b/i,
  concept:    /\b(concept|idea|plan|strategy|approach|pattern|design)\b/i,
  test:       /\b(test|spec|assert|expect|jest|mocha|pytest|unit|e2e)\b/i,
  log:        /\b(log|logs|logging|trace|stdout|stderr|journal)\b/i,
  command:    /\b(run|exec|command|cmd|bash|shell|script|cli)\b/i,
};

// Pos 2: Verb class — hex char
const VERB_PATTERNS = [
  [/\b(fix|patch|repair|resolve|correct|hotfix)\b/i,               '0'],
  [/\b(add|create|build|make|implement|generate|new|init)\b/i,     '1'],
  [/\b(run|execute|launch|invoke|call|exec|trigger)\b/i,           '2'],
  [/\b(read|check|look|inspect|see|view|show|find|grep|list)\b/i,  '3'],
  [/\b(delete|remove|kill|stop|drop|clear|wipe|purge)\b/i,         '4'],
  [/\b(write|update|modify|edit|set|rename|move|save|push)\b/i,    '5'],
  [/\b(test|verify|validate|ensure|confirm|assert)\b/i,            '6'],
  [/\b(plan|think|consider|should|would|maybe|design)\b/i,         '7'],
  [/\b(restart|reboot|reset|reload|refresh|bounce)\b/i,            '8'],
  [/\b(install|setup|configure|provision|bootstrap)\b/i,           '9'],
  [/\b(deploy|release|ship|publish|promote|rollout)\b/i,           'a'],
  [/\b(debug|diagnose|investigate|trace|profile)\b/i,              'b'],
  [/\b(commit|stage|stash|merge|rebase|cherry-pick)\b/i,           'c'],
  [/\b(revert|rollback|undo|restore|recover)\b/i,                  'd'],
  [/\b(ask|request|query|prompt|clarify)\b/i,                      'e'],
  [/\b(abort|cancel|terminate|halt|reject|skip)\b/i,               'f'],
];

// Pos 3: Target modifier
const TARGET_PATTERNS = [
  [/\b(self|itself|itself|this|current)\b/i,            '0'],
  [/\b(system|os|kernel|process|global)\b/i,            '1'],
  [/\b(user|aaron|human|you|me)\b/i,                    '2'],
  [/\b(remote|cloud|api|server|prod|production)\b/i,    '3'],
  [/\b(local|localhost|127|dev|development|machine)\b/i,'4'],
  [/\b(temp|tmp|cache|ephemeral|transient)\b/i,         '5'],
  [/\b(persist|db|disk|file|storage|permanent)\b/i,     '6'],
];

// Pos 4: Valence  (+ - ? .)
// Applied AFTER stripping code blocks
const VALENCE_POSITIVE = /(?<!not |don't |doesn't |didn't |never )(\b(works?|worked|shipped|fixed|merged|deployed|success|done|perfect|confirmed|passing|green|live|good|great|pushed|synced)\b)/i;
const VALENCE_NEGATIVE = /(?<!not |don't |doesn't |didn't |never )(\b(oops|broke|broken|failed|failing|error|crash|revert|rollback|undo|mistake|bug|problem|regress|wrong)\b)/i;
const VALENCE_UNCERTAIN = /\b(might|maybe|perhaps|not sure|unclear|could|possibly|probably)\b/i;

// ═══════════════════════════════════════════════════════════════════════════
// Signal patterns — hardened v0.2
// ═══════════════════════════════════════════════════════════════════════════

// Reversal signals (user-emitted only — must NOT be negated, NOT conditional)
const REVERSAL_PATTERNS = [
  /^(wait|hold on|actually|no,|nope|oops|scratch that)/im,
  /\b(that was wrong|that's wrong|broke it|undo|revert that|rolling back)\b/i,
  /\b(disregard|ignore that|forget what i said|never ?mind)\b/i,
  /\b(shit|damn|fuck).{0,40}\b(broke|failed|wrong|crashed)\b/i,
];

// Conditionals/interrogatives that disqualify a reversal
const CONDITIONAL_SKIP = /\b(if|when|could|would)\b|\?$/im;

// Success signals (user-emitted only)
const SUCCESS_PATTERNS = [
  /\b(that works?|it works?|works now|fixed it|nailed it|perfect|ship ?it)\b/i,
  /\b(merged|deployed|pushed|released|live in production)\b/i,
  /\b(all green|tests pass|passing|ci green|no errors|clean build)\b/i,
  /\b(beautiful|love it|exactly|that's the one)\b/i,
];

// Negation prefix (for signals)
const NEGATION_PREFIX = /(?:not |don't |doesn't |didn't |never )/i;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Strip code blocks (```...```) and indented blocks (4+ space / tab prefix)
 * from text before scanning for valence/signals.
 */
function stripCodeBlocks(text) {
  if (!text) return '';
  // Remove fenced code blocks
  let stripped = text.replace(/```[\s\S]*?```/g, ' CODE_BLOCK ');
  // Remove inline code
  stripped = stripped.replace(/`[^`\n]+`/g, ' INLINE_CODE ');
  // Remove lines that look like stack traces / deeply indented output
  stripped = stripped.split('\n').filter(line => {
    const trimmed = line.trimStart();
    const indent = line.length - trimmed.length;
    // Skip lines with 4+ spaces of indentation (tracebacks, code output)
    if (indent >= 4) return false;
    // Skip lines that look like stack frames
    if (/^\s+at\s|^\s+File\s|^\s+line\s\d/i.test(line)) return false;
    return true;
  }).join('\n');
  return stripped;
}

/**
 * Check if a sentence containing a signal keyword is negated or conditional.
 * Returns true if the match should be skipped.
 */
function isNegatedOrConditional(text, match) {
  // Get the 60 chars before the match
  const matchIdx = text.indexOf(match[0]);
  if (matchIdx < 0) return false;
  const before = text.slice(Math.max(0, matchIdx - 60), matchIdx);
  if (NEGATION_PREFIX.test(before)) return true;
  // Check if sentence is conditional/interrogative
  const sentenceStart = before.lastIndexOf('.') + 1;
  const sentence = before.slice(sentenceStart) + match[0];
  if (CONDITIONAL_SKIP.test(sentence)) return true;
  return false;
}

/**
 * Returns true if any pattern matches text and is NOT negated/conditional.
 */
function matchesSignal(text, patterns) {
  const clean = stripCodeBlocks(text);
  for (const p of patterns) {
    const m = p.exec(clean);
    if (m && !isNegatedOrConditional(clean, m)) return true;
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Argus
// ═══════════════════════════════════════════════════════════════════════════

class Argus {
  constructor({ db, router, memory, graph, config }) {
    this.db = db;
    this.router = router;
    this.memory = memory;
    this.graph = graph;
    this.cfg = {
      enabled: true,
      // How many recent glyphs form the "tail" we match against history
      tailLength: 5,
      // Minimum positional-dice similarity to trigger a prediction (0..1)
      matchThreshold: 0.72,
      // How far back (in glyphs) to label when signal fires
      labelWindowPoison: 3,   // rupture is recent — tight window
      labelWindowGold: 6,     // success may have long build-up
      // Max patterns kept in RAM for fast matching
      hotCacheSize: 500,
      // Namespace + db for persistence
      namespace: null,
      database: null,
      // Emit warnings/hints as context enrichment
      injectHints: true,
      // Shadow mode: record predictions but don't inject
      shadowMode: false,
      // Max matches to show in /argus/check
      maxMatches: 5,
      // Pattern must be seen this many times to fire
      minPatternSupport: 4,
      // Pattern must appear in this many distinct sessions to fire
      minSessionCount: 2,
      // Max 1 hint per N turns (cooldown)
      hintCooldownTurns: 5,
      // Require N signal hits within K turns before labeling
      signalHitsRequired: 2,
      signalHitsWindow: 3,
      ...config,
    };

    // Per-session state
    this._sessions = new Map();

    // Hot cache of labeled patterns
    this._hotPoison = [];
    this._hotGold   = [];

    this._stats = {
      turnsObserved:    0,
      glyphsEmitted:    0,
      reversalsCaught:  0,
      successesCaught:  0,
      predictionsFired: 0,
      poisonWarnings:   0,
      goldHints:        0,
      shadowPredictions:0,
      startedAt:        Date.now(),
    };

    this._initialized = false;
  }

  // ─── Init ─────────────────────────────────────────────────────────────────

  async init() {
    if (!this.cfg.enabled) {
      log('info', 'argus', 'disabled via config');
      return;
    }

    try {
      const ns = this.cfg.namespace || this.db.namespace;
      const dbName = this.cfg.database || this.db.database || 'memory';
      await this.db.recallFromDb(ns, dbName, `
        DEFINE TABLE IF NOT EXISTS argus_patterns SCHEMALESS;
        DEFINE INDEX IF NOT EXISTS argus_label_idx ON argus_patterns FIELDS label;
        DEFINE INDEX IF NOT EXISTS argus_support_idx ON argus_patterns FIELDS support;
        DEFINE TABLE IF NOT EXISTS argus_shadow SCHEMALESS;
      `);
    } catch (e) {
      log('warn', 'argus', `schema init skipped: ${e.message}`);
    }

    await this._warmCache();
    this._initialized = true;
    log('info', 'argus', `initialized v0.2 (${this._hotGold.length} gold, ${this._hotPoison.length} poison, tail=${this.cfg.tailLength}, threshold=${this.cfg.matchThreshold}, shadow=${this.cfg.shadowMode})`);
  }

  async _warmCache() {
    try {
      const ns = this.cfg.namespace || this.db.namespace;
      const dbName = this.cfg.database || this.db.database || 'memory';
      const rows = await this.db.recallFromDb(ns, dbName, `
        SELECT glyphs, label, support, session_count, continuation, last_seen
        FROM argus_patterns
        WHERE support >= ${this.cfg.minPatternSupport}
        ORDER BY support DESC
        LIMIT ${this.cfg.hotCacheSize}
      `);
      if (!Array.isArray(rows)) return;
      for (const r of rows) {
        const entry = {
          glyphs:       r.glyphs || '',
          support:      r.support || 1,
          sessionCount: r.session_count || 1,
          continuation: r.continuation || null,
          lastSeen:     r.last_seen || 0,
        };
        if (r.label === 'gold')   this._hotGold.push(entry);
        else if (r.label === 'poison') this._hotPoison.push(entry);
      }
    } catch {
      // No table yet — fine.
    }
  }

  // ─── Self-attach to Context Engine ────────────────────────────────────────

  attachTo(contextEngine) {
    if (!contextEngine || !this.cfg.enabled) return;
    if (contextEngine._argusAttached) return;
    contextEngine._argusAttached = true;

    const origBefore = contextEngine.beforeProxy.bind(contextEngine);
    const origAfter  = contextEngine.afterProxy.bind(contextEngine);

    contextEngine.beforeProxy = async (parsed, profile) => {
      const result = await origBefore(parsed, profile);
      try {
        await this.observeInbound(parsed, profile);
      } catch (e) {
        log('warn', 'argus', `observeInbound failed: ${e.message}`);
      }
      return result;
    };

    contextEngine.afterProxy = async (parsed, responseData, profile) => {
      const result = await origAfter(parsed, responseData, profile);
      try {
        await this.observeOutbound(parsed, responseData, profile);
      } catch (e) {
        log('warn', 'argus', `observeOutbound failed: ${e.message}`);
      }
      return result;
    };

    log('info', 'argus', 'attached to context engine');
  }

  // ─── Session keying ───────────────────────────────────────────────────────
  //
  // v0.2: prefer conversationId from profile/context to avoid conflating
  // multiple agents under a single generic key.

  _sessionKey(profile) {
    return profile?.conversationId
      || profile?.channelId
      || profile?.sessionId
      || profile?.name
      || 'session_unknown';
  }

  _getSession(key) {
    if (!this._sessions.has(key)) {
      this._sessions.set(key, {
        glyphs:          [],
        lastUpdate:      Date.now(),
        turnCount:       0,
        lastPrediction:  null,
        lastHintTurn:    -999,
        // Signal accumulator: track recent user reversal/success hits
        recentReversals: [], // turn numbers where reversal hit
        recentSuccesses: [], // turn numbers where success hit
      });
    }
    return this._sessions.get(key);
  }

  // ─── Observation Hooks ────────────────────────────────────────────────────

  async observeInbound(parsed, profile) {
    if (!this.cfg.enabled || !parsed?.messages) return;

    const sessionKey = this._sessionKey(profile);
    const session = this._getSession(sessionKey);

    const lastUser = this._lastRealUserMessage(parsed.messages);
    if (!lastUser) return;

    const glyph = this._glyphFromText(lastUser, 'user');
    session.glyphs.push(glyph);
    session.turnCount++;
    session.lastUpdate = Date.now();
    this._stats.glyphsEmitted++;
    this._stats.turnsObserved++;

    if (session.glyphs.length > 64) session.glyphs.shift();

    // Cooldown: skip if we hinted too recently
    const turnsSinceHint = session.turnCount - session.lastHintTurn;
    if (turnsSinceHint < this.cfg.hintCooldownTurns) {
      session.lastPrediction = null;
      return;
    }

    const prediction = this._matchTrajectory(session.glyphs);
    session.lastPrediction = prediction;

    if (prediction) {
      if (this.cfg.shadowMode) {
        // Shadow mode: log but don't inject
        this._stats.shadowPredictions++;
        this._logShadow(sessionKey, session, prediction).catch(() => {});
      } else if (this.cfg.injectHints) {
        this._injectHint(parsed.messages, prediction);
        session.lastHintTurn = session.turnCount;
        this._stats.predictionsFired++;
        if (prediction.label === 'poison') this._stats.poisonWarnings++;
        else if (prediction.label === 'gold') this._stats.goldHints++;
      }
    }
  }

  async observeOutbound(parsed, responseData, profile) {
    if (!this.cfg.enabled) return;
    const sessionKey = this._sessionKey(profile);
    const session = this._getSession(sessionKey);

    const content = responseData?.choices?.[0]?.message?.content || '';
    if (!content) return;

    // Assistant reply also gets glyphed (lowercase first char convention is gone
    // in v0.2 — direction encoded in meta pos 5 'a' vs 'n' would be complex;
    // keep user glyphs as canonical, assistant glyphs for trajectory completeness)
    const glyph = this._glyphFromText(content, 'assistant');
    session.glyphs.push(glyph);
    this._stats.glyphsEmitted++;
    if (session.glyphs.length > 64) session.glyphs.shift();

    // ── Signal detection (v0.2: user-only, accumulator, code-stripped) ──
    const lastUser = this._lastRealUserMessage(parsed.messages) || '';

    // USER reversal signal
    if (matchesSignal(lastUser, REVERSAL_PATTERNS)) {
      session.recentReversals.push(session.turnCount);
      // Trim to signalHitsWindow
      const cutoff = session.turnCount - this.cfg.signalHitsWindow;
      session.recentReversals = session.recentReversals.filter(t => t >= cutoff);

      if (session.recentReversals.length >= this.cfg.signalHitsRequired) {
        await this._labelRetroactive(session, 'poison', sessionKey, lastUser);
        this._stats.reversalsCaught++;
        session.recentReversals = []; // reset accumulator
      }
    }

    // USER success signal (not assistant self-congratulation)
    if (matchesSignal(lastUser, SUCCESS_PATTERNS)) {
      session.recentSuccesses.push(session.turnCount);
      const cutoff = session.turnCount - this.cfg.signalHitsWindow;
      session.recentSuccesses = session.recentSuccesses.filter(t => t >= cutoff);

      if (session.recentSuccesses.length >= this.cfg.signalHitsRequired) {
        await this._labelRetroactive(session, 'gold', sessionKey, lastUser);
        this._stats.successesCaught++;
        session.recentSuccesses = [];
      }
    }
  }

  // ─── Glyph extraction (v0.2: 5 chars) ────────────────────────────────────
  //
  // [entity][verb][target][valence][meta]
  //
  // Meta pos 5:
  //   n = normal
  //   r = retry (contains "again", "retry", "re-run")
  //   c = cascade (multiple entity types detected)
  //   f = first-time (turn 1 in session)

  _glyphFromText(text, source) {
    const raw    = (text || '').slice(0, 1000);
    const clean  = stripCodeBlocks(raw);
    const snippet = clean.slice(0, 800);

    // ── Pos 1: Entity ──
    let entityChar = '7'; // concept = default
    let entityHits = 0;
    let topEntity = null;

    // Use graph extractor if available
    if (this.graph?.extractEntities) {
      try {
        const ents = this.graph.extractEntities(snippet, 'argus');
        if (ents.length > 0) {
          entityHits = ents.length;
          for (const p of ENTITY_PRIORITY) {
            if (ents.some(e => e.type === p)) {
              topEntity = p;
              entityChar = ENTITY_MAP[p] || '7';
              break;
            }
          }
        }
      } catch {}
    }

    // Fallback regex detection
    if (!topEntity) {
      for (const p of ENTITY_PRIORITY) {
        if (ENTITY_PATTERNS[p]?.test(snippet)) {
          if (!topEntity) {
            topEntity = p;
            entityChar = ENTITY_MAP[p] || '7';
          }
          entityHits++;
        }
      }
    }

    // ── Pos 2: Verb ──
    let verbChar = '7'; // plan = default
    for (const [pat, char] of VERB_PATTERNS) {
      if (pat.test(snippet)) { verbChar = char; break; }
    }

    // ── Pos 3: Target ──
    let targetChar = '7'; // unknown = default
    for (const [pat, char] of TARGET_PATTERNS) {
      if (pat.test(snippet)) { targetChar = char; break; }
    }

    // ── Pos 4: Valence (on stripped text) ──
    let valence = '.';
    if (VALENCE_NEGATIVE.test(clean))  valence = '-';
    else if (VALENCE_POSITIVE.test(clean))  valence = '+';
    else if (VALENCE_UNCERTAIN.test(clean)) valence = '?';

    // ── Pos 5: Meta ──
    let meta = 'n';
    if (source === 'user' && this._sessions.size > 0) {
      // Check if this is turn 1 — caller is responsible for session state,
      // we approximate via the glyph buffer being empty
    }
    if (/\b(again|retry|re-run|redo|one more time)\b/i.test(snippet)) meta = 'r';
    else if (entityHits > 1) meta = 'c'; // cascade: multiple entities
    // first-time detection happens at caller level; we do best effort
    else meta = 'n';

    return `${entityChar}${verbChar}${targetChar}${valence}${meta}`;
  }

  _lastRealUserMessage(messages) {
    if (!Array.isArray(messages)) return null;
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role !== 'user') continue;
      const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '');
      if (c.includes('<session-context') || c.includes('<nautilus-context') || c.includes('<argus-hint')) continue;
      return c;
    }
    return null;
  }

  // ─── Retroactive labeling ─────────────────────────────────────────────────

  async _labelRetroactive(session, label, sessionKey, signalText = '') {
    const window = label === 'poison' ? this.cfg.labelWindowPoison : this.cfg.labelWindowGold;

    // v0.2 fixed slice math: take [len - 1 - window, len - 1) so we never
    // include the signal glyph itself, and the window doesn't silently shrink.
    const len = session.glyphs.length;
    const from = Math.max(0, len - 1 - window);
    const tail = session.glyphs.slice(from, len - 1);
    if (tail.length < 2) return;

    const glyphString = tail.join(' ');
    const continuation = label === 'gold' && signalText
      ? signalText.slice(0, 240).replace(/\s+/g, ' ').trim()
      : null;

    await this._upsertPattern(glyphString, label, continuation, sessionKey);
  }

  async _upsertPattern(glyphString, label, continuation, sessionKey = null) {
    try {
      const ns = this.cfg.namespace || this.db.namespace;
      const dbName = this.cfg.database || this.db.database || 'memory';

      const existing = await this.db.recallFromDb(ns, dbName, `
        SELECT id, support, session_count, sessions
        FROM argus_patterns
        WHERE glyphs = $glyphs AND label = $label
        LIMIT 1
      `, { glyphs: glyphString, label });

      if (Array.isArray(existing) && existing.length > 0) {
        const row = existing[0];
        const recordId = typeof row.id === 'object' ? (row.id.id || row.id) : row.id;
        if (!/^[a-z_]+:[a-z0-9_]+$/i.test(String(recordId))) return;

        // Track session diversity
        const seenSessions = Array.isArray(row.sessions) ? row.sessions : [];
        const isNewSession = sessionKey && !seenSessions.includes(sessionKey);
        const newSessions = isNewSession ? [...seenSessions, sessionKey] : seenSessions;
        const newSessionCount = newSessions.length;

        await this.db.recallFromDb(ns, dbName, `
          UPDATE ${recordId} SET
            support = support + 1,
            session_count = $sessionCount,
            sessions = $sessions,
            last_seen = time::millis()
            ${continuation ? `, continuation = $continuation` : ''}
        `, {
          sessionCount: newSessionCount,
          sessions: newSessions,
          ...(continuation ? { continuation } : {}),
        });

        // Update hot cache
        const cacheBucket = label === 'gold' ? this._hotGold : this._hotPoison;
        const entry = cacheBucket.find(p => p.glyphs === glyphString);
        if (entry) {
          entry.support++;
          entry.lastSeen = Date.now();
          if (isNewSession) entry.sessionCount = newSessionCount;
          if (continuation) entry.continuation = continuation;
        }
      } else {
        const sessions = sessionKey ? [sessionKey] : [];
        await this.db.recallFromDb(ns, dbName, `
          CREATE argus_patterns SET
            glyphs = $glyphs,
            label = $label,
            support = 1,
            session_count = $sessionCount,
            sessions = $sessions,
            continuation = ${continuation ? '$continuation' : 'NONE'},
            first_seen = time::millis(),
            last_seen = time::millis()
        `, {
          glyphs: glyphString,
          label,
          sessionCount: sessions.length,
          sessions,
          ...(continuation ? { continuation } : {}),
        });

        // Add to hot cache (won't fire warnings yet — below minPatternSupport)
        const cacheBucket = label === 'gold' ? this._hotGold : this._hotPoison;
        cacheBucket.push({
          glyphs: glyphString,
          support: 1,
          sessionCount: sessions.length,
          continuation,
          lastSeen: Date.now(),
        });
        if (cacheBucket.length > this.cfg.hotCacheSize) {
          cacheBucket.sort((a, b) => b.support - a.support || b.lastSeen - a.lastSeen);
          cacheBucket.length = this.cfg.hotCacheSize;
        }
      }

      log('debug', 'argus', `labeled ${label}: ${glyphString}`);
    } catch (e) {
      log('warn', 'argus', `pattern upsert failed: ${e.message}`);
    }
  }

  async _logShadow(sessionKey, session, prediction) {
    try {
      const ns = this.cfg.namespace || this.db.namespace;
      const dbName = this.cfg.database || this.db.database || 'memory';
      await this.db.recallFromDb(ns, dbName, `
        CREATE argus_shadow SET
          session_key = $sessionKey,
          tail = $tail,
          predicted_label = $label,
          score = $score,
          pattern = $pattern,
          turn = $turn,
          ts = time::millis()
      `, {
        sessionKey,
        tail: prediction.tail,
        label: prediction.label,
        score: prediction.score,
        pattern: prediction.pattern,
        turn: session.turnCount,
      });
    } catch { /* shadow failures are non-critical */ }
  }

  // ─── Trajectory matching (v0.2: positional Dice) ──────────────────────────

  _matchTrajectory(glyphs) {
    if (glyphs.length < 3) return null;
    const tail = glyphs.slice(-this.cfg.tailLength);
    const tailStr = tail.join(' ');

    // Check poison first (warnings outrank hints)
    const poisonMatch = this._bestMatch(tail, tailStr, this._hotPoison);
    if (
      poisonMatch &&
      poisonMatch.score >= this.cfg.matchThreshold &&
      poisonMatch.pattern.support >= this.cfg.minPatternSupport &&
      (poisonMatch.pattern.sessionCount || 1) >= this.cfg.minSessionCount
    ) {
      return {
        label:   'poison',
        score:   poisonMatch.score,
        pattern: poisonMatch.pattern.glyphs,
        support: poisonMatch.pattern.support,
        tail:    tailStr,
      };
    }

    const goldMatch = this._bestMatch(tail, tailStr, this._hotGold);
    if (
      goldMatch &&
      goldMatch.score >= this.cfg.matchThreshold &&
      goldMatch.pattern.support >= this.cfg.minPatternSupport &&
      (goldMatch.pattern.sessionCount || 1) >= this.cfg.minSessionCount
    ) {
      return {
        label:        'gold',
        score:        goldMatch.score,
        pattern:      goldMatch.pattern.glyphs,
        support:      goldMatch.pattern.support,
        continuation: goldMatch.pattern.continuation,
        tail:         tailStr,
      };
    }

    return null;
  }

  /**
   * Positional Dice similarity (v0.2).
   *
   * Encode each glyph as `${glyph}_${position_from_end}` so that the most
   * recent glyph contributes most. Weight: pos 0 = 1.0, pos 1 = 0.8, etc.
   * Final score = (positionalDice * 0.7) + (overallDice * 0.3)
   */
  _bestMatch(tailArr, tailStr, patterns) {
    let best = null;
    for (const p of patterns) {
      const score = this._positionalDice(tailArr, tailStr, p.glyphs);
      if (!best || score > best.score) best = { score, pattern: p };
    }
    return best;
  }

  _positionalDice(tailArr, tailStr, patternStr) {
    const patArr = patternStr.split(' ').filter(Boolean);
    if (tailArr.length === 0 || patArr.length === 0) return 0;

    // Position-weighted token sets (from end)
    const encode = (arr) => {
      const tokens = new Map();
      for (let i = 0; i < arr.length; i++) {
        const posFromEnd = arr.length - 1 - i;
        const weight = Math.max(0.2, 1.0 - posFromEnd * 0.2);
        const key = `${arr[i]}_${posFromEnd}`;
        tokens.set(key, (tokens.get(key) || 0) + weight);
      }
      return tokens;
    };

    const tMap = encode(tailArr);
    const pMap = encode(patArr);

    // Weighted intersection
    let intersection = 0;
    let tTotal = 0;
    let pTotal = 0;
    for (const [k, w] of tMap) tTotal += w;
    for (const [k, w] of pMap) {
      pTotal += w;
      const tw = tMap.get(k);
      if (tw) intersection += Math.min(tw, w);
    }

    const positionalScore = tTotal + pTotal > 0
      ? (2 * intersection) / (tTotal + pTotal)
      : 0;

    // Overall Dice on full strings (unweighted, positional-agnostic)
    const overallScore = diceCoefficient(tailStr, patternStr);

    return positionalScore * 0.7 + overallScore * 0.3;
  }

  // ─── Hint injection ───────────────────────────────────────────────────────

  _injectHint(messages, prediction) {
    let body;
    if (prediction.label === 'poison') {
      body = [
        `WARNING: the current trajectory resembles a pattern that has preceded`,
        `failure ${prediction.support} time(s) in this workspace.`,
        `Similar glyph path: ${prediction.pattern} (similarity ${(prediction.score * 100).toFixed(0)}%)`,
        `Pause and consider whether this action will actually work, or if a`,
        `different approach is warranted.`,
      ].join('\n');
    } else {
      const cont = prediction.continuation
        ? `\nLast time this succeeded, the next move was: "${prediction.continuation}"`
        : '';
      body = [
        `HINT: the current trajectory matches a pattern that has succeeded`,
        `${prediction.support} time(s) in this workspace.`,
        `Similar glyph path: ${prediction.pattern} (similarity ${(prediction.score * 100).toFixed(0)}%)${cont}`,
      ].join('\n');
    }

    const hint = {
      role: 'user',
      content: `<argus-hint label="${prediction.label}" score="${prediction.score.toFixed(2)}">\n${body}\n</argus-hint>\n\n(Nautilus Argus pattern match. Advisory only — do not mention unless relevant.)`,
    };

    const lastUserIdx = messages.findLastIndex(m => {
      if (m.role !== 'user') return false;
      const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '');
      return !c.includes('<session-context') && !c.includes('<nautilus-context') && !c.includes('<argus-hint');
    });
    if (lastUserIdx > 0) messages.splice(lastUserIdx, 0, hint);
    else if (lastUserIdx === 0) messages.unshift(hint);
  }

  // ─── HTTP API ─────────────────────────────────────────────────────────────

  async handleApi(req, res, url) {
    const endpoint = url.pathname.split('/')[2];
    const send = (status, data) => {
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(data));
    };

    try {
      if (endpoint === 'stats') {
        const uptimeS = Math.round((Date.now() - this._stats.startedAt) / 1000);
        return send(200, {
          ...this._stats,
          uptimeSeconds:     uptimeS,
          hotPoisonPatterns: this._hotPoison.length,
          hotGoldPatterns:   this._hotGold.length,
          activeSessions:    this._sessions.size,
          config: {
            tailLength:          this.cfg.tailLength,
            matchThreshold:      this.cfg.matchThreshold,
            labelWindowPoison:   this.cfg.labelWindowPoison,
            labelWindowGold:     this.cfg.labelWindowGold,
            minPatternSupport:   this.cfg.minPatternSupport,
            minSessionCount:     this.cfg.minSessionCount,
            hintCooldownTurns:   this.cfg.hintCooldownTurns,
            shadowMode:          this.cfg.shadowMode,
          },
        });
      }

      if (endpoint === 'trajectory') {
        const profile = url.searchParams.get('profile') || url.searchParams.get('session') || 'default';
        const session = this._sessions.get(profile);
        if (!session) return send(200, { profile, glyphs: [], trajectory: '', turnCount: 0 });
        return send(200, {
          profile,
          glyphs:         session.glyphs,
          trajectory:     session.glyphs.join(' '),
          turnCount:      session.turnCount,
          lastUpdate:     session.lastUpdate,
          lastPrediction: session.lastPrediction,
          lastHintTurn:   session.lastHintTurn,
        });
      }

      if (endpoint === 'patterns') {
        const label = url.searchParams.get('label') || 'all';
        const limit = Math.min(parseInt(url.searchParams.get('limit')) || 20, 100);
        const out = { gold: [], poison: [] };
        if (label === 'all' || label === 'gold') {
          out.gold = this._hotGold.slice().sort((a, b) => b.support - a.support).slice(0, limit);
        }
        if (label === 'all' || label === 'poison') {
          out.poison = this._hotPoison.slice().sort((a, b) => b.support - a.support).slice(0, limit);
        }
        return send(200, out);
      }

      if (endpoint === 'check') {
        const body = req.method === 'POST' ? await readBody(req) : '{}';
        let parsed = {};
        try { parsed = JSON.parse(body || '{}'); } catch {}

        let glyphs;
        if (Array.isArray(parsed.glyphs)) {
          glyphs = parsed.glyphs;
        } else {
          const profile = parsed.profile || url.searchParams.get('profile') || 'default';
          const session = this._sessions.get(profile);
          glyphs = session?.glyphs || [];
        }

        if (glyphs.length < 2) return send(200, { match: null, reason: 'trajectory too short' });

        const tailArr = glyphs.slice(-this.cfg.tailLength);
        const tail    = tailArr.join(' ');
        const poisonMatches = this._allMatches(tailArr, tail, this._hotPoison).slice(0, this.cfg.maxMatches);
        const goldMatches   = this._allMatches(tailArr, tail, this._hotGold).slice(0, this.cfg.maxMatches);

        return send(200, {
          tail,
          poisonMatches: poisonMatches.map(m => ({
            pattern:      m.pattern.glyphs,
            score:        Number(m.score.toFixed(3)),
            support:      m.pattern.support,
            sessionCount: m.pattern.sessionCount || 1,
          })),
          goldMatches: goldMatches.map(m => ({
            pattern:      m.pattern.glyphs,
            score:        Number(m.score.toFixed(3)),
            support:      m.pattern.support,
            sessionCount: m.pattern.sessionCount || 1,
            continuation: m.pattern.continuation,
          })),
        });
      }

      if (endpoint === 'label') {
        const body = await readBody(req);
        const parsed = JSON.parse(body || '{}');
        if (!parsed.glyphs || !parsed.label) {
          return send(400, { error: 'glyphs and label required' });
        }
        if (!['gold', 'poison'].includes(parsed.label)) {
          return send(400, { error: 'label must be gold or poison' });
        }
        await this._upsertPattern(parsed.glyphs, parsed.label, parsed.continuation || null, parsed.sessionKey || null);
        return send(200, { labeled: true, glyphs: parsed.glyphs, label: parsed.label });
      }

      if (endpoint === 'reset') {
        const body = req.method === 'POST' ? await readBody(req) : '{}';
        const parsed = JSON.parse(body || '{}');
        const profile = parsed.profile || 'default';
        this._sessions.delete(profile);
        return send(200, { reset: true, profile });
      }

      return send(404, {
        error: `Unknown argus endpoint: ${endpoint}`,
        endpoints: ['stats', 'trajectory', 'patterns', 'check', 'label', 'reset'],
      });
    } catch (e) {
      log('warn', 'argus', `api error on ${endpoint}: ${e.message}`);
      return send(500, { error: e.message });
    }
  }

  _allMatches(tailArr, tailStr, patterns) {
    return patterns
      .map(p => ({ score: this._positionalDice(tailArr, tailStr, p.glyphs), pattern: p }))
      .filter(m => m.score >= this.cfg.matchThreshold * 0.8)
      .sort((a, b) => b.score - a.score);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Similarity helpers — pure JS, zero deps
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Dice coefficient over character bigrams.
 * Kept as a utility (used in _positionalDice's overall component and exported
 * for testing). In v0.2 this is no longer the primary match signal.
 */
export function diceCoefficient(s1, s2) {
  if (!s1 || !s2) return 0;
  if (s1 === s2) return 1;
  if (s1.length < 2 || s2.length < 2) return 0;

  const b1 = bigrams(s1);
  const b2 = bigrams(s2);

  let intersection = 0;
  const map = new Map();
  for (const b of b1) map.set(b, (map.get(b) || 0) + 1);
  for (const b of b2) {
    const count = map.get(b);
    if (count > 0) {
      intersection++;
      map.set(b, count - 1);
    }
  }

  return (2 * intersection) / (b1.length + b2.length);
}

function bigrams(s) {
  const out = [];
  for (let i = 0; i < s.length - 1; i++) out.push(s.slice(i, i + 2));
  return out;
}

/**
 * Jaro-Winkler retained for reference / potential future use.
 * NOT used in matching as of v0.2 (replaced by positional Dice).
 * @deprecated use _positionalDice via Argus instance
 */
export function jaroWinkler(s1, s2) {
  if (!s1 || !s2) return 0;
  if (s1 === s2) return 1;
  const m = _jaro(s1, s2);
  if (m < 0.7) return m;
  let prefix = 0;
  const cap = Math.min(4, s1.length, s2.length);
  for (let i = 0; i < cap; i++) {
    if (s1[i] === s2[i]) prefix++;
    else break;
  }
  return m + prefix * 0.1 * (1 - m);
}

function _jaro(s1, s2) {
  const len1 = s1.length, len2 = s2.length;
  if (!len1 || !len2) return 0;
  const dist = Math.max(0, Math.floor(Math.max(len1, len2) / 2) - 1);
  const m1 = new Array(len1).fill(false);
  const m2 = new Array(len2).fill(false);
  let matches = 0;
  for (let i = 0; i < len1; i++) {
    const lo = Math.max(0, i - dist);
    const hi = Math.min(i + dist + 1, len2);
    for (let j = lo; j < hi; j++) {
      if (m2[j] || s1[i] !== s2[j]) continue;
      m1[i] = m2[j] = true; matches++; break;
    }
  }
  if (!matches) return 0;
  let k = 0, t = 0;
  for (let i = 0; i < len1; i++) {
    if (!m1[i]) continue;
    while (!m2[k]) k++;
    if (s1[i] !== s2[k]) t++;
    k++;
  }
  return (matches / len1 + matches / len2 + (matches - t / 2) / matches) / 3;
}

// ═══════════════════════════════════════════════════════════════════════════
// HTTP body helper (argus is self-contained — no shared utils)
// ═══════════════════════════════════════════════════════════════════════════

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    let size = 0;
    const MAX = 1024 * 1024;
    req.on('data', chunk => {
      size += chunk.length;
      if (size > MAX) { req.destroy(); reject(new Error('Body too large')); return; }
      body += chunk;
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}
