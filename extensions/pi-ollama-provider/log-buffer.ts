/**
 * Structured log buffer for the Ollama extension.
 *
 * Replaces console.log/error calls that pollute the Pi TUI with a
 * ring buffer of structured entries. Errors and warnings are surfaced
 * via the Pi status bar (using ctx.ui.setStatus) and the /ollama-status
 * command, keeping the conversation view clean.
 *
 * During startup, only `pi` (ExtensionAPI) is available — no UI context.
 * Errors are buffered here and displayed once `session_start` fires
 * (which provides `ctx` with `ctx.ui.setStatus`).
 *
 * Severity levels:
 *   - "warn": shown in status bar and /ollama-status
 *   - "error": shown in status bar and /ollama-status
 *
 * The ring buffer caps at MAX_ENTRIES (20) so stale entries are
 * naturally discarded as new entries arrive. The status bar is
 * refreshed on each session_start event.
 */

// ── types ──

export type LogLevel = "warn" | "error";

export type LogSource = "local" | "cloud" | "cache" | "settings" | "stream" | "auto-pull";

export interface LogEntry {
  /** Unix timestamp (Date.now()) */
  timestamp: number;
  /** Severity level */
  level: LogLevel;
  /** Which subsystem produced this entry */
  source: LogSource;
  /** Human-readable message */
  message: string;
}

// ── ring buffer ──

const MAX_ENTRIES = 20;

const logBuffer: LogEntry[] = [];

function addEntry(level: LogLevel, source: LogSource, message: string): void {
  logBuffer.push({ timestamp: Date.now(), level, source, message });
  if (logBuffer.length > MAX_ENTRIES) {
    logBuffer.splice(0, logBuffer.length - MAX_ENTRIES);
  }
}

// ── public API ──

/**
 * Log a warning. Shown in status bar on session_start and visible
 * in /ollama-status.
 */
export function logWarn(source: LogSource, message: string): void {
  addEntry("warn", source, message);
}

/**
 * Log an error. Shown persistently in status bar and in /ollama-status.
 * Remains in the ring buffer until overwritten by newer entries.
 */
export function logError(source: LogSource, message: string): void {
  addEntry("error", source, message);
}

/** Return all buffered entries (newest last). */
export function getLogEntries(): LogEntry[] {
  return [...logBuffer];
}

/** Return the N most recent entries (newest last). */
export function getRecentEntries(count: number = 5): LogEntry[] {
  return logBuffer.slice(-count);
}

/** Return only error-level entries. */
export function getErrors(): LogEntry[] {
  return logBuffer.filter((e) => e.level === "error");
}

/** Return only warning-level entries. */
export function getWarnings(): LogEntry[] {
  return logBuffer.filter((e) => e.level === "warn");
}

/** Clear all buffered entries. Called on session_start to reset stale state. */
export function clearLog(): void {
  logBuffer.length = 0;
}

/**
 * Format a one-line status string for the Pi footer/status bar.
 *
 * Returns `undefined` if there are no errors or warnings to show.
 *
 * Formatting rules:
 *   - Single entry → show the message directly (no count prefix)
 *   - Multiple errors-only → "N errors — msg1, msg2"
 *   - Multiple warnings-only → "N warnings — msg1, msg2"
 *   - Mixed errors + warnings → "N issues — msg1, msg2"
 *
 * Examples:
 *   "⚠ ollama: local server unreachable"                    (single entry)
 *   "⚠ ollama: 2 errors — local unreachable, auth failed"  (errors-only)
 *   "⚠ ollama: 2 warnings — discarding cache, slow model" (warnings-only)
 *   "⚠ ollama: 3 issues — local unreachable, auth failed, +1 more" (mixed)
 */
export function formatStatusLine(): string | undefined {
  const entries = getRecentEntries(3);
  if (entries.length === 0) return undefined;

  const errorCount = getErrors().length;
  const warningCount = getWarnings().length;

  if (errorCount === 0 && warningCount === 0) return undefined;

  // Single entry — show directly
  if (errorCount + warningCount === 1) {
    const entry = logBuffer[logBuffer.length - 1];
    return `⚠ ollama: ${entry.message}`;
  }

  // Multiple entries — summarize with severity-appropriate label
  const allRecent = entries;
  const totalEntries = logBuffer.length;
  const overflow = totalEntries - allRecent.length;

  const summaries = allRecent.map((e) => e.message);
  const suffix = overflow > 0 ? `, +${overflow} more` : "";

  if (errorCount > 0 && warningCount === 0) {
    return `⚠ ollama: ${errorCount} error${errorCount !== 1 ? "s" : ""} — ${summaries.join(", ")}${suffix}`;
  }

  if (warningCount > 0 && errorCount === 0) {
    return `⚠ ollama: ${warningCount} warning${warningCount !== 1 ? "s" : ""} — ${summaries.join(", ")}${suffix}`;
  }

  // Mixed errors and warnings
  return `⚠ ollama: ${errorCount + warningCount} issue${(errorCount + warningCount) !== 1 ? "s" : ""} — ${summaries.join(", ")}${suffix}`;
}