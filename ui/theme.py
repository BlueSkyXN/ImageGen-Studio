"""Visual system for the Chinese-first Studio interface."""

APP_CSS = r"""
:root {
  --studio-ink: #172033;
  --studio-muted: #667085;
  --studio-line: rgba(84, 96, 128, .16);
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
}

#studio-hero {
  padding: 28px 30px;
  margin: 14px 0 18px;
  border: 1px solid var(--studio-line);
  border-radius: 22px;
  background:
    radial-gradient(circle at 88% 12%, rgba(129, 92, 246, .18), transparent 32%),
    linear-gradient(135deg, rgba(79, 70, 229, .10), rgba(255, 255, 255, .72));
}

#studio-hero h1 { margin: 0 0 8px; color: var(--studio-ink); font-size: clamp(28px, 4vw, 46px); }
#studio-hero p { margin: 0; max-width: 820px; color: var(--studio-muted); font-size: 16px; line-height: 1.7; }

.studio-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
.studio-step { padding: 14px 16px; border: 1px solid var(--studio-line); border-radius: 14px; background: rgba(255,255,255,.62); }
.studio-step b { display: block; margin-bottom: 3px; color: var(--studio-ink); }
.studio-step span { color: var(--studio-muted); font-size: 13px; }

#task-switcher { margin-bottom: 8px; }
#task-switcher label { min-height: 42px; }
.task-help, .model-hint, .result-note { color: var(--studio-muted); }
.model-hint { padding: 10px 12px; border-left: 3px solid #6366f1; background: rgba(99,102,241,.06); border-radius: 8px; }
.prompt-card { padding: 6px; border: 1px solid rgba(99,102,241,.14) !important; }
#result-column { position: sticky; top: 10px; align-self: flex-start; }
.advanced-title { margin-top: 22px; }
.studio-footer { text-align: center; margin: 32px 0 12px; color: var(--studio-muted); font-size: 13px; }

@media (max-width: 900px) {
  .studio-steps { grid-template-columns: 1fr; }
  #workspace-row { flex-direction: column; }
  #result-column { position: static; }
  #studio-hero { padding: 22px 20px; }
}
"""
