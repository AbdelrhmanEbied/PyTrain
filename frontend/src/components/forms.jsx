import { useMemo, useState } from "react";

export function DatasetRefForm({ prefix, datasets, value, onChange }) {
  const quick = value.dataset || "";
  const source = value.source || "sklearn";
  const showSplit = source === "huggingface";

  return (
    <div className="ref-form">
      <label htmlFor={`${prefix}-quick`}>Registered dataset</label>
      <select
        id={`${prefix}-quick`}
        value={quick}
        onChange={(e) => {
          const name = e.target.value;
          if (!name) {
            onChange({ ...value, dataset: "" });
            return;
          }
          const d = datasets.find((x) => x.name === name);
          if (!d) return;
          onChange({
            dataset: d.name,
            source: d.source,
            identifier: d.identifier,
            split: d.split || "",
            target: d.target || value.target || "",
          });
        }}
      >
        <option value="">— custom —</option>
        {datasets.map((d) => (
          <option key={d.name} value={d.name}>
            {d.name} ({d.source})
          </option>
        ))}
      </select>

      <label htmlFor={`${prefix}-source`}>Source</label>
      <select
        id={`${prefix}-source`}
        value={source}
        onChange={(e) => onChange({ ...value, source: e.target.value })}
      >
        <option value="sklearn">sklearn</option>
        <option value="local">local path</option>
        <option value="huggingface">huggingface</option>
      </select>

      <label htmlFor={`${prefix}-identifier`}>Identifier</label>
      <input
        id={`${prefix}-identifier`}
        value={value.identifier || ""}
        placeholder="iris, data/titanic.csv, openai/gsm8k"
        onChange={(e) => onChange({ ...value, identifier: e.target.value })}
      />

      {showSplit ? (
        <>
          <label htmlFor={`${prefix}-split`}>Split</label>
          <input
            id={`${prefix}-split`}
            value={value.split || ""}
            placeholder="train"
            onChange={(e) => onChange({ ...value, split: e.target.value })}
          />
        </>
      ) : null}
    </div>
  );
}

const GROUP_LABELS = {
  general: "General",
  trees: "Trees",
  regularization: "Regularization",
  optimization: "Optimization",
  reproducibility: "Reproducibility",
  boosting: "Boosting",
  sampling: "Sampling",
};

const GROUP_ORDER = [
  "general",
  "regularization",
  "boosting",
  "optimization",
  "trees",
  "sampling",
  "reproducibility",
];

export function ParamForm({ params, values, onChange }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const groups = useMemo(() => {
    const map = new Map();
    for (const p of params || []) {
      const g = p.group || "general";
      if (!map.has(g)) map.set(g, []);
      map.get(g).push(p);
    }
    return [...map.entries()].sort((a, b) => {
      const ia = GROUP_ORDER.indexOf(a[0]);
      const ib = GROUP_ORDER.indexOf(b[0]);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
  }, [params]);

  const hasAdvanced = (params || []).some((p) => p.advanced);

  return (
    <div className="param-form">
      {groups.map(([g, list]) => (
        <div className="param-group" key={g}>
          <div className="param-group-title">{GROUP_LABELS[g] || g}</div>
          {list.map((p) => {
            if (p.advanced && !showAdvanced) return null;
            const id = `param-${p.name}`;
            const ph = p.default === null ? "null" : String(p.default);
            return (
              <div className="param-field" key={p.name}>
                <label htmlFor={id}>
                  {p.name}
                  {p.description ? (
                    <span className="help-dot" title={p.description}>
                      ?
                    </span>
                  ) : null}
                  {p.min_value != null || p.max_value != null ? (
                    <span className="param-range">
                      {p.min_value ?? "−∞"} … {p.max_value ?? "∞"}
                    </span>
                  ) : null}
                </label>
                {p.choices ? (
                  <select
                    id={id}
                    value={values[p.name] ?? ""}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === "") {
                        const next = { ...values };
                        delete next[p.name];
                        onChange(next);
                      } else onChange({ ...values, [p.name]: raw });
                    }}
                  >
                    <option value="">default ({ph})</option>
                    {p.choices.map((c) => (
                      <option key={String(c)} value={String(c)}>
                        {String(c)}
                      </option>
                    ))}
                  </select>
                ) : p.type === "bool" ? (
                  <select
                    id={id}
                    value={values[p.name] ?? ""}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === "") {
                        const next = { ...values };
                        delete next[p.name];
                        onChange(next);
                      } else onChange({ ...values, [p.name]: raw === "true" });
                    }}
                  >
                    <option value="">default ({ph})</option>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : (
                  <input
                    id={id}
                    type={p.type === "int" ? "number" : p.type === "float" ? "number" : "text"}
                    step={p.type === "float" ? "any" : p.type === "int" ? "1" : undefined}
                    min={p.min_value ?? undefined}
                    max={p.max_value ?? undefined}
                    placeholder={ph}
                    value={values[p.name] ?? ""}
                    onChange={(e) => {
                      const raw = e.target.value;
                      if (raw === "") {
                        const next = { ...values };
                        delete next[p.name];
                        onChange(next);
                      } else if (p.type === "int") onChange({ ...values, [p.name]: parseInt(raw, 10) });
                      else if (p.type === "float") onChange({ ...values, [p.name]: parseFloat(raw) });
                      else onChange({ ...values, [p.name]: raw });
                    }}
                  />
                )}
                {p.description ? <p className="param-help">{p.description}</p> : null}
              </div>
            );
          })}
        </div>
      ))}
      {hasAdvanced ? (
        <button type="button" className="btn-link" onClick={() => setShowAdvanced((v) => !v)}>
          {showAdvanced ? "Hide advanced" : "Show advanced"} parameters
        </button>
      ) : null}
    </div>
  );
}

export function MetricCard({ label, value }) {
  const key = label.replace(/^(test|val|train) /, "");
  const n = Number(value);
  const lowerBetter = ["mse", "rmse", "mae"].includes(key);
  let quality = "";
  let pct = null;
  let barClass = "";
  if (key === "r2" && Number.isFinite(n)) {
    pct = Math.max(0, Math.min(100, Math.round(n * 100)));
    quality = n >= 0.7 ? "good" : n >= 0.4 ? "mid" : "bad";
    barClass = n >= 0.7 ? "" : n >= 0.4 ? "warn" : "poor";
  } else if (Number.isFinite(n) && n >= 0 && n <= 1) {
    pct = Math.round(n * 100);
    if (lowerBetter) {
      quality = n <= 0.15 ? "good" : n <= 0.4 ? "mid" : "bad";
      barClass = n <= 0.15 ? "" : n <= 0.4 ? "warn" : "poor";
    } else {
      quality = n >= 0.85 ? "good" : n >= 0.6 ? "mid" : "bad";
      barClass = n >= 0.85 ? "" : n >= 0.6 ? "warn" : "poor";
    }
  }
  const shown = Number.isFinite(n) ? n.toFixed(4) : String(value);
  return (
    <div className="metric">
      <div className="k">{label}</div>
      <div className={`v ${quality}`}>{shown}</div>
      {pct != null ? (
        <div className={`bar ${barClass}`}>
          <span style={{ width: `${pct}%` }} />
        </div>
      ) : null}
    </div>
  );
}

export function Status({ msg, cls = "" }) {
  if (!msg) return <div className="status-line" />;
  return <div className={`status-line ${cls}`}>{msg}</div>;
}

export function EmptyState({ title, body }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <span>{body}</span>
    </div>
  );
}
