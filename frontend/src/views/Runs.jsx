import { useEffect, useState } from "react";
import { MetricCard, Status, EmptyState } from "../components/forms.jsx";
import PipelineGraph from "../components/PipelineGraph.jsx";
import { api, deleteJson, patchJson, postJson } from "../lib/api.js";

function higherBetter(key) {
  return !["mse", "rmse", "mae", "train_time", "eval_time"].includes(key);
}

export default function RunsView() {
  const [runs, setRuns] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [active, setActive] = useState(null);
  const [detail, setDetail] = useState(null);
  const [compare, setCompare] = useState([]);
  const [evalState, setEvalState] = useState({ msg: "", cls: "", data: null });
  const [busy, setBusy] = useState(false);
  const [renameMode, setRenameMode] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [storage, setStorage] = useState(null);
  const [showStorage, setShowStorage] = useState(false);

  async function loadRuns() {
    setStatus({ msg: "Loading…", cls: "" });
    try {
      const data = await api("/api/runs");
      setRuns(data);
      setStatus({ msg: `${data.length} run(s).`, cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    }
  }

  async function loadStorage() {
    try {
      const data = await api("/api/storage");
      setStorage(data);
      setShowStorage(true);
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    }
  }

  useEffect(() => {
    loadRuns();
  }, []);

  function toggleSelect(id) {
    setSelectedIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]));
  }

  function doCompare() {
    const chosen = runs.filter((r) => selectedIds.includes(r.run_id));
    setCompare(chosen);
  }

  async function openRun(run) {
    setBusy(true);
    setActive(run);
    setDetail(null);
    setEvalState({ msg: "", cls: "", data: null });
    setRenameMode(false);
    setRenameValue(run.name || "");
    setStatus({ msg: "Loading run details…", cls: "" });
    try {
      const data = await api(`/api/runs/${run.run_id}/details`);
      setDetail(data);
      setStatus({ msg: "Loaded.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function renameRun() {
    if (!active || !renameValue.trim()) return;
    setBusy(true);
    try {
      const data = await patchJson(`/api/runs/${active.run_id}`, { name: renameValue.trim() });
      setDetail(data);
      setActive((r) => ({ ...r, name: data.name }));
      setRenameMode(false);
      setStatus({ msg: "Renamed.", cls: "ok" });
      await loadRuns();
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function removeRun() {
    if (!active) return;
    if (!window.confirm(`Delete run ${active.run_id}?`)) return;
    setBusy(true);
    try {
      await deleteJson(`/api/runs/${active.run_id}`);
      setStatus({ msg: "Deleted.", cls: "ok" });
      setActive(null);
      setDetail(null);
      setEvalState({ msg: "", cls: "", data: null });
      await loadRuns();
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function restoreRun() {
    if (!active) return;
    setBusy(true);
    try {
      await postJson(`/api/runs/${active.run_id}/restore`, {});
      setStatus({ msg: "Restored.", cls: "ok" });
      await loadRuns();
      await openRun(active);
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function evaluate() {
    if (!active) return;
    setEvalState({ msg: "Evaluating…", cls: "", data: null });
    try {
      const data = await postJson("/api/evaluate", { run_id: active.run_id });
      setEvalState({ msg: "", cls: "ok", data });
    } catch (err) {
      setEvalState({ msg: err.message, cls: "error", data: null });
    }
  }

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Run history</h2>
        <div className="actions">
          <button type="button" className="small" onClick={loadRuns}>
            Refresh
          </button>
          <button
            type="button"
            className="small secondary"
            disabled={selectedIds.length < 2}
            onClick={doCompare}
          >
            {selectedIds.length >= 2 ? `Compare selected (${selectedIds.length})` : "Compare selected"}
          </button>
          <button type="button" className="small secondary" onClick={loadStorage}>
            Where are models?
          </button>
        </div>
        <Status msg={status.msg} cls={status.cls} />
        {showStorage && storage ? (
          <div className="storage-panel">
            <div className="section-label">Storage paths</div>
            <pre>
              {JSON.stringify(
                {
                  project_root: storage.project_root,
                  tracking_uri: storage.tracking_uri,
                  tracking_db: storage.tracking_db,
                  artifact_root: storage.artifact_root,
                  data_uploads: storage.data_uploads,
                  data_versions: storage.data_versions,
                },
                null,
                2
              )}
            </pre>
            <div className="muted-mini">
              Each run’s model artifact is listed under Artifacts when you open a run.
            </div>
          </div>
        ) : null}
        <div className="table-wrap runs-wrap" style={{ marginTop: 12 }}>
          <table className="runs-table">
            <thead>
              <tr>
                <th></th>
                <th>Run</th>
                <th>Status</th>
                <th>Metrics</th>
                <th className="actions-col"></th>
              </tr>
            </thead>
            <tbody>
              {!runs.length ? (
                <tr>
                  <td colSpan="5" className="placeholder">
                    No runs yet. Train with “Log to MLflow” enabled.
                  </td>
                </tr>
              ) : (
                runs.map((run) => {
                  const metricStr = Object.entries(run.metrics)
                    .filter(([k]) => !k.startsWith("train_time") && !k.startsWith("cv_"))
                    .slice(0, 3)
                    .map(([k, v]) => `${k}=${Number(v).toFixed(3)}`)
                    .join(" · ");
                  return (
                    <tr key={run.run_id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(run.run_id)}
                          onChange={() => toggleSelect(run.run_id)}
                        />
                      </td>
                      <td>
                        <code>{run.run_id.slice(0, 8)}</code>
                        <br />
                        <span className="muted-mini">{run.name || ""}</span>
                      </td>
                      <td>
                        {run.status === "FINISHED" ? (
                          <span className="pill ok">finished</span>
                        ) : run.status === "FAILED" ? (
                          <span className="pill err">failed</span>
                        ) : (
                          <span className="pill run">{(run.status || "").toLowerCase()}</span>
                        )}
                      </td>
                      <td className="metric-line">{metricStr || "—"}</td>
                      <td className="actions-col">
                        <button type="button" className="small secondary" onClick={() => openRun(run)}>
                          Open
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        {compare.length >= 2 ? <CompareTable runs={compare} /> : null}
      </section>

      <section className="panel">
        <h2>Run detail</h2>
        {!active ? (
          <EmptyState
            title="No run selected"
            body="Select a run to inspect full details, rename, evaluate, or download the model."
          />
        ) : (
          <RunDetailPanel
            run={active}
            detail={detail}
            busy={busy}
            renameMode={renameMode}
            renameValue={renameValue}
            setRenameMode={setRenameMode}
            setRenameValue={setRenameValue}
            onRename={renameRun}
            onDelete={removeRun}
            onRestore={restoreRun}
            onEvaluate={evaluate}
            evalState={evalState}
          />
        )}
      </section>
    </div>
  );
}

function CompareTable({ runs }) {
  const keys = [...new Set(runs.flatMap((r) => Object.keys(r.metrics)))].sort();
  const best = {};
  for (const k of keys) {
    const vals = runs.map((r) => Number(r.metrics[k])).filter(Number.isFinite);
    if (!vals.length) continue;
    best[k] = higherBetter(k) ? Math.max(...vals) : Math.min(...vals);
  }
  return (
    <div>
      <div className="section-label">Comparison ({runs.length} runs)</div>
      <div className="table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              <th>Metric</th>
              {runs.map((r) => (
                <th key={r.run_id} title={r.run_id}>
                  {r.name || r.run_id.slice(0, 8)}
                  <br />
                  <code className="muted-mini">{r.run_id.slice(0, 8)}</code>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k}>
                <td>
                  <code>{k}</code>
                </td>
                {runs.map((r) => {
                  const v = Number(r.metrics[k]);
                  if (!Number.isFinite(v)) return <td className="muted-mini">—</td>;
                  const isBest = best[k] != null && Math.abs(v - best[k]) < 1e-12;
                  const shown = k.endsWith("_time") ? `${v.toFixed(4)}s` : v.toFixed(4);
                  return (
                    <td key={r.run_id} className={isBest ? "compare-best" : ""}>
                      {shown}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RunDetailPanel({
  run,
  detail,
  busy,
  renameMode,
  renameValue,
  setRenameMode,
  setRenameValue,
  onRename,
  onDelete,
  onRestore,
  onEvaluate,
  evalState,
}) {
  const d = detail || run;
  const pipeline = d.pipeline || [];
  const history = d.metric_history || {};
  const artifacts = d.artifacts || [];
  const ops = d.preprocess_ops || [];
  const split = d.split_config || {};
  const hyper = d.model_hyperparams || {};
  const ds = d.dataset_info || {};

  return (
    <div>
      <div className="meta">
        <span className="chip">
          <strong>{(d.run_id || "").slice(0, 12)}</strong>
        </span>
        <span
          className={`pill ${
            d.status === "FINISHED" ? "ok" : d.status === "FAILED" ? "err" : "run"
          }`}
        >
          {(d.status || "").toLowerCase()}
        </span>
        {d.name ? (
          <span className="chip accent">
            name <strong>{d.name}</strong>
          </span>
        ) : null}
        {d.duration_ms != null ? (
          <span className="chip">
            duration <strong>{(d.duration_ms / 1000).toFixed(2)}s</strong>
          </span>
        ) : null}
        {d.lifecycle_stage ? (
          <span className="chip">
            lifecycle <strong>{d.lifecycle_stage}</strong>
          </span>
        ) : null}
        {(d.params || {}).model_name ? (
          <span className="chip">
            <strong>{d.params.model_name}</strong>
          </span>
        ) : null}
        {(ds.dataset || ds.identifier) ? (
          <span className="chip">
            data <strong>{ds.dataset || ds.identifier}</strong>
          </span>
        ) : null}
        {ops.length ? (
          <span className="chip accent">
            preprocess <strong>{ops.length}</strong>
          </span>
        ) : null}
      </div>

      {renameMode ? (
        <div className="actions">
          <input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            placeholder="Run name"
            style={{ maxWidth: 240 }}
          />
          <button type="button" className="small" onClick={onRename} disabled={busy || !renameValue.trim()}>
            Save
          </button>
          <button type="button" className="small secondary" onClick={() => setRenameMode(false)}>
            Cancel
          </button>
        </div>
      ) : (
        <div className="actions">
          <button type="button" className="small secondary" onClick={() => setRenameMode(true)}>
            Rename
          </button>
          <button type="button" className="small" onClick={onEvaluate} disabled={busy}>
            Evaluate
          </button>
          <button
            type="button"
            className="small secondary"
            onClick={() => {
              window.location.href = `/api/runs/${run.run_id}/model`;
            }}
          >
            Download model
          </button>
          <button type="button" className="small secondary" onClick={onRestore} disabled={busy}>
            Restore
          </button>
          <button type="button" className="small secondary" onClick={onDelete} disabled={busy}>
            Delete
          </button>
        </div>
      )}
      <Status msg={evalState.msg} cls={evalState.cls} />

      {!detail ? (
        <div className="status-line">Loading full details…</div>
      ) : (
        <>
          {pipeline.length ? (
            <>
              <div className="section-label">Pipeline</div>
              <PipelineGraph
                steps={pipeline}
                startLabel="Dataset"
                endLabel="Logged"
                selected={-1}
                endStatus="done"
              />
            </>
          ) : null}

          <div className="section-label">Metrics</div>
          <div className="metrics">
            {Object.entries(d.metrics || {}).length ? (
              Object.entries(d.metrics).map(([k, v]) => <MetricCard key={k} label={k} value={v} />)
            ) : (
              <span className="placeholder">None</span>
            )}
          </div>

          {Object.keys(history).length ? (
            <>
              <div className="section-label">Metric history</div>
              <MetricHistory history={history} />
            </>
          ) : null}

          <div className="section-label">Dataset</div>
          <pre>{JSON.stringify(ds, null, 2)}</pre>

          <div className="section-label">Split</div>
          <pre>{JSON.stringify(split, null, 2)}</pre>

          {ops.length ? (
            <>
              <div className="section-label">Preprocess ops</div>
              <pre>{JSON.stringify(ops, null, 2)}</pre>
            </>
          ) : null}

          <div className="section-label">Model hyperparams</div>
          <pre>{JSON.stringify(hyper, null, 2)}</pre>

          {d.grid_config ? (
            <>
              <div className="section-label">Grid search</div>
              <pre>{JSON.stringify(d.grid_config, null, 2)}</pre>
            </>
          ) : null}

          <div className="section-label">All params</div>
          <pre>{JSON.stringify(d.params, null, 2)}</pre>

          {artifacts.length ? (
            <>
              <div className="section-label">Artifacts</div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Path</th>
                      <th>Type</th>
                      <th>Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {artifacts.map((a) => (
                      <tr key={a.path}>
                        <td>
                          <code>{a.path}</code>
                        </td>
                        <td>{a.is_dir ? "dir" : "file"}</td>
                        <td>{a.file_size != null ? a.file_size : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}

          {d.artifact_uri ? (
            <>
              <div className="section-label">Model location</div>
              <pre>{d.artifact_uri}</pre>
              <div className="muted-mini">
                Downloadable model: /api/runs/{d.run_id}/model · Storage root via Runs → Where are models?
              </div>
            </>
          ) : null}

          {evalState.data ? <EvalDetails data={evalState.data} /> : null}
        </>
      )}
    </div>
  );
}

function MetricHistory({ history }) {
  const keys = Object.keys(history);
  if (!keys.length) return null;
  return (
    <div className="history-grid">
      {keys.map((k) => (
        <MetricSpark key={k} name={k} points={history[k] || []} />
      ))}
    </div>
  );
}

function MetricSpark({ name, points }) {
  const vals = points.map((p) => Number(p.value)).filter(Number.isFinite);
  if (!vals.length) return null;
  const w = 280;
  const h = 120;
  const pad = 28;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;
  const sx = (i) => pad + (points.length <= 1 ? 0 : (i / (points.length - 1)) * (w - pad * 2));
  const sy = (v) => h - pad - ((v - min) / span) * (h - pad * 2);
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(Number(p.value)).toFixed(1)}`)
    .join(" ");
  return (
    <div className="chart-card history-card">
      <div className="history-title">
        <code>{name}</code>
        <span className="muted-mini">
          last {vals[vals.length - 1].toFixed(4)} · min {min.toFixed(4)} · max {max.toFixed(4)}
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="scatter-svg" role="img" aria-label={name}>
        <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="var(--border)" />
        <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="var(--border)" />
        <path d={path} fill="none" stroke="rgba(79,140,255,0.9)" strokeWidth="2" />
        {points.map((p, i) => (
          <circle key={i} cx={sx(i)} cy={sy(Number(p.value))} r="3" fill="rgba(53,196,160,0.9)" />
        ))}
      </svg>
    </div>
  );
}

function EvalDetails({ data }) {
  const cm = data.confusion_matrix;
  const samples = data.samples || [];
  return (
    <div>
      <div className="section-label">Evaluation ({data.n_test} samples)</div>
      <div className="metrics">
        {Object.entries(data.metrics).map(([k, v]) => (
          <MetricCard key={k} label={k} value={v} />
        ))}
      </div>
      {cm ? <ConfusionMatrix cm={cm} /> : null}
      {data.task === "regression" && samples.length ? <Scatter samples={samples} /> : null}
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  const labels = cm.labels.map(String);
  const flat = cm.matrix.flat();
  const max = Math.max(...flat, 1);
  return (
    <div>
      <div className="section-label">Confusion matrix</div>
      <div className="cm-wrap">
        <div className="cm-axis-y">Actual</div>
        <table className="cm-table">
          <thead>
            <tr>
              <th></th>
              {labels.map((l) => (
                <th className="cm-label" key={l}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cm.matrix.map((row, i) => (
              <tr key={i}>
                <th className="cm-label">{labels[i]}</th>
                {row.map((v, j) => {
                  const intensity = v / max;
                  const diag = i === j;
                  const style = diag
                    ? { background: `rgba(53,196,160,${0.12 + intensity * 0.75})` }
                    : { background: `rgba(79,140,255,${0.06 + intensity * 0.65})` };
                  return (
                    <td key={j} className={`cm-cell ${diag ? "diag" : ""}`} style={style}>
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="cm-axis-x">Predicted</div>
      </div>
    </div>
  );
}

function Scatter({ samples }) {
  const w = 320;
  const h = 240;
  const pad = 36;
  const xs = samples.map((s) => s.actual);
  const ys = samples.map((s) => s.predicted);
  const min = Math.min(...xs, ...ys);
  const max = Math.max(...xs, ...ys);
  const span = max - min || 1;
  const sx = (v) => pad + ((v - min) / span) * (w - pad * 2);
  const sy = (v) => h - pad - ((v - min) / span) * (h - pad * 2);
  return (
    <div>
      <div className="section-label">Predicted vs actual</div>
      <div className="chart-card">
        <svg viewBox={`0 0 ${w} ${h}`} className="scatter-svg" role="img" aria-label="Predicted vs actual">
          <line
            x1={pad}
            y1={h - pad}
            x2={w - pad}
            y2={pad}
            stroke="rgba(53,196,160,0.7)"
            strokeDasharray="4 4"
          />
          <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="var(--border)" />
          <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="var(--border)" />
          {samples.map((s, i) => (
            <circle key={i} cx={sx(s.actual)} cy={sy(s.predicted)} r="3" fill="rgba(79,140,255,0.75)" />
          ))}
        </svg>
      </div>
    </div>
  );
}
