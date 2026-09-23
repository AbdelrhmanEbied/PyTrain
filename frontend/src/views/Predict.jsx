import { useEffect, useMemo, useState } from "react";
import { EmptyState, Status } from "../components/forms.jsx";
import { api, postJson } from "../lib/api.js";

export default function PredictView({ initialRunId }) {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(initialRunId || "");
  const [schema, setSchema] = useState(null);
  const [values, setValues] = useState({});
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [storage, setStorage] = useState(null);

  useEffect(() => {
    Promise.all([
      api("/api/runs").catch(() => []),
      api("/api/storage").catch(() => null),
    ]).then(([runList, store]) => {
      setRuns(runList || []);
      setStorage(store);
      const finished = (runList || []).find((r) => r.status === "FINISHED");
      if (!initialRunId && finished) setRunId(finished.run_id);
      if (initialRunId) setRunId(initialRunId);
    });
  }, [initialRunId]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    setSchema(null);
    setResult(null);
    setStatus({ msg: "Loading schema…", cls: "" });
    api(`/api/predict/schema?run_id=${encodeURIComponent(runId)}`)
      .then((data) => {
        if (cancelled) return;
        setSchema(data);
        const next = {};
        for (const f of data.features || []) {
          next[f.name] = f.sample != null ? f.sample : f.mean != null ? f.mean : "";
        }
        setValues(next);
        setStatus({ msg: `${data.n_features} feature(s) ready.`, cls: "ok" });
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus({ msg: err.message, cls: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const runOptions = useMemo(
    () => runs.filter((r) => r.status === "FINISHED" || r.status === "FAILED"),
    [runs]
  );

  async function onPredict() {
    if (!runId || !schema) return;
    setBusy(true);
    setStatus({ msg: "Predicting…", cls: "" });
    setResult(null);
    try {
      const features = {};
      for (const f of schema.features) {
        const raw = values[f.name];
        if (raw === "" || raw == null) throw new Error(`Feature ${f.name} is required`);
        const n = Number(raw);
        features[f.name] = Number.isFinite(n) ? n : raw;
      }
      const data = await postJson("/api/predict", { run_id: runId, features });
      setResult(data);
      setStatus({ msg: "Prediction ready.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  function loadSample() {
    if (!schema) return;
    const next = {};
    for (const f of schema.features) {
      next[f.name] = f.sample != null ? f.sample : f.mean != null ? f.mean : "";
    }
    setValues(next);
    setStatus({ msg: "Sample row loaded.", cls: "ok" });
  }

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Try a model</h2>
        {!runOptions.length ? (
          <EmptyState
            title="No logged runs"
            body="Train with “Log to MLflow” enabled, then come back to try predictions."
          />
        ) : (
          <>
            <label htmlFor="pred-run">Run</label>
            <select id="pred-run" value={runId} onChange={(e) => setRunId(e.target.value)}>
              <option value="">— select a run —</option>
              {runOptions.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {(r.name || r.run_id.slice(0, 8)) + " · " + (r.params?.model_name || "model")}
                </option>
              ))}
            </select>

            {schema ? (
              <>
                <div className="meta" style={{ marginTop: 12 }}>
                  <span className="chip accent">
                    {schema.task}
                  </span>
                  <span className="chip">
                    model <strong>{schema.model_name || "—"}</strong>
                  </span>
                  <span className="chip">
                    features <strong>{schema.n_features}</strong>
                  </span>
                  {schema.target ? (
                    <span className="chip">
                      target <strong>{schema.target}</strong>
                    </span>
                  ) : null}
                </div>

                <div className="section-label">Features</div>
                <div className="predict-grid">
                  {schema.features.map((f) => (
                    <div key={f.name}>
                      <label htmlFor={`feat-${f.name}`}>{f.name}</label>
                      <input
                        id={`feat-${f.name}`}
                        type="number"
                        step="any"
                        value={values[f.name] ?? ""}
                        onChange={(e) =>
                          setValues((v) => ({ ...v, [f.name]: e.target.value }))
                        }
                      />
                    </div>
                  ))}
                </div>

                <div className="actions">
                  <button type="button" disabled={busy} onClick={onPredict}>
                    {busy ? "Predicting…" : "Predict"}
                  </button>
                  <button type="button" className="small secondary" onClick={loadSample}>
                    Load sample row
                  </button>
                </div>
              </>
            ) : (
              <div className="status-line">Select a run to load its feature schema.</div>
            )}
            <Status msg={status.msg} cls={status.cls} />
          </>
        )}
      </section>

      <section className="panel">
        <h2>Prediction</h2>
        {!result ? (
          <EmptyState
            title="No prediction yet"
            body="Pick a logged run, fill features (or load a sample), then hit Predict."
          />
        ) : (
          <div>
            <div className="meta">
              <span className="chip accent">
                {result.task} · <strong>{String(result.prediction)}</strong>
              </span>
              <span className="chip">
                run <strong>{result.run_id.slice(0, 8)}</strong>
              </span>
            </div>

            {result.probabilities ? (
              <>
                <div className="section-label">Class probabilities</div>
                <div className="metrics">
                  {Object.entries(result.probabilities).map(([k, p]) => (
                    <div key={k} className="metric-card">
                      <span className="metric-label">P({k})</span>
                      <span className="metric-value">{(p * 100).toFixed(1)}%</span>
                      <div className="prob-bar">
                        <div className="prob-fill" style={{ width: `${Math.min(100, p * 100)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : null}

            <div className="section-label">Raw response</div>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}

        <div className="section-label">Where models live</div>
        {!storage ? (
          <div className="status-line">Storage paths unavailable.</div>
        ) : (
          <pre>
            {JSON.stringify(
              {
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
        )}
      </section>
    </div>
  );
}
