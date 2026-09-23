import { useEffect, useState } from "react";
import { Status, EmptyState } from "../components/forms.jsx";
import { api, postJson } from "../lib/api.js";

export default function MonitoringView() {
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [info, setInfo] = useState(null);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uiStatus, setUiStatus] = useState(null);
  const [uiBusy, setUiBusy] = useState(false);
  const [uiMsg, setUiMsg] = useState({ msg: "", cls: "" });

  async function loadUiStatus() {
    try {
      const data = await api("/api/mlflow/ui/status");
      setUiStatus(data);
      return data;
    } catch (err) {
      setUiMsg({ msg: err.message, cls: "error" });
      return null;
    }
  }

  async function startUi() {
    setUiBusy(true);
    setUiMsg({ msg: "Starting MLflow UI…", cls: "" });
    try {
      const data = await postJson("/api/mlflow/ui/start", {});
      setUiMsg({
        msg: data.already_running ? "MLflow UI already running." : "MLflow UI started.",
        cls: "ok",
      });
      await loadUiStatus();
      await load();
    } catch (err) {
      setUiMsg({ msg: err.message, cls: "error" });
    } finally {
      setUiBusy(false);
    }
  }

  async function stopUi() {
    setUiBusy(true);
    setUiMsg({ msg: "Stopping MLflow UI…", cls: "" });
    try {
      await postJson("/api/mlflow/ui/stop", {});
      setUiMsg({ msg: "MLflow UI stopped.", cls: "ok" });
      await loadUiStatus();
      await load();
    } catch (err) {
      setUiMsg({ msg: err.message, cls: "error" });
    } finally {
      setUiBusy(false);
    }
  }

  async function load() {
    setLoading(true);
    setStatus({ msg: "Loading MLflow status…", cls: "" });
    try {
      const [ml, runList, ui] = await Promise.all([
        api("/api/mlflow/status"),
        api("/api/runs"),
        api("/api/mlflow/ui/status").catch(() => null),
      ]);
      setInfo(ml);
      setRuns(runList);
      if (ui) setUiStatus(ui);
      setStatus({
        msg: ml.server_up ? "MLflow UI is reachable." : "MLflow UI is not reachable yet.",
        cls: ml.server_up ? "ok" : "error",
      });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    loadUiStatus();
  }, []);

  const ui = info?.ui_url || uiStatus?.ui_url || "http://127.0.0.1:5000";
  const uiUp = info?.server_up || uiStatus?.server_up;

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>MLflow monitoring</h2>
        <Status msg={status.msg} cls={status.cls} />
        {loading && !info ? (
          <EmptyState title="Checking MLflow…" body="Querying tracking status and run history." />
        ) : info ? (
          <>
            <div className="meta" style={{ marginTop: 8 }}>
              <span className={`pill ${uiUp ? "ok" : "err"}`}>{uiUp ? "UI up" : "UI down"}</span>
              <span className="chip">
                runs <strong>{info.run_count}</strong>
              </span>
              <span className="chip">
                experiment <strong>{info.experiment_name}</strong>
              </span>
              {uiStatus?.managed ? (
                <span className="chip accent">
                  managed pid <strong>{uiStatus.managed_pid ?? uiStatus.pid ?? "—"}</strong>
                </span>
              ) : null}
            </div>
            <div className="section-label">Tracking</div>
            <pre>{info.tracking_uri}</pre>
            <div className="section-label">UI</div>
            <pre>{ui}</pre>
            <div className="actions">
              <button type="button" className="small" onClick={startUi} disabled={uiBusy}>
                {uiBusy ? "Working…" : uiUp ? "Restart MLflow UI" : "Start MLflow UI"}
              </button>
              <button type="button" className="small secondary" onClick={stopUi} disabled={uiBusy}>
                Stop MLflow UI
              </button>
              <a className="button-link" href={ui} target="_blank" rel="noreferrer">
                Open MLflow UI
              </a>
              <button type="button" className="small secondary" onClick={load}>
                Refresh status
              </button>
            </div>
            <Status msg={uiMsg.msg} cls={uiMsg.cls} />
            <div className="section-label">How monitoring works</div>
            <ul className="help-list">
              <li>
                Train with <strong>Log to MLflow</strong> enabled to record params, metrics, and model
                artifacts.
              </li>
              <li>
                Use <strong>Start MLflow UI</strong> to launch the tracking UI on demand (port 5000 by
                default).
              </li>
              <li>PyTrain’s Runs tab reads the same tracking store for comparison and management.</li>
            </ul>
          </>
        ) : (
          <EmptyState title="No MLflow status" body="Could not load monitoring info." />
        )}
      </section>

      <section className="panel">
        <h2>Recent runs</h2>
        {!runs.length ? (
          <EmptyState
            title="No logged runs"
            body="Enable “Log to MLflow” when training to populate monitoring data."
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Key metrics</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 20).map((r) => (
                  <tr key={r.run_id}>
                    <td>
                      <code>{r.run_id.slice(0, 8)}</code>
                    </td>
                    <td>{r.name || "—"}</td>
                    <td>
                      <span
                        className={`pill ${
                          r.status === "FINISHED" ? "ok" : r.status === "FAILED" ? "err" : "run"
                        }`}
                      >
                        {(r.status || "").toLowerCase()}
                      </span>
                    </td>
                    <td className="metric-line">
                      {Object.entries(r.metrics)
                        .filter(([k]) => !k.startsWith("cv_") && !k.endsWith("_time"))
                        .slice(0, 3)
                        .map(([k, v]) => `${k}=${Number(v).toFixed(3)}`)
                        .join(" ") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="actions">
          <a className="button-link secondary" href={ui} target="_blank" rel="noreferrer">
            Open MLflow UI ↗
          </a>
        </div>
      </section>
    </div>
  );
}
