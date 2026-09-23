import { useEffect, useState } from "react";
import { DatasetRefForm, Status, EmptyState } from "../components/forms.jsx";
import { api } from "../lib/api.js";

export default function DatasetsView({ datasets, setDatasets }) {
  const [ref, setRef] = useState({
    source: "sklearn",
    identifier: "iris",
    split: "",
    dataset: "",
    target: "",
  });
  const [rows, setRows] = useState("20");
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [uploadStatus, setUploadStatus] = useState({ msg: "", cls: "" });
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const res = await api("/api/datasets");
      setDatasets(res);
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function loadPreview() {
    setBusy(true);
    setStatus({ msg: "Loading…", cls: "" });
    try {
      if (!ref.identifier) throw new Error("Identifier is required");
      const data = await api("/api/datasets/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: ref.source,
          identifier: ref.identifier,
          split: ref.split || null,
          dataset: ref.dataset || null,
          target: ref.target || null,
          n_rows: Number(rows) || 20,
        }),
      });
      setPreview(data);
      setStatus({ msg: "Loaded.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function upload() {
    if (!file) {
      setUploadStatus({ msg: "Choose a file first.", cls: "error" });
      return;
    }
    setUploadStatus({ msg: "Uploading…", cls: "" });
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await api("/api/datasets/upload", { method: "POST", body: form });
      setUploadStatus({ msg: `Uploaded ${data.name} (${data.rows} rows).`, cls: "ok" });
      await refresh();
    } catch (err) {
      setUploadStatus({ msg: err.message, cls: "error" });
    }
  }

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Preview dataset</h2>
        <DatasetRefForm prefix="ds" datasets={datasets} value={ref} onChange={setRef} />
        <label htmlFor="ds-rows">Preview rows</label>
        <input
          id="ds-rows"
          type="number"
          min="1"
          max="500"
          value={rows}
          onChange={(e) => setRows(e.target.value)}
        />
        <button type="button" disabled={busy} onClick={loadPreview}>
          {busy ? "Loading…" : "Load preview"}
        </button>
        <Status msg={status.msg} cls={status.cls} />

        <div className="section-label">Upload file</div>
        <label htmlFor="ds-file">CSV / JSON / Parquet / Excel</label>
        <input
          id="ds-file"
          type="file"
          accept=".csv,.json,.parquet,.xlsx,.xls,.feather"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button type="button" className="secondary" onClick={upload}>
          Upload dataset
        </button>
        <Status msg={uploadStatus.msg} cls={uploadStatus.cls} />
      </section>

      <section className="panel">
        <h2>Datasets</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Source</th>
                <th>Identifier</th>
                <th>Target</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {!datasets.length ? (
                <tr>
                  <td colSpan="5" className="placeholder">
                    No datasets.
                  </td>
                </tr>
              ) : (
                datasets.map((d) => (
                  <tr key={d.name}>
                    <td>{d.name}</td>
                    <td>
                      <span className="pill">{d.source}</span>
                    </td>
                    <td>
                      <code>{d.identifier}</code>
                    </td>
                    <td>{d.target || "—"}</td>
                    <td>
                      <button
                        type="button"
                        className="small secondary"
                        onClick={() => {
                          setRef({
                            source: d.source,
                            identifier: d.identifier,
                            split: d.split || "",
                            dataset: d.name,
                            target: d.target || "",
                          });
                          loadPreviewFor(d);
                        }}
                      >
                        Preview
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="section-label">Preview</div>
        {!preview ? (
          <EmptyState
            title="No dataset selected"
            body="Pick a dataset from the list or enter a source and identifier."
          />
        ) : (
          <PreviewPanel data={preview} />
        )}
      </section>
    </div>
  );

  async function loadPreviewFor(d) {
    setBusy(true);
    setStatus({ msg: "Loading…", cls: "" });
    try {
      const data = await api("/api/datasets/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: d.source,
          identifier: d.identifier,
          split: d.split || null,
          dataset: d.name,
          target: d.target || null,
          n_rows: Number(rows) || 20,
        }),
      });
      setPreview(data);
      setStatus({ msg: "Loaded.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }
}

function PreviewPanel({ data }) {
  const stats = data.columns_stats || [];
  const dist = data.class_distribution;
  return (
    <div>
      <div className="meta">
        <span className="chip">
          <strong>{data.dataset}</strong>
        </span>
        <span className="chip">
          rows <strong>{data.rows}</strong>
        </span>
        <span className="chip">
          cols <strong>{data.columns.length}</strong>
        </span>
        {data.duplicates ? (
          <span className="chip">
            duplicate rows <strong>{data.duplicates}</strong>
          </span>
        ) : null}
        {data.target ? (
          <span className="chip accent">
            target <strong>{data.target}</strong>
          </span>
        ) : null}
        {Object.entries(data.missing).map(([k, v]) => (
          <span className="chip" key={k}>
            missing {k}: <strong>{v}</strong>
          </span>
        ))}
      </div>

      {dist && Object.keys(dist).length ? (
        <>
          <div className="section-label">Class distribution</div>
          <div className="dist-list">
            {Object.entries(dist).map(([k, v]) => {
              const max = Math.max(...Object.values(dist), 1);
              return (
                <div className="dist-row" key={k}>
                  <span className="dist-label">{k}</span>
                  <div className="dist-track">
                    <span style={{ width: `${(v / max) * 100}%` }} />
                  </div>
                  <span className="dist-val">{Number(v).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </>
      ) : null}

      {stats.length ? (
        <>
          <div className="section-label">Column statistics</div>
          <div className="table-wrap">
            <table className="stats-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Dtype</th>
                  <th>Missing</th>
                  <th>Unique</th>
                  <th>Summary</th>
                </tr>
              </thead>
              <tbody>
                {stats.map((s) => (
                  <tr key={s.column}>
                    <td>
                      <code>{s.column}</code>
                    </td>
                    <td>
                      <span className="th-dtype">{s.dtype}</span>
                    </td>
                    <td className="stat-cell">
                      <span>
                        {s.missing}
                        {s.missing_pct ? ` (${s.missing_pct}%)` : ""}
                      </span>
                      {s.missing_pct > 0 ? (
                        <div
                          className={`mini-bar ${
                            s.missing_pct > 20 ? "poor" : s.missing_pct > 5 ? "warn" : ""
                          }`}
                        >
                          <span style={{ width: `${Math.min(100, s.missing_pct)}%` }} />
                        </div>
                      ) : null}
                    </td>
                    <td>{s.unique}</td>
                    <td className="stat-nums">
                      {s.mean != null ? (
                        <>
                          <span className="stat-num">μ {Number(s.mean).toFixed(2)}</span>
                          <span className="stat-num muted-mini">σ {Number(s.std ?? 0).toFixed(2)}</span>
                          <span className="stat-num muted-mini">
                            [{Number(s.min).toFixed(2)}, {Number(s.max).toFixed(2)}]
                          </span>
                        </>
                      ) : s.top != null ? (
                        <>
                          <span className="stat-num">
                            top <strong>{String(s.top)}</strong>
                          </span>
                          <span className="stat-num muted-mini">{s.top_pct ?? 0}%</span>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      <div className="section-label">Preview</div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {data.columns.map((c) => (
                <th key={c}>
                  {c} <span className="th-dtype">{data.dtypes[c]}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.preview.map((row, i) => (
              <tr key={i}>
                {data.columns.map((c) => (
                  <td key={c}>{row[c] === null ? <span className="null">∅</span> : String(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
