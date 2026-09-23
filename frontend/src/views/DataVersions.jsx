import { useEffect, useState } from "react";
import { DatasetRefForm, Status, EmptyState } from "../components/forms.jsx";
import { api, deleteJson, postJson } from "../lib/api.js";

export default function DataVersionsView({ datasets, setDatasets }) {
  const [versions, setVersions] = useState([]);
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [selected, setSelected] = useState(null);
  const [diffIds, setDiffIds] = useState({ a: "", b: "" });
  const [diff, setDiff] = useState(null);
  const [preview, setPreview] = useState(null);
  const [message, setMessage] = useState("");
  const [ref, setRef] = useState({
    source: "sklearn",
    identifier: "iris",
    split: "",
    dataset: "",
    target: "",
  });
  const [busy, setBusy] = useState(false);

  async function load() {
    setStatus({ msg: "Loading versions…", cls: "" });
    try {
      const data = await api("/api/data-versions");
      setVersions(data);
      setStatus({ msg: `${data.length} version(s).`, cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function snapshot() {
    setBusy(true);
    setStatus({ msg: "Creating snapshot…", cls: "" });
    try {
      const body = {
        dataset: ref.dataset || null,
        source: ref.dataset ? null : ref.source || null,
        identifier: ref.dataset ? null : ref.identifier || null,
        split: ref.split || null,
        target: ref.target || null,
        message: message.trim(),
      };
      if (!body.dataset && !body.identifier) throw new Error("Choose a dataset or enter an identifier");
      const created = await postJson("/api/data-versions", body);
      setStatus({ msg: `Created ${created.dataset} v${created.version}.`, cls: "ok" });
      setSelected(created);
      setMessage("");
      await load();
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function openVersion(id) {
    setBusy(true);
    setStatus({ msg: "Loading version…", cls: "" });
    try {
      const data = await api(`/api/data-versions/${id}`);
      setSelected(data);
      const prev = await api(`/api/data-versions/${id}/preview?n_rows=20`);
      setPreview(prev);
      setStatus({ msg: "Loaded.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function removeVersion(id) {
    if (!window.confirm(`Delete version ${id}?`)) return;
    setBusy(true);
    try {
      await deleteJson(`/api/data-versions/${id}`);
      setStatus({ msg: "Deleted.", cls: "ok" });
      if (selected?.id === id) {
        setSelected(null);
        setPreview(null);
      }
      await load();
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function restoreVersion(id) {
    setBusy(true);
    setStatus({ msg: "Restoring…", cls: "" });
    try {
      const restored = await postJson(`/api/data-versions/${id}/restore`, {});
      setStatus({ msg: `Restored as ${restored.name}.`, cls: "ok" });
      try {
        const list = await api("/api/datasets");
        if (setDatasets) setDatasets(list);
      } catch {
      }
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function runDiff() {
    if (!diffIds.a || !diffIds.b) {
      setStatus({ msg: "Pick two versions to diff.", cls: "error" });
      return;
    }
    setBusy(true);
    setStatus({ msg: "Computing diff…", cls: "" });
    try {
      const data = await api(`/api/data-versions/diff?a=${encodeURIComponent(diffIds.a)}&b=${encodeURIComponent(diffIds.b)}`);
      setDiff(data);
      setStatus({ msg: "Diff ready.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Create snapshot</h2>
        <DatasetRefForm prefix="dv" datasets={datasets} value={ref} onChange={setRef} />
        <label htmlFor="dv-msg">Message</label>
        <input
          id="dv-msg"
          value={message}
          placeholder="optional note for this version"
          onChange={(e) => setMessage(e.target.value)}
        />
        <div className="actions">
          <button type="button" className="small" onClick={snapshot} disabled={busy}>
            {busy ? "Working…" : "Snapshot dataset"}
          </button>
          <button type="button" className="small secondary" onClick={load} disabled={busy}>
            Refresh
          </button>
        </div>
        <Status msg={status.msg} cls={status.cls} />

        <div className="section-label">Diff two versions</div>
        <div className="row">
          <div>
            <label htmlFor="dv-a">Version A</label>
            <select id="dv-a" value={diffIds.a} onChange={(e) => setDiffIds({ ...diffIds, a: e.target.value })}>
              <option value="">—</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.dataset} v{v.version} ({v.id.slice(0, 6)})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="dv-b">Version B</label>
            <select id="dv-b" value={diffIds.b} onChange={(e) => setDiffIds({ ...diffIds, b: e.target.value })}>
              <option value="">—</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.dataset} v{v.version} ({v.id.slice(0, 6)})
                </option>
              ))}
            </select>
          </div>
        </div>
        <button type="button" className="small secondary" onClick={runDiff} disabled={busy}>
          Compare
        </button>

        {diff ? (
          <div>
            <div className="section-label">Diff result</div>
            <div className="meta">
              <span className={`chip ${diff.same_hash ? "accent" : ""}`}>
                {diff.same_hash ? "identical hash" : "content differs"}
              </span>
              <span className="chip">
                rows <strong>{diff.row_delta >= 0 ? "+" : ""}{diff.row_delta}</strong>
              </span>
              <span className="chip">
                cols <strong>{diff.col_delta >= 0 ? "+" : ""}{diff.col_delta}</strong>
              </span>
            </div>
            {diff.added_columns?.length ? (
              <p className="muted-mini">Added: {diff.added_columns.join(", ")}</p>
            ) : null}
            {diff.removed_columns?.length ? (
              <p className="muted-mini">Removed: {diff.removed_columns.join(", ")}</p>
            ) : null}
            {diff.changed_columns?.length ? (
              <p className="muted-mini">Changed: {diff.changed_columns.join(", ")}</p>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>Data versions</h2>
        {!versions.length ? (
          <EmptyState
            title="No versions yet"
            body="Snapshot a dataset to create a restorable, diffable version."
          />
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th>Ver</th>
                    <th>Shape</th>
                    <th>Message</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {versions.map((v) => (
                    <tr key={v.id}>
                      <td>{v.dataset}</td>
                      <td>v{v.version}</td>
                      <td>
                        {v.rows}×{v.columns}
                      </td>
                      <td className="muted-mini">{v.message || "—"}</td>
                      <td>
                        <div className="actions" style={{ marginTop: 0 }}>
                          <button type="button" className="small secondary" onClick={() => openVersion(v.id)} disabled={busy}>
                            Open
                          </button>
                          <button type="button" className="small secondary" onClick={() => restoreVersion(v.id)} disabled={busy}>
                            Restore
                          </button>
                          <button type="button" className="small secondary" onClick={() => removeVersion(v.id)} disabled={busy}>
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {selected ? (
              <div>
                <div className="section-label">
                  {selected.dataset} v{selected.version}
                </div>
                <div className="meta">
                  <span className="chip">
                    {selected.rows}×{selected.columns}
                  </span>
                  <span className="chip">
                    missing <strong>{selected.missing_total ?? 0}</strong>
                  </span>
                  <span className="chip">
                    dup <strong>{selected.duplicates ?? 0}</strong>
                  </span>
                  <span className="chip">
                    hash <strong>{(selected.hash || "").slice(0, 8)}</strong>
                  </span>
                </div>
                {preview ? (
                  <>
                    <div className="section-label">Preview</div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            {preview.columns.map((c) => (
                              <th key={c}>{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {preview.preview.map((row, i) => (
                            <tr key={i}>
                              {preview.columns.map((c) => (
                                <td key={c}>{row[c] === null ? <span className="null">∅</span> : String(row[c])}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}
