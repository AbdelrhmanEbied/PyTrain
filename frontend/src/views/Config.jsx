import { useState } from "react";
import { Status } from "../components/forms.jsx";
import { postForm, postJson } from "../lib/api.js";

const SAMPLE = `dataset:
  dataset: titanic
  target: survived
training:
  model_name: logistic_regression
  task: classification
  params:
    max_iter: 500
  test_size: 0.2
  validation_size: 0.1
  log_mlflow: true
preprocessing:
  operations:
    - operation: drop_columns
      columns: [boat, body, home.dest]
    - operation: shuffle
      random_state: 42`;

export default function ConfigPanel({ onApply, onRun }) {
  const [open, setOpen] = useState(false);
  const [content, setContent] = useState(SAMPLE);
  const [format, setFormat] = useState("");
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [validateState, setValidateState] = useState({ msg: "", cls: "", data: null });
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState(null);

  async function validate(bodyFactory) {
    setBusy(true);
    setValidateState({ msg: "Validating…", cls: "", data: null });
    setStatus({ msg: "", cls: "" });
    try {
      const data = bodyFactory();
      setValidateState({ msg: "Config is valid.", cls: "ok", data });
      setStatus({ msg: "Validated. Apply it or train.", cls: "ok" });
      return data;
    } catch (err) {
      setValidateState({ msg: err.message, cls: "error", data: null });
      setStatus({ msg: err.message, cls: "error" });
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function onValidate() {
    return validate(() =>
      postJson("/api/config/validate", { content, format: format || null })
    );
  }

  async function onValidateFile() {
    if (!file) {
      setValidateState({ msg: "Choose a YAML/JSON file first.", cls: "error", data: null });
      return;
    }
    setBusy(true);
    setValidateState({ msg: "Validating file…", cls: "", data: null });
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await postForm("/api/config/validate-file", form);
      setValidateState({ msg: "Config is valid.", cls: "ok", data });
      setStatus({ msg: "File validated.", cls: "ok" });
      if (data.normalized_yaml) setContent(data.normalized_yaml);
      if (onApply) onApply(data.request);
      return data;
    } catch (err) {
      setValidateState({ msg: err.message, cls: "error", data: null });
      setStatus({ msg: err.message, cls: "error" });
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function onApplyClick() {
    const data = await onValidate();
    if (!data || !onApply) return;
    onApply(data.request);
    setStatus({ msg: "Applied to train form.", cls: "ok" });
  }

  async function onTrain() {
    const data = await onValidate();
    if (!data || !onApply) return;
    onApply(data.request);
    if (onRun) onRun(data.request);
  }

  function onFileChange(e) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      setContent(text);
      const name = f.name.toLowerCase();
      setFormat(name.endsWith(".json") ? "json" : name.endsWith(".yml") || name.endsWith(".yaml") ? "yaml" : "");
    };
    reader.readAsText(f);
  }

  if (!open) {
    return (
      <div className="config-collapsed">
        <button type="button" className="small secondary" onClick={() => setOpen(true)}>
          Load config (YAML / JSON)
        </button>
        <span className="muted-mini">Paste or upload a pipeline config to fill this form.</span>
      </div>
    );
  }

  return (
    <section className="config-panel">
      <div className="config-panel-head">
        <strong>Pipeline config</strong>
        <button type="button" className="small secondary" onClick={() => setOpen(false)}>
          Close
        </button>
      </div>

      <div className="row">
        <div>
          <label htmlFor="cfg-format">Format</label>
          <select id="cfg-format" value={format} onChange={(e) => setFormat(e.target.value)}>
            <option value="">auto-detect</option>
            <option value="yaml">yaml</option>
            <option value="json">json</option>
          </select>
        </div>
        <div>
          <label htmlFor="cfg-file">Upload file</label>
          <input
            id="cfg-file"
            type="file"
            accept=".yaml,.yml,.json"
            onChange={onFileChange}
          />
        </div>
      </div>

      <label htmlFor="cfg-content">Config content</label>
      <textarea
        id="cfg-content"
        spellCheck="false"
        style={{ minHeight: 220 }}
        value={content}
        onChange={(e) => setContent(e.target.value)}
      />

      <div className="actions">
        <button type="button" className="small secondary" onClick={onValidate} disabled={busy}>
          Validate
        </button>
        <button type="button" className="small secondary" onClick={onValidateFile} disabled={busy || !file}>
          Validate file
        </button>
        <button type="button" className="small" onClick={onApplyClick} disabled={busy}>
          Apply to form
        </button>
        <button type="button" className="small" onClick={onTrain} disabled={busy}>
          Validate &amp; train
        </button>
      </div>
      <Status msg={status.msg} cls={status.cls} />
      <div className={`status-line ${validateState.cls}`}>{validateState.msg}</div>

      {validateState.data ? (
        <div className="config-preview">
          <div className="meta">
            <span className="chip accent">valid</span>
            {validateState.data.request?.model_name ? (
              <span className="chip">
                model <strong>{validateState.data.request.model_name}</strong>
              </span>
            ) : null}
            {validateState.data.request?.dataset || validateState.data.request?.identifier ? (
              <span className="chip">
                data{" "}
                <strong>
                  {validateState.data.request.dataset || validateState.data.request.identifier}
                </strong>
              </span>
            ) : null}
          </div>
          <div className="section-label">Normalized request</div>
          <pre>{JSON.stringify(validateState.data.request, null, 2)}</pre>
        </div>
      ) : null}
    </section>
  );
}
