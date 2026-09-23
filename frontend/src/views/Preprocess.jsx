import { useState } from "react";
import { DatasetRefForm, Status, EmptyState } from "../components/forms.jsx";
import { postJson } from "../lib/api.js";
import { DEFAULT_ARGS, OP_GROUPS } from "../lib/preprocessOps.js";
import PipelineGraph from "../components/PipelineGraph.jsx";

export default function PreprocessView({ datasets }) {
  const [ref, setRef] = useState({
    source: "sklearn",
    identifier: "iris",
    split: "",
    dataset: "",
    target: "",
  });
  const [operations, setOperations] = useState([]);
  const [selected, setSelected] = useState(-1);
  const [opType, setOpType] = useState("shuffle");
  const [opArgs, setOpArgs] = useState(DEFAULT_ARGS.shuffle);
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  function addOp() {
    let args;
    try {
      args = JSON.parse(opArgs.trim() || "{}");
    } catch {
      setStatus({ msg: "Arguments must be valid JSON.", cls: "error" });
      return;
    }
    const next = [...operations, { operation: opType, ...args }];
    setOperations(next);
    setSelected(next.length - 1);
    setStatus({ msg: "", cls: "" });
  }

  function removeOp(i) {
    setOperations(operations.filter((_, idx) => idx !== i));
    setSelected(-1);
  }

  function selectOp(i) {
    if (i < 0) return;
    setSelected(i);
    const op = operations[i];
    setOpType(op.operation);
    const args = Object.fromEntries(Object.entries(op).filter(([k]) => k !== "operation"));
    setOpArgs(JSON.stringify(args, null, 2));
  }

  function plannedSteps() {
    return operations.map((op, i) => ({
      operation: op.operation,
      label: op.operation,
      args: Object.fromEntries(Object.entries(op).filter(([k]) => k !== "operation")),
      status: selected === i ? "active" : null,
      isStage: false,
    }));
  }

  async function runPipeline() {
    setBusy(true);
    setStatus({ msg: "Running…", cls: "" });
    try {
      if (!ref.identifier) throw new Error("Identifier is required");
      const body = {
        source: ref.source,
        identifier: ref.identifier,
        split: ref.split || null,
        dataset: ref.dataset || null,
        target: ref.target || null,
        operations,
        n_rows: 20,
      };
      const data = await postJson("/api/preprocess", body);
      setResult(data);
      setStatus({ msg: "Done.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Pipeline builder</h2>
        <p className="muted small">
          Preview only — this tab does not change your data. Copy the same operations into the
          Train tab (or a config file) to apply them when you train.
        </p>
        <DatasetRefForm prefix="pp" datasets={datasets} value={ref} onChange={setRef} />

        <div className="section-label">Add operation</div>
        <div className="op-builder">
          <div>
            <label htmlFor="pp-op">Operation</label>
            <select
              id="pp-op"
              value={opType}
              onChange={(e) => {
                setOpType(e.target.value);
                setOpArgs(DEFAULT_ARGS[e.target.value] || "{}");
              }}
            >
              {OP_GROUPS.map((g) => (
                <optgroup key={g.label} label={g.label}>
                  {g.ops.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="pp-args">Arguments (JSON)</label>
            <textarea
              id="pp-args"
              spellCheck="false"
              className="short"
              value={opArgs}
              onChange={(e) => setOpArgs(e.target.value)}
            />
          </div>
          <button type="button" className="small" onClick={addOp}>
            Add
          </button>
        </div>

        <div className="section-label">Pipeline graph</div>
        <PipelineGraph
          steps={plannedSteps()}
          startLabel="Dataset"
          endLabel="Output"
          selected={selected}
          onSelect={selectOp}
          onRemove={removeOp}
        />

        <div className="actions">
          <button type="button" className="small" onClick={runPipeline} disabled={busy}>
            {busy ? "Running…" : "Run pipeline"}
          </button>
          <button
            type="button"
            className="small secondary"
            onClick={() => {
              setOperations([]);
              setSelected(-1);
            }}
          >
            Clear
          </button>
        </div>
        <Status msg={status.msg} cls={status.cls} />
      </section>

      <section className="panel">
        <h2>Result</h2>
        {!result ? (
          <EmptyState
            title="No pipeline result"
            body="Add operations and run the pipeline to see the transformed data."
          />
        ) : (
          <PreprocessResult data={result} />
        )}
      </section>
    </div>
  );
}

function PreprocessResult({ data }) {
  const dRows = data.after_shape[0] - data.before_shape[0];
  const dCols = data.after_shape[1] - data.before_shape[1];
  const stats = data.columns_stats || [];
  return (
    <div>
      <div className="meta">
        <span className="chip">
          <strong>{data.dataset}</strong>
        </span>
        <span className="chip">
          before{" "}
          <strong>
            {data.before_shape[0]}×{data.before_shape[1]}
          </strong>
        </span>
        <span className="chip">
          after{" "}
          <strong>
            {data.after_shape[0]}×{data.after_shape[1]}
          </strong>
        </span>
        {dRows || dCols ? (
          <span className="chip accent">
            Δ {dRows >= 0 ? "+" : ""}
            {dRows}r {dCols >= 0 ? "+" : ""}
            {dCols}c
          </span>
        ) : null}
        <span className="chip">
          ops <strong>{data.operations.length}</strong>
        </span>
        {data.duplicates ? (
          <span className="chip">
            dup rows <strong>{data.duplicates}</strong>
          </span>
        ) : null}
        {data.target ? (
          <span className="chip accent">
            target <strong>{data.target}</strong>
          </span>
        ) : null}
      </div>

      <div className="section-label">Executed pipeline</div>
      <PipelineGraph
        steps={(data.steps || []).map((s, i) => ({
          ...s,
          label: s.operation,
          status: "done",
          isStage: false,
          idx: i,
        }))}
        startLabel="Input"
        endLabel="Output"
        startShape={data.before_shape}
        selected={-1}
        showShapes
        endStatus="done"
      />

      {stats.length ? (
        <>
          <div className="section-label">Output columns</div>
          <div className="table-wrap">
            <table className="stats-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Dtype</th>
                  <th>Missing</th>
                  <th>Unique</th>
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
                    <td>
                      {s.missing}
                      {s.missing_pct ? ` (${s.missing_pct}%)` : ""}
                    </td>
                    <td>{s.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      <div className="section-label">Operations log</div>
      <pre>{JSON.stringify(data.operations, null, 2)}</pre>

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
