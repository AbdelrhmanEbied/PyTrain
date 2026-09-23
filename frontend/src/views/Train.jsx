import { useMemo, useState } from "react";
import { DatasetRefForm, MetricCard, ParamForm, Status, EmptyState } from "../components/forms.jsx";
import PipelineGraph from "../components/PipelineGraph.jsx";
import ConfigPanel from "./Config.jsx";
import { DEFAULT_ARGS, OP_GROUPS } from "../lib/preprocessOps.js";
import { parseJsonText, pollJob, postJson, STAGE_LABELS, stageStepsFromJob } from "../lib/api.js";

export default function TrainView({ datasets, models, onOpenPredict }) {
  const [ref, setRef] = useState({
    source: "sklearn",
    identifier: "iris",
    split: "",
    dataset: "",
    target: "",
  });
  const [target, setTarget] = useState("target");
  const [modelName, setModelName] = useState(models[0]?.model_name || "");
  const [paramValues, setParamValues] = useState({});
  const [paramMode, setParamMode] = useState("form");
  const [jsonParams, setJsonParams] = useState("{}");
  const [testSize, setTestSize] = useState("0.2");
  const [valSize, setValSize] = useState("0.1");
  const [useGrid, setUseGrid] = useState(false);
  const [gridJson, setGridJson] = useState('{"C": [0.1, 1.0]}');
  const [cv, setCv] = useState("3");
  const [scoring, setScoring] = useState("");
  const [logMlflow, setLogMlflow] = useState(false);
  const [operations, setOperations] = useState([]);
  const [selectedOp, setSelectedOp] = useState(-1);
  const [opType, setOpType] = useState("shuffle");
  const [opArgs, setOpArgs] = useState(DEFAULT_ARGS.shuffle);
  const [status, setStatus] = useState({ msg: "", cls: "" });
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  const model = useMemo(
    () => models.find((m) => m.model_name === modelName) || models[0],
    [models, modelName]
  );

  function onRefChange(next) {
    setRef(next);
    if (next.target) setTarget(next.target);
    if (next.dataset && !next.identifier) {
      const d = datasets.find((x) => x.name === next.dataset);
      if (d?.identifier) setRef({ ...next, identifier: d.identifier, source: d.source });
    }
  }

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
    setSelectedOp(next.length - 1);
    setStatus({ msg: "", cls: "" });
  }

  function removeOp(i) {
    const next = operations.filter((_, idx) => idx !== i);
    setOperations(next);
    setSelectedOp(-1);
  }

  function selectOp(i) {
    if (i < 0) return;
    setSelectedOp(i);
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
      status: selectedOp === i ? "active" : null,
      isStage: false,
    }));
  }

  function collectParams() {
    if (paramMode === "json") return parseJsonText(jsonParams, {});
    return { ...paramValues };
  }

  function applyConfigRequest(req) {
    if (!req) return;
    let source = req.source || "";
    let identifier = req.identifier || "";
    let split = req.split || "";
    let dataset = req.dataset || "";
    if (dataset && (!source || !identifier)) {
      const d = datasets.find((x) => x.name === dataset);
      if (d) {
        source = source || d.source;
        identifier = identifier || d.identifier;
        split = split || d.split || "";
      }
    }
    const nextRef = {
      source: source || ref.source || "sklearn",
      identifier: identifier || (dataset ? "" : ref.identifier) || "",
      split,
      dataset,
      target: req.target || ref.target || "",
    };
    setRef(nextRef);
    if (req.target) setTarget(req.target);
    if (req.model_name) setModelName(req.model_name);
    if (req.params) {
      setParamValues(req.params);
      setJsonParams(JSON.stringify(req.params, null, 2));
    }
    if (req.test_size != null) setTestSize(String(req.test_size));
    if (req.validation_size !== undefined) {
      setValSize(req.validation_size == null ? "" : String(req.validation_size));
    }
    if (req.param_config) {
      setUseGrid(true);
      if (req.param_config.grid) setGridJson(JSON.stringify(req.param_config.grid, null, 2));
      if (req.param_config.cv != null) setCv(String(req.param_config.cv));
      if (req.param_config.scoring != null) setScoring(String(req.param_config.scoring || ""));
    } else if (req.param_config === null) {
      setUseGrid(false);
    }
    if (req.log_mlflow != null) setLogMlflow(Boolean(req.log_mlflow));
    if (Array.isArray(req.operations)) setOperations(req.operations);
    setSelectedOp(-1);
    setStatus({ msg: "Config applied to form.", cls: "ok" });
  }

  async function startTraining(requestOverride) {
    setBusy(true);
    setResult(null);
    setJob(null);
    setStatus({ msg: "Starting job…", cls: "" });
    try {
      let param_config = null;
      if (useGrid) {
        const grid = parseJsonText(gridJson, null);
        if (!grid || !Object.keys(grid).length) throw new Error("Grid must be a non-empty JSON object");
        param_config = {
          grid,
          cv: Number(cv) || 3,
          scoring: scoring.trim() ? scoring.trim() : null,
        };
      }
      const body = requestOverride
        ? {
            source: requestOverride.source || ref.source,
            identifier: requestOverride.identifier || ref.identifier,
            split: requestOverride.split || ref.split || null,
            dataset: requestOverride.dataset || ref.dataset || null,
            target: requestOverride.target || target.trim() || null,
            model_name: requestOverride.model_name || modelName,
            task: requestOverride.task || model?.task || "classification",
            params: requestOverride.params || collectParams(),
            param_config:
              requestOverride.param_config !== undefined
                ? requestOverride.param_config
                : param_config,
            test_size: requestOverride.test_size ?? Number(testSize) ?? 0.2,
            validation_size:
              requestOverride.validation_size !== undefined
                ? requestOverride.validation_size
                : valSize === ""
                  ? null
                  : Number(valSize),
            stratify: requestOverride.stratify ?? true,
            random_state: requestOverride.random_state ?? 42,
            log_mlflow: requestOverride.log_mlflow ?? logMlflow,
            experiment_name: requestOverride.experiment_name || undefined,
            tracking_uri: requestOverride.tracking_uri || undefined,
            operations: requestOverride.operations || operations,
          }
        : {
            source: ref.source,
            identifier: ref.identifier,
            split: ref.split || null,
            dataset: ref.dataset || null,
            target: target.trim() || null,
            model_name: modelName,
            task: model?.task || "classification",
            params: collectParams(),
            param_config,
            test_size: Number(testSize) || 0.2,
            validation_size: valSize === "" ? null : Number(valSize),
            stratify: true,
            random_state: 42,
            log_mlflow: logMlflow,
            operations,
          };
      const started = await postJson("/api/jobs/train", body);
      const final = await pollJob(started.job_id, (j) => {
        setJob(j);
        const label = STAGE_LABELS[j.stage] || j.stage || "training";
        setStatus({ msg: `${label}…`, cls: "" });
      });
      setJob(final);
      if (final.status === "error") throw new Error(final.error || "Training failed");
      setResult(final.result);
      setStatus({ msg: "Done.", cls: "ok" });
    } catch (err) {
      setStatus({ msg: err.message, cls: "error" });
    } finally {
      setBusy(false);
    }
  }

  const stageGraph = job ? stageStepsFromJob(job) : [];

  return (
    <div className="grid-2">
      <section className="panel">
        <h2>Configuration</h2>
        <ConfigPanel onApply={applyConfigRequest} onRun={(req) => startTraining(req)} />
        <DatasetRefForm prefix="train" datasets={datasets} value={ref} onChange={onRefChange} />
        <label htmlFor="train-target">Target column</label>
        <input id="train-target" value={target} onChange={(e) => setTarget(e.target.value)} />
        <label htmlFor="train-model">Model</label>
        <select
          id="train-model"
          value={modelName}
          onChange={(e) => {
            setModelName(e.target.value);
            setParamValues({});
          }}
        >
          {models.map((m) => (
            <option key={m.model_name} value={m.model_name}>
              {m.model_name} ({m.framework}, {m.task})
            </option>
          ))}
        </select>

        <div className="grid-toggle">
          <input
            type="checkbox"
            id="train-params-mode"
            checked={paramMode === "json"}
            onChange={(e) => setParamMode(e.target.checked ? "json" : "form")}
          />
          <label htmlFor="train-params-mode" style={{ margin: 0, color: "var(--text)" }}>
            Params as JSON
          </label>
        </div>
        {paramMode === "form" ? (
          <ParamForm params={model?.parameters || []} values={paramValues} onChange={setParamValues} />
        ) : (
          <>
            <label htmlFor="train-params">Fixed params (JSON)</label>
            <textarea
              id="train-params"
              spellCheck="false"
              value={jsonParams}
              onChange={(e) => setJsonParams(e.target.value)}
            />
          </>
        )}

        <div className="section-label">Preprocessing pipeline</div>
        <div className="op-builder">
          <div>
            <label htmlFor="train-op">Operation</label>
            <select
              id="train-op"
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
            <label htmlFor="train-op-args">Arguments (JSON)</label>
            <textarea
              id="train-op-args"
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
        <div className="actions">
          <button type="button" className="small secondary" onClick={() => setOperations([])} disabled={!operations.length}>
            Clear ops
          </button>
          <span className="muted-mini">{operations.length} operation(s)</span>
        </div>
        <div id="train-ops-graph" style={{ marginTop: 10 }}>
          <PipelineGraph
            steps={plannedSteps()}
            startLabel="Dataset"
            endLabel="To training"
            selected={selectedOp}
            onSelect={selectOp}
            onRemove={removeOp}
          />
        </div>

        <div className="section-label">Data split</div>
        <div className="row">
          <div>
            <label htmlFor="train-test">Test size</label>
            <input
              id="train-test"
              type="number"
              step="0.05"
              min="0.05"
              max="0.5"
              value={testSize}
              onChange={(e) => setTestSize(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="train-val">Validation size</label>
            <input
              id="train-val"
              type="number"
              step="0.05"
              min="0"
              max="0.5"
              value={valSize}
              onChange={(e) => setValSize(e.target.value)}
            />
          </div>
        </div>

        <div className="grid-toggle">
          <input type="checkbox" id="train-use-grid" checked={useGrid} onChange={(e) => setUseGrid(e.target.checked)} />
          <label htmlFor="train-use-grid" style={{ margin: 0, color: "var(--text)" }}>
            Grid search CV
          </label>
        </div>
        {useGrid ? (
          <div>
            <label htmlFor="train-grid">Grid (JSON)</label>
            <textarea
              id="train-grid"
              spellCheck="false"
              value={gridJson}
              onChange={(e) => setGridJson(e.target.value)}
            />
            <div className="row">
              <div>
                <label htmlFor="train-cv">CV folds</label>
                <input id="train-cv" type="number" min="2" max="20" value={cv} onChange={(e) => setCv(e.target.value)} />
              </div>
              <div>
                <label htmlFor="train-scoring">Scoring</label>
                <input
                  id="train-scoring"
                  placeholder="auto"
                  value={scoring}
                  onChange={(e) => setScoring(e.target.value)}
                />
              </div>
            </div>
          </div>
        ) : null}

        <div className="check">
          <input type="checkbox" id="train-mlflow" checked={logMlflow} onChange={(e) => setLogMlflow(e.target.checked)} />
          <label htmlFor="train-mlflow" style={{ margin: 0, color: "var(--text)" }}>
            Log to MLflow
          </label>
        </div>

        <button type="button" id="train-btn" disabled={busy} onClick={() => startTraining()}>
          {busy ? "Training…" : "Train model"}
        </button>
        <Status msg={status.msg} cls={status.cls} />

        {job ? (
          <div className="stage-panel">
            <div className="section-label">Training pipeline</div>
            <PipelineGraph
              steps={stageGraph}
              startLabel="Job"
              endLabel={job.status === "error" ? "Failed" : "Complete"}
              selected={-1}
              endStatus={job.status === "error" ? "error" : job.status === "done" ? "done" : "pending"}
            />
            <div className="stage-detail">
              {job.status === "error"
                ? `Failed at ${STAGE_LABELS[job.stage] || job.stage}${job.detail ? ` — ${job.detail}` : ""}`
                : `${STAGE_LABELS[job.stage] || job.stage}${job.detail ? ` — ${job.detail}` : ""}`}
            </div>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>Results</h2>
        {!result ? (
          <EmptyState
            title="No training run yet"
            body="Configure a dataset, optional preprocess ops, and a model — then train to see metrics and the full pipeline graph."
          />
        ) : (
          <TrainResults data={result} onOpenPredict={onOpenPredict} />
        )}
      </section>
    </div>
  );
}

function TrainResults({ data, onOpenPredict }) {
  const test = data.test_metrics || {};
  const val = data.val_metrics;
  const pipe = data.pipeline || [];
  return (
    <div>
      <div className="meta">
        <span className="chip">
          <strong>{data.model_info.model_name}</strong>
        </span>
        <span className="chip">
          task <strong>{data.model_info.task}</strong>
        </span>
        <span className="chip">
          framework <strong>{data.model_info.framework}</strong>
        </span>
        <span className="chip">
          features <strong>{data.n_features}</strong>
        </span>
        <span className="chip">
          train <strong>{data.n_train}</strong>
        </span>
        <span className="chip">
          val <strong>{data.n_val}</strong>
        </span>
        <span className="chip">
          test <strong>{data.n_test}</strong>
        </span>
        <span className="chip">
          time <strong>{data.train_time.toFixed(4)}s</strong>
        </span>
        {data.logged_to_mlflow ? (
          <span className="chip accent">
            mlflow · {data.mlflow_run_id ? data.mlflow_run_id.slice(0, 8) : "logged"}
          </span>
        ) : null}
        {data.preprocess_ops?.length ? (
          <span className="chip accent">
            preprocess <strong>{data.preprocess_ops.length}</strong>
          </span>
        ) : null}
      </div>

      <div className="actions">
        {data.logged_to_mlflow && data.mlflow_run_id && onOpenPredict ? (
          <button type="button" className="small" onClick={onOpenPredict}>
            Try this model
          </button>
        ) : null}
        <span className="muted-mini">
          Models live in MLflow (mlruns + mlflow.db). See Runs → Storage paths.
        </span>
      </div>

      {pipe.length ? (
        <>
          <div className="section-label">Executed pipeline</div>
          <PipelineGraph
            steps={pipe}
            startLabel="Input"
            endLabel="Done"
            selected={-1}
            showShapes={pipe.some((s) => s.rows != null)}
            endStatus="done"
          />
        </>
      ) : null}

      <div className="section-label">Test metrics</div>
      <div className="metrics">
        {Object.entries(test).map(([k, v]) => (
          <MetricCard key={k} label={`test ${k}`} value={v} />
        ))}
      </div>
      {val ? (
        <>
          <div className="section-label">Validation metrics</div>
          <div className="metrics">
            {Object.entries(val).map(([k, v]) => (
              <MetricCard key={k} label={`val ${k}`} value={v} />
            ))}
          </div>
        </>
      ) : null}
      {Object.keys(data.train_metrics || {}).length ? (
        <>
          <div className="section-label">Train metrics</div>
          <div className="metrics">
            {Object.entries(data.train_metrics).map(([k, v]) => (
              <MetricCard key={k} label={`train ${k}`} value={v} />
            ))}
          </div>
        </>
      ) : null}

      <div className="section-label">Params</div>
      <pre>{JSON.stringify(data.params, null, 2)}</pre>
      {data.best_params ? (
        <>
          <div className="section-label">Best params (grid search)</div>
          <pre>{JSON.stringify(data.best_params, null, 2)}</pre>
        </>
      ) : null}
      {data.cv_results ? (
        <>
          <div className="section-label">CV results</div>
          <pre>{JSON.stringify(data.cv_results, null, 2)}</pre>
        </>
      ) : null}
    </div>
  );
}
