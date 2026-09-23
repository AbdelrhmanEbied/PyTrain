const OP_ICONS = {
  drop_columns: "M3 6h18M8 6V4h8v2m-9 0 1 14h8l1-14",
  rename_columns: "M4 7h16M4 12h10M4 17h7",
  drop_duplicates: "M8 8h12v12H8zM4 4h12v4M4 4v12h4",
  drop_missing_rows: "M4 6h16M7 10h10l-1 8H8zM12 4v4",
  fill_missing: "M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z",
  convert_dtype: "M4 7h7l-3-3m3 3-3 3M20 17h-7l3-3m-3 3 3 3",
  parse_dates: "M4 6h16v14H4zM4 10h16M8 3v4M16 3v4",
  clean_numeric: "M5 12l4 4L19 6M4 20h16",
  filter_rows: "M4 5h16l-6 7v6l-4 2v-8z",
  shuffle: "M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5",
  sample_rows: "M4 6h16M4 12h10M4 18h7",
  remove_outliers: "M6 6l12 12M18 6 6 18M12 3v2M12 19v2M3 12h2M19 12h2",
  clip_outliers: "M4 8h16v8H4zM8 8V6m8 2V6M8 18v-2m8 2v-2",
  create_date_features: "M12 3v18M3 12h18M6 6l12 12M18 6 6 18",
  transform_numeric: "M4 18c4 0 4-12 8-12s4 12 8 12",
  load_dataset:
    "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3m0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  prepare_features: "M4 7h10M4 12h16M4 17h8M18 5l2 2-2 2M16 15l2 2-2 2",
  split_data: "M6 4v6a4 4 0 0 0 4 4h8M18 4v6a4 4 0 0 1-4 4H6M6 20v-4a4 4 0 0 1 4-4h8M14 14l4 4-4 4",
  train_model: "M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3zm0 0v18M4 7.5l8 4.5 8-4.5",
  evaluate: "M4 19V5m0 14h16M8 15l3-4 3 2 4-6",
  log_mlflow: "M5 19V9m4 10V5m4 14v-7m4 7V8m4 11V3",
  finish: "M5 12l5 5L20 7",
  queued: "M12 7v5l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
  loading: "M4 7h16M4 12h10M4 17h7",
  preprocessing: "M4 7h10M4 12h16M4 17h8",
  preparing: "M4 7h10M4 12h16M4 17h8",
  training: "M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z",
  evaluating: "M4 19V5m0 14h16M8 15l3-4 3 2 4-6",
  logging: "M5 19V9m5 10V5m5 14v-8m5 8V7",
  done: "M5 12l5 5L20 7",
  error: "M12 8v5m0 3h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  running: "M12 3v3m0 12v3M3 12h3m12 0h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1",
};

function formatArgs(args) {
  const entries = Object.entries(args || {}).filter(([, v]) => v !== "" && v != null);
  if (!entries.length) return null;
  return entries.map(([k, v]) => {
    const val = typeof v === "object" ? JSON.stringify(v) : String(v);
    return (
      <span className="g-arg" key={k}>
        <b>{k}</b>
        {val.length > 48 ? `${val.slice(0, 48)}…` : val}
      </span>
    );
  });
}

function deltaBadge(prev, cur) {
  if (!prev || !cur) return null;
  const dRows = cur.rows - prev.rows;
  const dCols = cur.cols - prev.cols;
  if (dRows === 0 && dCols === 0) return <span className="g-delta same">unchanged</span>;
  const parts = [];
  if (dRows !== 0) parts.push(`${dRows > 0 ? "+" : ""}${dRows} rows`);
  if (dCols !== 0) parts.push(`${dCols > 0 ? "+" : ""}${dCols} cols`);
  const cls = dRows < 0 || dCols < 0 ? "cut" : "grow";
  return <span className={`g-delta ${cls}`}>{parts.join(" · ")}</span>;
}

function StatusBadge({ status }) {
  if (!status) return null;
  const label = status === "pending" ? "pending" : status;
  return <span className={`g-status ${status}`}>{label}</span>;
}

function Icon({ op }) {
  const d = OP_ICONS[op] || "M4 12h16M12 4v16";
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

export default function PipelineGraph({
  steps = [],
  startLabel = "Dataset",
  endLabel = "Result",
  startShape = null,
  selected = -1,
  onSelect = null,
  onRemove = null,
  showShapes = false,
  startStatus = null,
  endStatus = null,
}) {
  let prevShape = startShape;
  const nodes = [];

  nodes.push(
    <div
      key="start"
      className={`gnode source ${selected === -1 ? "selected" : ""} ${startStatus || ""}`}
      onClick={() => onSelect && onSelect(-1)}
    >
      <div className="gnode-top">
        <span className="gnode-icon source-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
            <ellipse cx="12" cy="6" rx="7" ry="3" />
            <path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6" />
            <path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
          </svg>
        </span>
        <span className="gnode-name">{startLabel}</span>
        {startShape ? (
          <span className="gnode-shape">
            {startShape[0]} × {startShape[1]}
          </span>
        ) : null}
        <StatusBadge status={startStatus} />
      </div>
    </div>
  );

  steps.forEach((step, i) => {
    const shape = step.rows != null ? { rows: step.rows, cols: step.cols } : null;
    const status = step.status || null;
    const linkActive = status === "active";
    nodes.push(
      <div key={`link-${i}`} className={`glink ${linkActive ? "active" : ""}`}>
        <span className="glink-line" />
        <span className="glink-dot" />
      </div>
    );
    const argsHtml = formatArgs(step.args || (step.isStage ? {} : step));
    const name = step.label || step.operation;
    let body = null;
    if (step.rows != null) {
      body = (
        <div className="gnode-bottom">
          <span className="gnode-shape strong">
            {step.rows} × {step.cols}
          </span>
          {showShapes ? deltaBadge(prevShape, shape) : null}
          <StatusBadge status={status} />
        </div>
      );
    } else if (status || step.detail) {
      body = (
        <div className="gnode-bottom">
          <StatusBadge status={status} />
          {step.detail ? <span className="gnode-detail">{step.detail}</span> : null}
        </div>
      );
    }
    nodes.push(
      <div
        key={`step-${i}`}
        className={`gnode op ${selected === i ? "selected" : ""} ${status || ""}`}
        onClick={() => onSelect && onSelect(i)}
      >
        <div className="gnode-top">
          <span className="gnode-icon">
            <Icon op={step.operation} />
          </span>
          <span className="gnode-name">{name}</span>
          {!step.isStage ? <span className="gnode-idx">{i + 1}</span> : null}
          {onRemove && !step.isStage ? (
            <button
              type="button"
              className="gnode-remove"
              title="Remove"
              onClick={(e) => {
                e.stopPropagation();
                onRemove(i);
              }}
            >
              ×
            </button>
          ) : null}
        </div>
        {argsHtml ? <div className="gnode-args">{argsHtml}</div> : null}
        {body}
      </div>
    );
    if (step.rows != null) prevShape = shape;
  });

  if (steps.length) {
    const lastActive = steps.some((s) => s.status === "active") && (!endStatus || endStatus === "active");
    nodes.push(
      <div key="endlink" className={`glink ${lastActive ? "active" : ""}`}>
        <span className="glink-line" />
        <span className="glink-dot" />
      </div>
    );
  }
  const lastDone = steps.length && steps.every((s) => !s.status || s.status === "done");
  const autoEnd = endStatus || (steps.length ? (lastDone ? "done" : null) : null);
  const endShape =
    steps.length && steps[steps.length - 1].rows != null
      ? { rows: steps[steps.length - 1].rows, cols: steps[steps.length - 1].cols }
      : null;

  nodes.push(
    <div
      key="end"
      className={`gnode result ${selected === -2 ? "selected" : ""} ${autoEnd || ""}`}
      onClick={() => onSelect && onSelect(-2)}
    >
      <div className="gnode-top">
        <span className="gnode-icon result-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
            <path d="M5 12l5 5L20 7" />
          </svg>
        </span>
        <span className="gnode-name">{endLabel}</span>
        {endShape ? (
          <span className="gnode-shape strong">
            {endShape.rows} × {endShape.cols}
          </span>
        ) : null}
        <StatusBadge status={autoEnd} />
      </div>
    </div>
  );

  return <div className="gflow">{nodes.map((n) => n)}</div>;
}
