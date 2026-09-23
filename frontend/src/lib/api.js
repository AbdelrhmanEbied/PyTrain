export async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const text =
      typeof detail === "string" ? detail : JSON.stringify(detail || res.statusText);
    throw new Error(text);
  }
  return data;
}

export function postJson(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function patchJson(path, body) {
  return api(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function deleteJson(path) {
  return api(path, { method: "DELETE" });
}

export function parseJsonText(text, fallback) {
  const raw = String(text ?? "").trim();
  if (!raw) return fallback;
  return JSON.parse(raw);
}

export function stageStepsFromJob(job) {
  const names = [...(job.pipeline || [])];
  if (!names.length && job.stage) names.push(job.stage);
  if (job.status === "error" && job.stage && !names.includes(job.stage)) names.push(job.stage);
  if (job.status === "done" && !names.includes("done")) names.push("done");
  const errIdx = job.status === "error" ? names.indexOf(job.stage) : -1;
  return names.map((name, i) => {
    let status = "pending";
    if (job.status === "done") status = "done";
    else if (errIdx >= 0) {
      if (i < errIdx) status = "done";
      else if (i === errIdx) status = "error";
    } else {
      const cur = names.indexOf(job.stage);
      if (cur >= 0 && i < cur) status = "done";
      else if (cur >= 0 && i === cur) status = "active";
      else if (cur < 0 && job.stage === name) status = "active";
    }
    return {
      operation: name,
      label: STAGE_LABELS[name] || name,
      detail: name === job.stage ? job.detail : null,
      status,
      isStage: true,
    };
  });
}

export const STAGE_LABELS = {
  queued: "Queued",
  loading: "Load data",
  preprocessing: "Preprocess",
  preparing: "Prepare features",
  training: "Train model",
  evaluating: "Evaluate",
  logging: "Log to MLflow",
  done: "Complete",
  error: "Failed",
  running: "Running",
};

export function postForm(path, formData) {
  return api(path, { method: "POST", body: formData });
}

export async function pollJob(jobId, onTick) {
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    if (onTick) onTick(job);
    if (job.status === "done" || job.status === "error") return job;
    await new Promise((r) => setTimeout(r, 350));
  }
}
