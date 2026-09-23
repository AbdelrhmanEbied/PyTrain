import { useEffect, useState } from "react";
import { api } from "./lib/api.js";
import TrainView from "./views/Train.jsx";
import DatasetsView from "./views/Datasets.jsx";
import PreprocessView from "./views/Preprocess.jsx";
import RunsView from "./views/Runs.jsx";
import MonitoringView from "./views/Monitoring.jsx";
import DataVersionsView from "./views/DataVersions.jsx";
import PredictView from "./views/Predict.jsx";

const TABS = [
  { id: "train", label: "Train" },
  { id: "preprocess", label: "Preprocess" },
  { id: "datasets", label: "Datasets" },
  { id: "versions", label: "Versions" },
  { id: "runs", label: "Runs" },
  { id: "predict", label: "Predict" },
  { id: "monitoring", label: "Monitoring" },
];

export default function App() {
  const [tab, setTab] = useState("train");
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [bootError, setBootError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    Promise.all([api("/api/datasets"), api("/api/models")])
      .then(([d, m]) => {
        setDatasets(d);
        setModels(m);
        setReady(true);
      })
      .catch((err) => setBootError(err.message));
  }, []);

  if (bootError) {
    return (
      <div className="shell">
        <div className="boot-status error">{bootError}</div>
      </div>
    );
  }

  if (!ready) {
    return (
      <div className="shell">
        <div className="boot-status">
          <span className="spinner" /> Loading…
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <header>
        <div className="brand">
          <div className="brand-mark">λ</div>
          <div>
            <h1>PyTrain</h1>
            <span className="tagline">load · preprocess · train · evaluate</span>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tab === t.id ? "active" : ""}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        <div className={`view ${tab === "train" ? "active" : ""}`}>
          {tab === "train" && (
            <TrainView datasets={datasets} models={models} onOpenPredict={() => setTab("predict")} />
          )}
        </div>
        <div className={`view ${tab === "preprocess" ? "active" : ""}`}>
          {tab === "preprocess" && <PreprocessView datasets={datasets} />}
        </div>
        <div className={`view ${tab === "datasets" ? "active" : ""}`}>
          {tab === "datasets" && <DatasetsView datasets={datasets} setDatasets={setDatasets} />}
        </div>
        <div className={`view ${tab === "versions" ? "active" : ""}`}>
          {tab === "versions" && <DataVersionsView datasets={datasets} setDatasets={setDatasets} />}
        </div>
        <div className={`view ${tab === "runs" ? "active" : ""}`}>
          {tab === "runs" && <RunsView />}
        </div>
        <div className={`view ${tab === "predict" ? "active" : ""}`}>
          {tab === "predict" && <PredictView />}
        </div>
        <div className={`view ${tab === "monitoring" ? "active" : ""}`}>
          {tab === "monitoring" && <MonitoringView />}
        </div>
      </main>
    </div>
  );
}
