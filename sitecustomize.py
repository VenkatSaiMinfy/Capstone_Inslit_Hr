# ─────────────────────────────────────────────────────────────
# 🛠️ PATCH FOR MLFLOW YAML SERIALIZATION ISSUE
# ─────────────────────────────────────────────────────────────

# This script patches the YAML dumper in MLflow to prevent serialization
# errors when saving metrics containing custom Metric objects (common in MLflow <2.11).

import warnings
from mlflow.entities import Metric                                  # MLflow Metric object
from mlflow.utils.yaml_utils import YamlSafeDumper                  # MLflow's YAML dumper

# 🔕 Suppress all warnings globally
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 🧩 FUNCTION TO PATCH YAML SERIALIZER
# ─────────────────────────────────────────────────────────────
def configure_yaml_dumper():
    """
    Adds a custom representer for `mlflow.entities.Metric` objects
    to allow them to be dumped to YAML using PyYAML.
    This fixes a known issue when logging nested metric structures
    to artifacts (like run summaries) in YAML format.
    """
    try:
        YamlSafeDumper.add_multi_representer(
            Metric,
            lambda dumper, metric: dumper.represent_mapping(
                'tag:yaml.org,2002:map',
                {
                    'key': metric.key,
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'step': metric.step,
                }
            )
        )
        print("✅ Patched YAML dumper for mlflow.entities.Metric")
    except Exception as e:
        print("❌ Failed to patch YAML dumper:", e)

# 🛠️ Run immediately when this script is imported
configure_yaml_dumper()
