# sitecustomize.py
import warnings
from mlflow.entities import Metric
from mlflow.utils.yaml_utils import YamlSafeDumper

warnings.filterwarnings("ignore")

def configure_yaml_dumper():
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

configure_yaml_dumper()
