import mlflow
import os

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Get the model
model_name = "AttritionProductionModel"
run_id = "ccfd88ff5ac2429f9e4f9778a0153363"  # This is the run_id from your error message

print("Attempting to get model versions...")
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions(model_name, stages=["Production"])
print(f"Found versions: {versions}")

print("\nAttempting to download artifacts...")
artifact_path = "drift_reference"
local_dir = "test_artifacts"
os.makedirs(local_dir, exist_ok=True)

# Try to list artifacts first
artifacts = client.list_artifacts(run_id, artifact_path)
print(f"\nAvailable artifacts: {artifacts}")

# Try to download one artifact
if artifacts:
    client.download_artifacts(run_id, artifact_path, local_dir)
    print(f"\nArtifacts downloaded to {local_dir}")