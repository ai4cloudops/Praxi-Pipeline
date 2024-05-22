from kubernetes import client, config
from datetime import datetime, timedelta

# Configure API access
config.load_kube_config()  # Use this for local access to K8s/OpenShift, e.g., via ~/.kube/config
# For in-cluster access (running inside an OpenShift cluster), use: config.load_incluster_config()

def calculate_pod_runtime(openshift_namespace='default'):
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(openshift_namespace)

    # Define the suffixes to match
    suffixes = [
        "generate-changesets-pod",
        "generate-tagset-pod",
        "gen-prediction-pod",
        "load-model-pod"
    ]

    # Initialize a dictionary to hold run times for each suffix
    duration_dict = {suffix: [] for suffix in suffixes}

    for pod in pods.items:
        for suffix in suffixes:
            if pod.metadata.name.endswith(suffix):
                # Ensure the pod has started and stopped
                if pod.status.start_time and pod.status.phase in ('Succeeded', 'Failed'):
                    start_time = pod.status.start_time
                    end_time = None
                    for container_status in pod.status.container_statuses:
                        if container_status.state.terminated:
                            end_time = container_status.state.terminated.finished_at
                            break
                    if end_time:
                        # Calculate duration
                        duration = end_time - start_time
                        # Append the duration to the appropriate list in the dictionary
                        duration_dict[suffix].append(duration)
                    else:
                        print(f"Pod: {pod.metadata.name}, Start: {start_time}, End: Unknown, Duration: Unknown")
                else:
                    print(f"Pod: {pod.metadata.name} has not completed execution or lacks timing information.")
                # Break since we found the suffix match and processed it
                break

    return duration_dict

if __name__ == "__main__":
    namespace = 'ai4cloudops-11855c'  # Replace with your namespace
    pod_runtimes = calculate_pod_runtime(namespace)
    for suffix, durations in pod_runtimes.items():
        print(f"Suffix: {suffix}, Durations: {[str(duration) for duration in durations]}")
