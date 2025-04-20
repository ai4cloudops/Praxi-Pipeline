# import requests, json
# prom_addr = "https://grafana.apps.obs.nerc.mghpcc.org/api/datasources/uid/a0fa9d88-8932-41f7-af80-6f678d4fb1e7/resources/api/v1/label/__name__/values"

# headers = {
#     'Authorization':  'Bearer ',
#     'Content-Type': 'application/x-www-form-urlencoded'
# }

# prom_metrics = requests.get(prom_addr, headers=headers).json()

# print(json.dumps(prom_metrics, indent=4))

import requests, json
grafana_addr = 'https://grafana.apps.obs.nerc.mghpcc.org/api/datasources/proxy/1/api/v1/query'
# grafana_addr = 'https://console.apps.shift.nerc.mghpcc.org/api/prometheus-tenancy/api/v1/'
headers = {
    'Authorization':  'Bearer ',
    'Content-Type': 'application/x-www-form-urlencoded'
}

# queries = {
#     "memory_usage": "namespace:container_memory_usage_bytes:sum{namespace=\"ai4cloudops-11855c\"}[3h:1m]",
#     "cpu_usage": "node_namespace_pod_container:container_cpu_usage_seconds_total:sum_rate{namespace=\"ai4cloudops-11855c\"}[3h:1m]"
# }

queries_cpu = {
    "start": 1738776801.901,
    "end": 1738949601.901,
    "step": 576,
    "namespace": 'ai4cloudops-11855c',
    "query": 'topk(25, sort_desc(sum(container_fs_usage_bytes{pod!="",namespace=\'ai4cloudops-11855c\'}) BY (pod, namespace)))',
}


cpu_usage = requests.get(grafana_addr, params=queries_cpu, headers=headers).json()
print(cpu_usage['data']['result'])



# https://console.apps.shift.nerc.mghpcc.org/api/prometheus-tenancy/api/v1/query_range?start=1738952955.063&end=1738954755.063&step=22.22222222222222&namespace=ai4cloudops-11855c&query=container_fs_usage_bytes%7Bpod%21%3D%22%22%7D&timeout=60s