import json, os, glob

charts_dir = 'processed_pez/charts'
json_files = glob.glob(os.path.join(charts_dir, '*.json'))

# Analyze event structure in RPE format
event_types = {}
event_samples = {}
total_events = 0

for f in json_files[:200]:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        
        if 'judgeLineList' not in data:
            continue
        
        for jl in data.get('judgeLineList', []):
            event_layers = jl.get('eventLayers', [])
            if not event_layers:
                continue
            
            for layer in event_layers:
                if not isinstance(layer, dict):
                    continue
                for key in layer:
                    if key.endswith('Events'):
                        events = layer[key]
                        if not isinstance(events, list):
                            continue
                        event_type = key
                        count = len(events)
                        total_events += count
                        
                        if event_type not in event_types:
                            event_types[event_type] = 0
                            event_samples[event_type] = None
                        event_types[event_type] += count
                        
                        if event_samples[event_type] is None and events:
                            event_samples[event_type] = events[0]
    except:
        continue

print("=== Event Type Distribution ===")
for k, v in sorted(event_types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\nTotal events: {total_events}")

print("\n=== Event Samples ===")
for k, v in event_samples.items():
    print(f"\n{k}:")
    print(f"  {json.dumps(v, indent=4, ensure_ascii=False)[:400]}")
