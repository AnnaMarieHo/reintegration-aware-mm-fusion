import json
from collections import Counter

with open("partition.json") as f:
    partition = json.load(f)

label_names = {0:"neutral",1:"sadness",2:"joy",3:"anger",4:"disgust",5:"fear",6:"surprise"}
client_ids = [k for k in partition if k not in ("dev","test")]

print("Per-client label distribution (%):")
for cid in sorted(client_ids, key=int):
    distinct_utterance_counts = Counter()
    scenes = partition[cid]
    counts = Counter(utt[2] for scene in scenes for utt in scene)
    total = sum(counts.values())
    row = {label_names[i]: f"{counts.get(i,0)/total*100:.1f}" for i in range(7)}
    print(f"  C{cid}: {row}  scenes={len(scenes)}")

for client_id in sorted(client_ids, key=int):
    scenes = partition[client_id]
    scene_lengths = [len(scene) for scene in scenes]
    lengths = Counter(scene_lengths)
    min_len = min(scene_lengths)
    print(f"Client {client_id}: {scene_lengths}")
    print(f"Number of min scene length: {lengths[min_len]}")
    # print(f"  C{client_id}: {lengths}")
    print(f"  min_scene_length={min(scene_lengths)}")
    print(f"  max_scene_length={max(scene_lengths)}")
    print(f"  average_scene_length={sum(scene_lengths) / len(scene_lengths)}")




test_scenes = partition["test"]
test_scene_lengths = [len(scene) for scene in test_scenes]
test_lengths = Counter(test_scene_lengths)
test_min_len = min(test_scene_lengths)
print(f"Test: {test_scene_lengths}")
print(f"Number of min scene length: {test_lengths[test_min_len]}")
print(f"  min_scene_length={test_min_len}")
print(f"  max_scene_length={max(test_scene_lengths)}")
print(f"  average_scene_length={sum(test_scene_lengths) / len(test_scene_lengths)}")

# Label at each position across all scenes
position_labels = {}
for scene in test_scenes:
    for pos, utt in enumerate(scene):
        position_labels.setdefault(pos, []).append(utt[2])

print("Label distribution at each position (test):")
for pos in sorted(position_labels.keys())[:8]:
    counts = Counter(position_labels[pos])
    total = sum(counts.values())
    neutral_pct = counts.get(0, 0) / total * 100
    print(f"  t={pos}: neutral={neutral_pct:.1f}%  n={total}  counts={dict(counts)}")

# Scene diversity
all_neutral = sum(1 for s in test_scenes if all(u[2]==0 for u in s))
has_nonneutral_at_t5 = sum(1 for s in test_scenes if len(s) > 5 and s[5][2] != 0)
long_enough = sum(1 for s in test_scenes if len(s) > 5)

print(f"\nAll-neutral scenes: {all_neutral}/{len(test_scenes)}")
print(f"Scenes long enough for t=5: {long_enough}/{len(test_scenes)}")
print(f"Of those, non-neutral at t=5: {has_nonneutral_at_t5}/{long_enough} ({has_nonneutral_at_t5/long_enough*100:.1f}%)")

# Transition diversity
same, diff = 0, 0
for scene in test_scenes:
    for i in range(len(scene)-1):
        if scene[i][2] == scene[i+1][2]: same += 1
        else: diff += 1
print(f"\nWithin-scene transitions: same={same}, different={diff}, "
      f"change_rate={diff/(same+diff)*100:.1f}%")