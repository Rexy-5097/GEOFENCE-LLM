from run_windowed_validation import evaluate_windows, evaluate_length_split, BASELINE_PROMPTS, OBFUSCATED_PROMPTS

base_feat = "data/phase5_5_windowed_baseline_feat.jsonl"
res_base = evaluate_windows(base_feat, BASELINE_PROMPTS, "Baseline (Windowed)")
evaluate_length_split(res_base, BASELINE_PROMPTS)

obf_feat = "data/phase5_5_windowed_obf_feat.jsonl"
evaluate_windows(obf_feat, OBFUSCATED_PROMPTS, "Obfuscated (Windowed)")
