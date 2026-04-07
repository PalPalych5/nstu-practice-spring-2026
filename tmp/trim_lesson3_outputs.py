import json
from pathlib import Path

manual_path = Path('students/kudryavcev/lesson3/lesson3_manual.ipynb')
auto_path = Path('students/kudryavcev/lesson3/lesson3_auto.ipynb')

# Manual
manual_nb = json.loads(manual_path.read_text(encoding='utf-8-sig'))
manual_search = ''.join(manual_nb['cells'][6]['source'])
manual_search = manual_search.replace(
"""
plot_df = manual_results.head(10).copy()
plot_df['label'] = (
    plot_df['arch_name']
    + ' | lr=' + plot_df['lr'].astype(str)
    + ' | bs=' + plot_df['batch_size'].astype(str)
    + ' | ep=' + plot_df['epochs'].astype(str)
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(plot_df['label'], plot_df['mean_val_accuracy'], color='steelblue')
ax.set_title('Manual search: top validation accuracies')
ax.set_ylabel('Mean validation accuracy')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
""",
"""
"""
)
manual_nb['cells'][6]['source'] = [line + '\n' for line in manual_search.rstrip('\n').split('\n')]

manual_final = ''.join(manual_nb['cells'][7]['source'])
manual_final = manual_final.replace(
"""
manual_test_pred = predict_classes(final_manual_model, X_test_norm)
manual_test_accuracy = float(accuracy_score(y_test, manual_test_pred))
plot_confusion(
    y_test,
    manual_test_pred,
    title=f'Manual final model | acc={manual_test_accuracy:.4f}',
)
""",
"""
manual_test_pred = predict_classes(final_manual_model, X_test_norm)
manual_test_accuracy = float(accuracy_score(y_test, manual_test_pred))
"""
)
manual_nb['cells'][7]['source'] = [line + '\n' for line in manual_final.rstrip('\n').split('\n')]
manual_path.write_text(json.dumps(manual_nb, ensure_ascii=False, indent=2), encoding='utf-8')

# Auto
auto_nb = json.loads(auto_path.read_text(encoding='utf-8-sig'))
auto_imports = ''.join(auto_nb['cells'][2]['source'])
auto_imports = auto_imports.replace('from optuna.trial import FrozenTrial, TrialState\n', 'from optuna.trial import TrialState\n')
auto_nb['cells'][2]['source'] = [line + '\n' for line in auto_imports.rstrip('\n').split('\n')]

auto_search = ''.join(auto_nb['cells'][6]['source'])
auto_search = auto_search.replace(
"""
        units = int(cast(int, units_choice))
        activation = str(cast(str, activation_choice))
""",
"""
        units = int(units_choice)
        activation = str(activation_choice)
"""
)
auto_search = auto_search.replace(
"""
    batch_size = int(cast(int, trial.suggest_categorical('batch_size', [16, 32, 64, 128])))
""",
"""
    batch_size = int(trial.suggest_categorical('batch_size', [16, 32, 64, 128]))
"""
)
auto_search = auto_search.replace(
"""
plot_trials_df = trials_df[trials_df['value'].notna()].copy()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(plot_trials_df['number'], plot_trials_df['value'], marker='o', linewidth=1.5)
ax.set_title('Optuna trials by validation accuracy')
ax.set_xlabel('Trial number')
ax.set_ylabel('Validation accuracy')
plt.tight_layout()
plt.show()

""",
"""
"""
)
auto_search = auto_search.replace(
"""
complete_trials = [
    cast(FrozenTrial, trial)
    for trial in study.trials
    if trial.state == TrialState.COMPLETE and trial.value is not None
]
""",
"""
complete_trials = [
    trial
    for trial in study.trials
    if trial.state == TrialState.COMPLETE and trial.value is not None
]
"""
)
auto_search = auto_search.replace(
"""
best_trial = max(complete_trials, key=lambda trial: float(cast(float, trial.value)))
""",
"""
best_trial = max(complete_trials, key=lambda trial: float(trial.value))
"""
)
auto_nb['cells'][6]['source'] = [line + '\n' for line in auto_search.rstrip('\n').split('\n')]

auto_final = ''.join(auto_nb['cells'][7]['source'])
auto_final = auto_final.replace(
"""
auto_test_pred = predict_classes(final_auto_model, X_test_norm)
auto_test_accuracy = float(accuracy_score(y_test, auto_test_pred))
auto_arch_name = format_architecture_name(best_hidden_layers)
plot_confusion(
    y_test,
    auto_test_pred,
    title=f'Auto final model | acc={auto_test_accuracy:.4f}',
)
""",
"""
auto_test_pred = predict_classes(final_auto_model, X_test_norm)
auto_test_accuracy = float(accuracy_score(y_test, auto_test_pred))
auto_arch_name = format_architecture_name(best_hidden_layers)
"""
)
auto_nb['cells'][7]['source'] = [line + '\n' for line in auto_final.rstrip('\n').split('\n')]

auto_path.write_text(json.dumps(auto_nb, ensure_ascii=False, indent=2), encoding='utf-8')
