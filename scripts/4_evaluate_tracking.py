import os
import numpy as np
import pandas as pd
import argparse

# Compatibilidad con motmetrics en entornos con NumPy 2.0+
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a: np.asarray(a, dtype=float)

import motmetrics as mm


def evaluar_tracker_individual(gt_source_dir: str, tracker_hypotheses_dir: str) -> pd.DataFrame:
    """
    Evalúa las trayectorias de un tracker frente al Ground Truth y devuelve
    una tabla con métricas por secuencia y una fila agregada global.
    """
    metrics_host = mm.metrics.create()
    metric_accumulators = []
    sequence_names = []

    if not os.path.exists(gt_source_dir):
        print(f"[!] ERROR CRÍTICO: Directorio Ground Truth inaccesible: {gt_source_dir}")
        return None

    valid_sequences = sorted([
        d for d in os.listdir(gt_source_dir)
        if os.path.isdir(os.path.join(gt_source_dir, d))
    ])

    for sequence_name in valid_sequences:
        gt_file_path = os.path.join(gt_source_dir, sequence_name, "gt", "gt.txt")
        hypothesis_file_path = os.path.join(tracker_hypotheses_dir, f"{sequence_name}.txt")

        if not os.path.exists(gt_file_path):
            continue

        # Si falta la salida del tracker para una secuencia, se omite.
        if not os.path.exists(hypothesis_file_path):
            print(f"[!] Advertencia: Ausencia de hipótesis para {sequence_name} en {tracker_hypotheses_dir}")
            continue

        ground_truth_data = mm.io.loadtxt(gt_file_path, fmt='mot15-2D', min_confidence=1)
        hypothesis_data = mm.io.loadtxt(hypothesis_file_path, fmt='mot15-2D')

        # Asociación espacial basada en IoU con umbral del 50%.
        sequence_accumulator = mm.utils.compare_to_groundtruth(
            ground_truth_data,
            hypothesis_data,
            'iou',
            distth=0.5
        )

        metric_accumulators.append(sequence_accumulator)
        sequence_names.append(sequence_name)

    if not metric_accumulators:
        return None

    tracker_summary_df = metrics_host.compute_many(
        metric_accumulators,
        metrics=[
            'idf1',
            'mota',
            'motp',
            'num_false_positives',
            'num_misses',
            'num_switches',
            'mostly_tracked',
            'mostly_lost'
        ],
        names=sequence_names,
        generate_overall=True
    )

    return tracker_summary_df


def evaluar_benchmark_completo(gt_source_dir: str, benchmark_results_dir: str) -> None:
    """
    Evalúa todos los trackers presentes en un benchmark y construye
    una tabla comparativa final con sus métricas globales.
    """
    print(f"[*] Inicializando auditoría de métricas MOT en: {benchmark_results_dir}")

    if not os.path.exists(benchmark_results_dir):
        print(f"[!] ERROR: Directorio de inferencias no localizado: {benchmark_results_dir}")
        return

    tracker_algorithms = sorted([
        d for d in os.listdir(benchmark_results_dir)
        if os.path.isdir(os.path.join(benchmark_results_dir, d))
    ])

    if not tracker_algorithms:
        print("[!] No se detectaron algoritmos a evaluar.")
        return

    all_trackers_overall_metrics = []

    for tracker_name in tracker_algorithms:
        print(f"\n📊 Procesando métricas para: {tracker_name.upper()}...")
        current_tracker_dir = os.path.join(benchmark_results_dir, tracker_name)

        tracker_metrics_df = evaluar_tracker_individual(gt_source_dir, current_tracker_dir)

        if tracker_metrics_df is not None:
            tracker_overall_row = tracker_metrics_df.loc['OVERALL'].copy()
            tracker_overall_row['Tracker'] = tracker_name.upper()
            all_trackers_overall_metrics.append(tracker_overall_row)
        else:
            print(f"  [!] Fallo en el cómputo de métricas para {tracker_name}")

    if not all_trackers_overall_metrics:
        print("Auditoría abortada: Ausencia de datos válidos.")
        return

    final_comparison_df = pd.DataFrame(all_trackers_overall_metrics)

    reordered_columns = ['Tracker'] + [c for c in final_comparison_df.columns if c != 'Tracker']
    final_comparison_df = final_comparison_df[reordered_columns]

    final_comparison_df.rename(columns={
        'idf1': 'IDF1',
        'mota': 'MOTA',
        'motp': 'MOTP',
        'num_false_positives': 'FP',
        'num_misses': 'FN',
        'num_switches': 'IDsw',
        'mostly_tracked': 'MT',
        'mostly_lost': 'ML'
    }, inplace=True)

    print("\n" + "=" * 85)
    print("🏆 RESULTADOS FINALES DEL BENCHMARK (Métricas MOT)")
    print("=" * 85)

    display_formatters = {
        'IDF1': '{:.1%}'.format,
        'MOTA': '{:.1%}'.format,
        'MOTP': '{:.3f}'.format
    }

    print(final_comparison_df.to_string(index=False, formatters=display_formatters))
    print("=" * 85)

    final_csv_dest_path = os.path.join(benchmark_results_dir, "resultados_finales_tfg.csv")
    final_comparison_df.to_csv(final_csv_dest_path, index=False)

    print(f"\n✅ Matriz de resultados exportada con éxito a: {final_csv_dest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auditor Masivo de Algoritmos MOT (Multiple Object Tracking)"
    )

    parser.add_argument(
        "--gt",
        type=str,
        default="datasets/raw/BEE24/test",
        help="Directorio origen del Ground Truth"
    )

    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default="runs/benchmark_results",
        help="Directorio con las inferencias a evaluar"
    )

    args = parser.parse_args()

    evaluar_benchmark_completo(
        gt_source_dir=args.gt,
        benchmark_results_dir=args.benchmark_dir
    )