
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import polars as pl
import argparse
from pathlib import Path
import sys

# Filtering thresholds
TOTAL_DEPTH = 20
TOTAL_SUPPORT = 3
AVERAGE_UNC_RATIO = 0.02
AVERAGE_CLU_RATIO = 0.5
AVERAGE_MUL_RATIO = 0.2

def import_df_lazy(file_path: str, sample_id: str) -> pl.LazyFrame:
    count_cols = [
        "convertedBaseCount_unfiltered_uniq",
        "unconvertedBaseCount_unfiltered_uniq",
        "convertedBaseCount_unfiltered_multi",
        "unconvertedBaseCount_unfiltered_multi",
        "convertedBaseCount_filtered_uniq",
        "unconvertedBaseCount_filtered_uniq",
        "convertedBaseCount_filtered_multi",
        "unconvertedBaseCount_filtered_multi",
    ]
    df = pl.scan_ipc(file_path)
    rename_map = {col: f"{col}_{sample_id}" for col in count_cols}
    return df.rename(rename_map)

def combine_and_filter_lazy(file_paths: list[str]) -> pl.LazyFrame:
    samples = [Path(f).stem.split("_genome")[0] for f in file_paths]
    df_base = import_df_lazy(file_paths[0], samples[0])
    for file_path, sample_id in zip(file_paths[1:], samples[1:]):
        df_next = import_df_lazy(file_path, sample_id)
        df_base = df_base.join(df_next, on=["ref", "pos", "strand"], how="outer_coalesce")

    df_combined = (
        df_base
        .with_columns([
            pl.sum_horizontal([f"unconvertedBaseCount_filtered_uniq_{s}" for s in samples]).alias("u"),
            pl.sum_horizontal([
                f"{t}_filtered_uniq_{s}"
                for s in samples
                for t in ["convertedBaseCount", "unconvertedBaseCount"]
            ]).alias("d"),
            pl.sum_horizontal([
                f"{t1}_unfiltered_{t2}_{s}"
                for s in samples
                for t1 in ["convertedBaseCount", "unconvertedBaseCount"]
                for t2 in ["uniq", "multi"]
            ]).alias("_t")
        ])
        .with_columns([
            (pl.col("u") / pl.col("d").fill_null(1e-9)).alias("ur"),
            (
                pl.sum_horizontal([
                    f"{t}_unfiltered_multi_{s}"
                    for s in samples
                    for t in ["convertedBaseCount", "unconvertedBaseCount"]
                ]) / pl.col("_t").fill_null(1e-9)
            ).alias("mr"),
            (
                1 - pl.sum_horizontal([
                    f"{t1}_filtered_{t2}_{s}"
                    for s in samples
                    for t1 in ["convertedBaseCount", "unconvertedBaseCount"]
                    for t2 in ["uniq", "multi"]
                ]) / pl.col("_t").fill_null(1e-9)
            ).alias("cr"),
        ])
        .filter(
            (pl.col("d") >= TOTAL_DEPTH) &
            (pl.col("u") >= TOTAL_SUPPORT) &
            (pl.col("ur") >= AVERAGE_UNC_RATIO) &
            (pl.col("cr") < AVERAGE_CLU_RATIO) &
            (pl.col("mr") < AVERAGE_MUL_RATIO)
        )
        .select(["ref", "pos", "strand"])
        .unique(maintain_order=True)
    )

    return df_combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-files", nargs="+", required=True, help="Input genome.arrow files")
    parser.add_argument("-o", "--output-file", required=True, help="Output prefilter TSV file")
    args = parser.parse_args()

    df_final = combine_and_filter_lazy(args.input_files).collect()
    df_final.write_csv(args.output_file, separator="\t", include_header=False)
