#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2024 Ye Chang yech1990@gmail.com
# Distributed under terms of the GNU license.
#
# Created: 2024-02-09 13:41


import polars as pl


def import_df(file_name, suffix):
    df = pl.read_csv(
        file_name,
        separator="\t",
        columns=["ref", "pos", "strand", "convertedBaseCount", "unconvertedBaseCount"],
        dtypes={
            "ref": pl.Utf8,
            "pos": pl.Int64,
            "strand": pl.Utf8,
            "convertedBaseCount": pl.Int64,
            "unconvertedBaseCount": pl.Int64,
        },
    )
    df = df.rename(
        {
            "convertedBaseCount": "convertedBaseCount_" + suffix,
            "unconvertedBaseCount": "unconvertedBaseCount_" + suffix,
        }
    )
    return df


from concurrent.futures import ThreadPoolExecutor

def combine_files(*files):
    suffixes = [
        "unfiltered_uniq",
        "unfiltered_multi",
        "filtered_uniq",
        "filtered_multi",
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        dfs = list(executor.map(import_df, files, suffixes))

    do_join = lambda left, right: left.join(right, on=["ref", "pos", "strand"], how="outer_coalesce")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(do_join, dfs[0], dfs[1]),
            executor.submit(do_join, dfs[2], dfs[3]),
        ]
        temp1, temp2 = [f.result() for f in futures]
    return do_join(temp1, temp2).fill_null(0)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i",
        "--input-files",
        nargs=4,
        required=True,
        help="4 input files: unfiltered_uniq, unfiltered_multi, filtered_uniq, filtered_multi",
    )
    arg_parser.add_argument("-o", "--output-file", help="output file")
    args = arg_parser.parse_args()

    # Write the combined DataFrame to a CSV file
    combine_files(*args.input_files).write_ipc(args.output_file, compression=None)
