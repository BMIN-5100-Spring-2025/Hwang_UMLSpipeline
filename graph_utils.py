from __future__ import annotations

"""Utility to build adjacency list from UMLS MRREL.RRF."""

from pathlib import Path
from typing import Dict, Set
import csv
from collections import defaultdict
import logging

KEEP_RELS = {"PAR", "CHD", "SY", "RB", "RN"}


def load_umls_graph(mrrel_path: Path, keep_rels: Set[str] | None = None) -> Dict[str, Set[str]]:
    if keep_rels is None:
        keep_rels = KEEP_RELS
    graph: Dict[str, Set[str]] = defaultdict(set)

    logging.info(f"Building UMLS graph from {mrrel_path} ...")
    with open(mrrel_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            cui1, rel, cui2 = row[0], row[3], row[4]
            if rel not in keep_rels:
                continue
            graph[cui1].add(cui2)
            graph[cui2].add(cui1)
    logging.info(f"Graph built with {len(graph)} nodes")
    return graph 