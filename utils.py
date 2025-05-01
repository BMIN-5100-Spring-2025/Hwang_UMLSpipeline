from dataclasses import dataclass
from typing import Optional, Dict, Iterator, Union
from pathlib import Path
from itertools import islice
import csv
import sys
import os

@dataclass
class ProcessingConfig:
	"""Configuration settings for NLP processing."""
	input_file: Union[Path, str]
	output_file: Union[Path, str]
	umls_path: Union[Path, str]
	batch_size: int = 100
	num_workers: Optional[int] = None
	id_column: str = "note_id"
	text_column: str = "AGG_TEXT"
	parallelize: bool = False
	embeddings_path: Optional[str] = None
	visualization_path: Optional[str] = None
	sbert_model: str = "sapbert"  # choices: 'sapbert', 'minilm'
	fusion_strategy: str = "concat"  # choices: 'concat', 'linear'
	fallback_strategy: str = "text2vec"  # choices: 'text2vec', 'graph'
	mrrel_path: Optional[str] = None
	vectors_out: Optional[str] = None
	viz_dimension: int = 2
	cluster_method: Optional[str] = None
	n_clusters: Optional[int] = None
	hdb_min_cluster_size: Optional[int] = None
	hdb_min_samples: Optional[int] = None
	
	@property
	def input_path(self) -> Path:
		"""
		Convert input file to Path and validate existence.
		In Fargate mode, validation is more lenient to allow for placeholder paths.
		"""
		path = Path(self.input_file)
		# Check if we're running in Fargate mode (indicated by env var)
		fargate_mode = os.getenv('MODE', '').lower() == 'fargate'
		
		if not path.exists() and not fargate_mode:
			# Only raise error if not in Fargate mode
			raise ValueError(f"Input file does not exist: {path}")
		return path
	
	@property
	def output_path(self) -> Path:
		"""Convert output file to Path."""
		return Path(self.output_file)
	
	@property
	def umls_dir(self) -> Path:
		"""Convert UMLS path to Path and validate existence."""
		path = Path(self.umls_path)
		if not path.exists():
			raise ValueError(f"UMLS path does not exist: {path}")
		return path
		

class DataLoader:
	"""Handles loading and iteration over input data files."""
	
	def __init__(self, chunk_size: int = 1000) -> None:
		"""
		Initialize the DataLoader.
		
		Args:
			chunk_size: Number of records to load at once for efficient memory usage
		"""
		self.filename: Optional[Path] = None
		self.chunk_size = chunk_size
		csv.field_size_limit(sys.maxsize)

	def from_file(self, 
				 filename: Path, 
				 id_column: Optional[str] = None, 
				 text_column: Optional[str] = None) -> Iterator[str]:
		"""
		Load data from a file and yield contents based on file type.
		
		Args:
			filename: Path to the input file
			id_column: Name of the ID column for CSV files
			text_column: Name of the text column for CSV files
			
		Returns:
			Iterator yielding file contents
			
		Raises:
			ValueError: If file type is not supported
		"""
		self.filename = Path(filename)
		suffix = self.filename.suffix.lower()
		
		if suffix == '.txt':
			yield from self._read_txt_file()
		elif suffix == '.csv':
			yield from self._read_csv_file(id_column, text_column)
		else:
			raise ValueError(f"Unsupported file type: {suffix}")

	def _read_csv_file(self, 
					  id_column: Optional[str], 
					  text_column: Optional[str]) -> Iterator[str]:
		"""Read data from CSV file in chunks."""
		with open(self.filename, 'r', encoding='utf-8') as infile:
			reader = csv.DictReader(infile)
			while True:
				chunk = list(islice(reader, self.chunk_size))
				if not chunk:
					break
					
				if not id_column:
					for row in chunk:
						yield row[text_column]
				else:
					for row in chunk:
						yield row[id_column]

	def _read_txt_file(self) -> Iterator[str]:
		"""Read data from text file in chunks."""
		with open(self.filename, 'r', encoding='utf-8') as infile:
			while True:
				lines = list(islice(infile, self.chunk_size))
				if not lines:
					break
				for line in lines:
					yield line.strip()