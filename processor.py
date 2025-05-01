from typing import Iterator, Dict, Any, List, Tuple, Optional
import multiprocessing as mp
from tqdm import tqdm
from spacy.language import Language
from spacy.tokens import Doc
from quickumls.core import QuickUMLS
from utils import ProcessingConfig
import os
import logging


class NLPProcessor:
	"""Handles NLP processing of medical texts using spaCy and medspaCy."""
	
	def __init__(self, nlp: Language, config: ProcessingConfig) -> None:
		"""
		Initialize the NLP processor.
		
		Args:
			nlp: Initialized spaCy/medspaCy pipeline
			config: Processing configuration
		"""
		self.nlp = nlp
		self.config = config
		self._setup_pipeline()
		self.chunk_size = self._calculate_chunk_size()
  
		logging.basicConfig(level=logging.INFO)
		
		# Log UMLS path info when initializing
		logging.info(f"Attempting to use UMLS path: {self.config.umls_dir}")
		if os.path.exists(self.config.umls_dir):
			logging.info(f"UMLS path exists: True")
			logging.info(f"UMLS path contents: {os.listdir(self.config.umls_dir)}")
		else:
			logging.info(f"UMLS path exists: False")

	def _calculate_chunk_size(self) -> int:
		"""Calculate optimal chunk size based on system resources."""
		cpu_count = mp.cpu_count()
		return max(100, cpu_count * 2)

	def _setup_pipeline(self) -> None:
		"""Configure the NLP pipeline with QuickUMLS."""
		self.matcher = QuickUMLS(quickumls_fp=self.config.umls_dir,
								overlapping_criteria="score",
								threshold=0.7)

	def process_document(self, doc_tuple: Tuple[str, Doc]) -> Dict[str, Any]:
		"""
		Process a single document and extract UMLS entities.
		
		Args:
			doc_tuple: Tuple of (document_id, spaCy Doc)
			
		Returns:
			Dictionary containing extracted information
		"""
		try:
			note_id, doc = doc_tuple
			
			result = {
				'row_id': note_id,
				'text': doc.text,
				'umls': []
			}
			
			# Use QuickUMLS matcher directly on the text
			matches = self.matcher.match(doc.text, best_match=True, ignore_syntax=False)
			
			for match in matches:
				for candidate in match:
					entity_info = {
						'start_char': candidate['start'],
						'end_char': candidate['end'],
						'raw': candidate['ngram'],
						'cui': candidate['cui'],
						'score': candidate['similarity'],
						'semtype': list(candidate['semtypes'])
					}
					result['umls'].append(entity_info)
					
			return result
		except Exception as e:
			# Log error and return partial result
			print(f"Error processing document {note_id}: {str(e)}")
			return {'row_id': note_id, 'error': str(e)}

	def process_documents(self, 
						note_ids: Iterator[str], 
						docs: Iterator[str]) -> Iterator[Dict[str, Any]]:
		"""
		Process all documents using either parallel or sequential processing.
		
		Args:
			note_ids: Iterator of document IDs
			docs: Iterator of document texts
			
		Returns:
			Iterator of processed results
		"""
		mode = os.environ.get('MODE', 'local').lower()
		
		if mode == 'fargate':
			# In Fargate mode, get total from the doc_pairs directly
			doc_pairs = list(zip(note_ids, docs))
			total_docs = len(doc_pairs)
		else:
			try:
				total_docs = sum(1 for _ in open(self.config.input_path)) - 1
			except (ValueError, FileNotFoundError) as e:
				# Fallback if file not found or other file-related issues
				import logging
				logging.warning(f"Could not count lines in file: {e}")
				# Convert iterators to lists for counting and batching
				doc_pairs = list(zip(note_ids, docs))
				total_docs = len(doc_pairs)
		
		if 'doc_pairs' not in locals():
			doc_pairs = list(zip(note_ids, docs))
		
		with tqdm(total=total_docs, desc="Processing documents") as pbar:
			if self.config.parallelize:
				num_workers = self.config.num_workers or mp.cpu_count()
				
				# Process in batches using multiple processes
				with mp.Pool(num_workers) as pool:
					for i in range(0, len(doc_pairs), self.chunk_size):
						batch = doc_pairs[i:i + self.chunk_size]
						results = pool.map(self.process_document,
										 zip([id_ for id_, _ in batch],
											 self.nlp.pipe([text for _, text in batch])))
						for result in results:
							yield result
						pbar.update(len(batch))
			else:
				# Process in batches sequentially, but use n_process for nlp.pipe
				num_workers = self.config.num_workers or os.cpu_count()
				for i in range(0, len(doc_pairs), self.chunk_size):
					batch = doc_pairs[i:i + self.chunk_size]
					docs = self.nlp.pipe([text for _, text in batch],
									   batch_size=self.config.batch_size,
									   n_process=num_workers) # Use multiple processes for tokenization
					for result in self.process_batch(list(zip([id_ for id_, _ in batch], docs)), pbar):
						yield result

	def process_batch(self, 
					 doc_tuples: List[Tuple[str, Doc]], 
					 pbar: Optional[tqdm] = None) -> Iterator[Dict[str, Any]]:
		"""
		Process a batch of documents.
		
		Args:
			doc_tuples: List of (document_id, spaCy Doc) tuples
			pbar: Optional progress bar
			
		Returns:
			Iterator of processed document results
		"""
		for doc_tuple in doc_tuples:
			result = self.process_document(doc_tuple)
			if pbar:
				pbar.update(1)
			yield result