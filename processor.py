from typing import Iterator, Dict, Any, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import json
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
		# Pre-calculate optimal chunk size based on available memory and CPU cores
		self.chunk_size = self._calculate_chunk_size()
		# Set up logging
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
		# Default to processing 2 documents per CPU core at a time
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
					# print('CUI: {}'.format(candidate['cui']))
					# print('Similarity: {}'.format(candidate['similarity']))
					# print('Semtypes: {}'.format(candidate['semtypes']))
					# print('********************')
					
			return result
		except Exception as e:
			# Log error and return partial result
			print(f"Error processing document {note_id}: {str(e)}")
			return {'row_id': note_id, 'error': str(e)}

	def _process_batch_parallel(self, 
							  batch: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
		"""Process a batch of documents in parallel using threads."""
		with ThreadPoolExecutor() as executor:
			docs = list(self.nlp.pipe([text for _, text in batch]))
			return list(executor.map(self.process_document, 
								   zip([id_ for id_, _ in batch], docs)))

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
		total_docs = sum(1 for _ in open(self.config.input_path)) - 1  # Subtract header
		
		# Convert iterators to lists for batching
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
				# Process in batches sequentially
				for i in range(0, len(doc_pairs), self.chunk_size):
					batch = doc_pairs[i:i + self.chunk_size]
					docs = self.nlp.pipe([text for _, text in batch],
									   batch_size=self.config.batch_size)
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