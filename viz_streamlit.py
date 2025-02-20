import streamlit as st
import tempfile
import os
import math
import json
import spacy
import logging
import pandas as pd

from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor
from embeddings import ConceptEmbedder
from visualizer import ConceptVisualizer

# Set up basic logging
logging.basicConfig(level=logging.INFO)

st.title("Medical Concept Visualization")

st.markdown("""
This app takes an input CSV file of clinical notes from a patient cohort, runs the QuickUMLS-based NLP pipeline to extract medical concepts,
and produces an interactive visualization of medical concept embeddings.
""")

# Sidebar configuration
st.sidebar.header("Pipeline Configuration")
umls_path = st.sidebar.text_input("UMLS Database Path", value="2020AB-full/2020AB-quickumls-install/")
embeddings_path = st.sidebar.text_input("Embeddings File Path", value="data/embeddings/cui2vec_pretrained.txt")
text_column = st.sidebar.text_input("Text Column Name", value="transcription")
id_column = st.sidebar.text_input("Document ID Column Name", value="note_id")
viz_dimension_option = st.sidebar.radio("Visualization Dimension", options=["2D", "3D"], index=0)
weight_exponent = st.sidebar.slider("Weight Scaling Exponent", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
size_multiplier = st.sidebar.slider("Marker Size Scaling Factor", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    st.write(f"Processing file: **{uploaded_file.name}** ...")
    
    # Determine number of rows/documents in CSV for the progress bar
    try:
        df = pd.read_csv(tmp_path)
        total_docs = df.shape[0]
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        total_docs = None

    # Temporary output file (required by the ProcessingConfig; not used further)
    temp_output = "temp_output.jsonl"

    config = ProcessingConfig(
        input_file=tmp_path,
        output_file=temp_output,
        umls_path=umls_path,
        embeddings_path=embeddings_path.strip() if embeddings_path.strip() != "" else None,
        visualization_path="visualization.html",
        text_column=text_column,
        id_column=id_column,
        parallelize=False  # Use sequential processing for the app
    )
    
    # Initialize spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize loader and processor
    loader = DataLoader()
    processor = NLPProcessor(nlp, config)
    
    # Load the embedder if provided
    if config.embeddings_path:
        st.write(f"Loading embeddings from **{config.embeddings_path}**...")
        embedder = ConceptEmbedder(config.embeddings_path)
    else:
        st.error("Embeddings file path not provided. Visualization requires embeddings.")
        embedder = None

    # Dictionaries for tracking concept frequencies and terms
    concept_raw_counts = {}   # raw frequency: total occurrences across documents
    concept_doc_counts = {}   # document frequency: in how many documents the concept appears
    concept_terms = {}        # mapping from CUI to a representative concept term (from the 'raw' field)
    doc_counter = 0
    results = []
    
    if total_docs is None:
        st.error("Could not determine total number of documents for progress updates.")
    else:
        st.write(f"Total documents to process: {total_docs}")

    # Process documents with a progress bar
    with st.spinner("Processing documents..."):
        docs_iterator = loader.from_file(config.input_file, text_column=config.text_column)
        ids_iterator = loader.from_file(config.input_file, id_column=config.id_column)
        
        progress_bar = st.progress(0)
        
        for result in processor.process_documents(ids_iterator, docs_iterator):
            if embedder:
                result['embeddings'] = embedder.embed_document(result['umls'])
            doc_counter += 1
            
            unique_cuis = set()
            for entity in result.get('umls', []):
                cui = entity['cui']
                # Update raw frequency counts
                concept_raw_counts[cui] = concept_raw_counts.get(cui, 0) + 1
                unique_cuis.add(cui)
                # Keep track of the concept term (using 'raw' text)
                if cui not in concept_terms:
                    concept_terms[cui] = entity.get('raw', '')
            for cui in unique_cuis:
                concept_doc_counts[cui] = concept_doc_counts.get(cui, 0) + 1
            
            results.append(result)
            
            if total_docs:
                progress_bar.progress(int((doc_counter / total_docs) * 100))
    
    st.success("Document processing complete!")
    
    # Calculate weights with a progress bar for each concept.
    st.write("Calculating weights for each concept...")
    weights = {}
    if concept_raw_counts:
        num_concepts = len(concept_raw_counts)
        weight_progress_bar = st.progress(0)
        for i, (cui, raw_count) in enumerate(concept_raw_counts.items()):
            df_count = concept_doc_counts.get(cui, 0)
            if df_count > 0 and doc_counter > 0:
                basic_weight = raw_count * math.log(doc_counter / df_count)
                # Exaggerate differences using the scaling exponent and multiply by the size factor
                weights[cui] = (basic_weight ** weight_exponent) * size_multiplier
            else:
                weights[cui] = 0
            weight_progress_bar.progress(int(((i + 1) / num_concepts) * 100))
        st.success("Weight calculation complete!")
    else:
        st.error("No concepts found in the processed documents!")
    
    if embedder is None:
        st.error("Embeddings were not loaded so visualization cannot be produced.")
    else:
        viz_dims = 3 if viz_dimension_option == "3D" else 2
        
        # Directly access the full embeddings dictionary
        embeddings_dict = embedder.embeddings
        
        visualizer = ConceptVisualizer()
        df_vis = visualizer.prepare_data(
            embeddings_dict=embeddings_dict,
            frequency_dict=concept_raw_counts,
            weight_dict=weights,
            term_dict=concept_terms,
            dimensions=viz_dims
        )
        fig = visualizer.create_plot(df_vis, title="Medical Concept Map", dimensions=viz_dims)
        st.plotly_chart(fig, use_container_width=True)
        
        html_bytes = fig.to_html().encode("utf-8")
        st.download_button(
            label="Download Visualization as HTML",
            data=html_bytes,
            file_name="visualization.html",
            mime="text/html"
        )
    
    os.remove(tmp_path)