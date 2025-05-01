from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.express as px
from umap import UMAP
import logging
import umap

class ConceptVisualizer:
    """Handles visualization of medical concepts and their embeddings."""
    
    def __init__(self, dimension_reducer: str = 'umap') -> None:
        """
        Initialize the visualizer.
        
        Args:
            dimension_reducer: Method to reduce dimensions ('umap' or 'tsne')
        """
        self.dimension_reducer = dimension_reducer
        self._setup_reducer()
        
    def _setup_reducer(self) -> None:
        """Configure the dimension reduction method."""
        if self.dimension_reducer == 'umap':
            self.reducer = UMAP(n_components=2, random_state=42)
        else:
            # Can add support for t-SNE or other methods here
            raise ValueError(f"Unsupported reducer: {self.dimension_reducer}")
            
    def prepare_data(self, 
                     embeddings_dict: Dict[str, np.ndarray],
                     frequency_dict: Optional[Dict[str, int]] = None,
                     weight_dict: Optional[Dict[str, float]] = None,
                     term_dict: Optional[Dict[str, str]] = None,
                     dimensions: int = 2) -> pd.DataFrame:
        """
        DEPRECATED: Prepare data for concept-level visualization.
        Use prepare_note_data for document-level visualization.
        """
        logging.warning("prepare_data is deprecated for concept maps. Use prepare_note_data.")
        # ... (Implementation can remain or be simplified/removed if no longer needed)
        # For safety, let's raise an error if called directly now
        raise NotImplementedError("prepare_data is deprecated. Use prepare_note_data.")
        
    def prepare_note_data(
        self,
        vec_arr: np.ndarray,            # shape (n_docs, dim)
        note_ids: List[str],
        top_strings: List[str],
        cluster_labels: Optional[List[int]] = None,
        dimensions: int = 2) -> pd.DataFrame:
        """
        Prepare data for note-level visualization.
        
        Args:
            vec_arr: Numpy array of document embeddings (n_docs, dim).
            note_ids: List of note IDs corresponding to vec_arr rows.
            top_strings: List of formatted top concept strings for hover data.
            cluster_labels: Optional list of cluster assignments for notes.
            dimensions: Number of dimensions (2 or 3) for visualization.
            
        Returns:
            DataFrame with reduced dimensions and metadata for notes.
        """
        if not isinstance(vec_arr, np.ndarray) or vec_arr.ndim != 2:
            raise ValueError("vec_arr must be a 2D numpy array.")
        if not (len(note_ids) == vec_arr.shape[0] and len(top_strings) == vec_arr.shape[0]):
            raise ValueError("Length mismatch between vectors, note_ids, and top_strings.")
        if cluster_labels is not None and len(cluster_labels) != vec_arr.shape[0]:
            raise ValueError("Length mismatch between vectors and cluster_labels.")
            
        # Configure reducer for the requested dimensions
        # Note: UMAP might behave slightly differently if n_components changes after init,
        #       re-initializing might be safer if switching dimensions often.
        reducer = umap.UMAP(n_components=dimensions, random_state=42) # Use local reducer
        logging.info(f"Reducing {vec_arr.shape[0]} vectors from {vec_arr.shape[1]}D to {dimensions}D using UMAP...")
        try:
            reduced_embeddings = reducer.fit_transform(vec_arr)
            logging.info("Dimensionality reduction complete.")
        except Exception as e:
            logging.error(f"Error reducing dimensions: {e}")
            raise
            
        data = {
            'note_id': note_ids,
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'top': top_strings # Column for hover data
        }
        if dimensions == 3:
            data['z'] = reduced_embeddings[:, 2]
        
        if cluster_labels is not None:
            data['cluster'] = cluster_labels
            
        df = pd.DataFrame(data)
        
        # Convert cluster labels to string for discrete color mapping if present
        if 'cluster' in df.columns:
            df['cluster'] = df['cluster'].astype(str)
        
        return df
        
    def create_plot(self, 
                    df: pd.DataFrame,
                    title: str = "Medical Concept Map",
                    dimensions: int = 2) -> "plotly.graph_objects.Figure":
        """
        Create an interactive plot.
        (Works for both concept and note level if df has expected columns)
        """
        # Define hover columns based on expected note_data columns
        hover_cols = ['note_id', 'cluster', 'top'] if 'cluster' in df.columns else ['note_id', 'top']
        
        # Define size/color based on availability 
        size_col = 'weight' if 'weight' in df.columns else None 
        # Use cluster for color if available, otherwise fallback
        color_col = 'cluster' if 'cluster' in df.columns else ('frequency' if 'frequency' in df.columns else None)
        
        # Determine plot function and axes based on dimensions
        plot_func = px.scatter_3d if dimensions == 3 else px.scatter
        axis_args = {'x':'x', 'y':'y'}
        if dimensions == 3:
            axis_args['z'] = 'z'

        fig = plot_func(
            df,
            **axis_args,
            size=size_col, 
            color=color_col,
            hover_data=hover_cols,
            title=title,
            template='plotly_white',
            size_max=60
        )
        
        # Original trace update remains useful
        fig.update_traces(
            marker=dict(sizemin=5),
            selector=dict(mode='markers')
        )
        
        return fig
        
    def save_plot(self, 
                  fig: "plotly.graph_objects.Figure",
                  output_path: str) -> None:
        """Save plot to HTML file."""
        try:
            fig.write_html(output_path)
            logging.info(f"Plot saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
            raise 