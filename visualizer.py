import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.express as px
import logging
import umap

class ConceptVisualizer:
    """Handles visualization of medical concepts and their embeddings."""
    
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
            

        reducer = umap.UMAP(n_components=dimensions, random_state=42)
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
            'top': top_strings
        }
        if dimensions == 3:
            data['z'] = reduced_embeddings[:, 2]
        
        if cluster_labels is not None:
            data['cluster'] = cluster_labels
            
        df = pd.DataFrame(data)
        
        if 'cluster' in df.columns:
            df['cluster'] = df['cluster'].astype(str)
        
        return df
        
    def create_plot(self, 
                    df: pd.DataFrame,
                    title: str = "Medical Concept Map",
                    dimensions: int = 2) -> "plotly.graph_objects.Figure":
        """
        Create an interactive plot for note embeddings.
        Assumes df has columns: 'note_id', 'x', 'y', 'top', and optionally 'z', 'cluster'.
        """
        # Define hover columns based on expected note_data columns
        hover_cols = ['note_id', 'cluster', 'top'] if 'cluster' in df.columns else ['note_id', 'top']
        
        # Define size/color based on availability 
        size_col = None 
        color_col = 'cluster' if 'cluster' in df.columns else None
        
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