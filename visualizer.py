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
        Prepare data for visualization.
        
        Args:
            embeddings_dict: Dictionary mapping CUIs to their embeddings.
            frequency_dict: Optional dictionary mapping CUIs to their raw frequencies.
            weight_dict: Optional dictionary mapping CUIs to their scaled tfidf-like weights.
            term_dict: Optional dictionary mapping CUIs to their representative concept terms.
            dimensions: Number of dimensions (2 or 3) for visualization.
            
        Returns:
            DataFrame with reduced dimensions and metadata.
        """
        if not embeddings_dict:
            raise ValueError("No embeddings provided")
        
        cuis = list(embeddings_dict.keys())
        embeddings_array = np.array(list(embeddings_dict.values()))
        
        # Use UMAP for dimensionality reduction with the chosen number of dimensions.
        reducer = umap.UMAP(n_components=dimensions)
        try:
            reduced_embeddings = reducer.fit_transform(embeddings_array)
        except Exception as e:
            logging.error(f"Error reducing dimensions: {e}")
            raise
        
        data = {'cui': cuis, 'x': reduced_embeddings[:, 0], 'y': reduced_embeddings[:, 1]}
        if dimensions == 3:
            data['z'] = reduced_embeddings[:, 2]
        
        df = pd.DataFrame(data)
        
        if frequency_dict:
            df['frequency'] = df['cui'].map(lambda x: frequency_dict.get(x, 0))
        if weight_dict:
            df['weight'] = df['cui'].map(lambda x: weight_dict.get(x, 0))
        if term_dict:
            df['term'] = df['cui'].map(lambda x: term_dict.get(x, ''))
        
        return df
        
    def create_plot(self, 
                    df: pd.DataFrame,
                    title: str = "Medical Concept Map",
                    dimensions: int = 2) -> "plotly.graph_objects.Figure":
        """
        Create an interactive plot of concepts.
        
        Args:
            df: DataFrame with reduced dimensions and metadata.
            title: Plot title.
            dimensions: Number of dimensions (2 or 3) for visualization.
            
        Returns:
            Plotly figure object.
        """
        # Increase size_max to make differences in marker sizes more pronounced.
        if dimensions == 2:
            fig = px.scatter(
                df,
                x='x',
                y='y',
                size='weight' if 'weight' in df.columns else None,
                color='frequency' if 'frequency' in df.columns else None,
                hover_data=['cui', 'term'],
                title=title,
                template='plotly_white',
                size_max=100
            )
        else:  # For 3D visualization
            fig = px.scatter_3d(
                df,
                x='x',
                y='y',
                z='z',
                size='weight' if 'weight' in df.columns else None,
                color='frequency' if 'frequency' in df.columns else None,
                hover_data=['cui', 'term'],
                title=title,
                template='plotly_white',
                size_max=100
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