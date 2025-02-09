from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.express as px
from umap import UMAP
import logging

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
                    frequency_dict: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """
        Prepare data for visualization.
        
        Args:
            embeddings_dict: Dictionary mapping CUIs to their embeddings
            frequency_dict: Optional dictionary mapping CUIs to their frequencies
            
        Returns:
            DataFrame with reduced dimensions and metadata
        """
        if not embeddings_dict:
            raise ValueError("No embeddings provided")
            
        # Convert embeddings to array for dimension reduction
        cuis = list(embeddings_dict.keys())
        embeddings_array = np.array(list(embeddings_dict.values()))
        
        # Reduce dimensions
        try:
            reduced_embeddings = self.reducer.fit_transform(embeddings_array)
        except Exception as e:
            logging.error(f"Error reducing dimensions: {e}")
            raise
            
        # Create DataFrame
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'cui': cuis,
        })
        
        # Add frequencies if provided
        if frequency_dict:
            df['frequency'] = df['cui'].map(lambda x: frequency_dict.get(x, 0))
        
        return df
        
    def create_plot(self, 
                   df: pd.DataFrame,
                   title: str = "Medical Concept Map") -> "plotly.graph_objects.Figure":
        """
        Create interactive plot of concepts.
        
        Args:
            df: DataFrame with reduced dimensions and metadata
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            df,
            x='x',
            y='y',
            size='frequency' if 'frequency' in df.columns else None,
            hover_data=['cui'],
            title=title,
            template='plotly_white'
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