"""
Defines main entry point for Report process.
"""

import sys
import yaml

from .execute import Execute

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Load YAML config
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Get parameters from YAML, with defaults if not specified
        options = config.get('options', {})
        topn = options.get('topn', 50)  # Default to 50 if not specified
        render = options.get('render', 'md')  # Default to 'md' if not specified
        
        # Run report with params from YAML: input file, embeddings model path, qa model path, threshold
        Execute.run(
            sys.argv[1],  # YAML config path
            topn,
            render,
            sys.argv[2] if len(sys.argv) > 2 else None,  # embeddings model path
            sys.argv[3] if len(sys.argv) > 3 else None,  # qa model path
            sys.argv[4] if len(sys.argv) > 4 else None,  # threshold
        )
