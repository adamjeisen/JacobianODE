## Scientific Skills

Before starting domain-specific work, call `find_helpful_skills` to check for relevant
guidance. The following installed skills are especially relevant to this project — look for
opportunities to apply them:
- **fluidsim** — fluid dynamics simulation
- **scikit-learn** — ML modeling, preprocessing, evaluation
- **optimize-for-gpu** — GPU/CUDA optimization for training and inference
- **aeon** — time series classification, regression, and forecasting

## Jupyter Notebooks

Always use the notebook-mcp tools to read and edit `.ipynb` files. Never use the default 
notebook read/edit/grep tools. Specifically:
- Use `notebook_get_outline` first to understand the structure before editing
- Use `notebook_search` to locate specific cells by keyword
- Use `notebook_edit_cell` for targeted edits
- Never try to read the raw `.ipynb` JSON directly