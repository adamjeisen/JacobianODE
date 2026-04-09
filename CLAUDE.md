## Scientific Skills

Before starting domain-specific work, call `find_helpful_skills` to check for relevant
guidance. The following installed skills are especially relevant to this project — look for
opportunities to apply them:
- **fluidsim** — fluid dynamics simulation
- **scikit-learn** — ML modeling, preprocessing, evaluation
- **optimize-for-gpu** — GPU/CUDA optimization for training and inference
- **aeon** — time series classification, regression, and forecasting

## Jupyter Notebooks

Never use the default notebook read/edit/grep tools. Never try to read the raw `.ipynb` 
JSON directly. Two MCP servers are available for notebook work:

### notebook-mcp (structural editing)
Use for reading, editing, and organizing notebook structure:
- Use `notebook_get_outline` first to understand the structure before editing
- Use `notebook_search` to locate specific cells by keyword
- Use `notebook_edit_cell` for targeted edits
- Use `notebook_add_cell`, `notebook_delete_cell`, `notebook_move_cell` for structure changes

### jupyter-server MCP (execution & debugging)
Use for running code and inspecting results. Requires a running Jupyter server 
(user starts one with `jlab` in terminal, which runs on localhost:8888):
- Use `execute_notebook_code` to run cells and get outputs
- Use `setup_notebook` to connect to a notebook on the server
- Use `query_notebook` to inspect notebook state
- Best for: iterating on code, debugging, checking outputs, running analysis