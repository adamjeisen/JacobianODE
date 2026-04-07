## Jupyter Notebooks

Always use the notebook-mcp tools to read and edit `.ipynb` files. Never use the default 
notebook read/edit/grep tools. Specifically:
- Use `notebook_get_outline` first to understand the structure before editing
- Use `notebook_search` to locate specific cells by keyword
- Use `notebook_edit_cell` for targeted edits
- Never try to read the raw `.ipynb` JSON directly